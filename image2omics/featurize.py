"""
Copyright 2023 Rahil Mehrizi, Cuong Nguyen, GSK plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from scipy.special import expit, softmax
from torch import nn
from tqdm import tqdm

import image2omics.utils as utils
from image2omics.aggregate import aggregate_array
from image2omics.torch_utils import to_numpy

logging.basicConfig(level=logging.INFO)

AGG_MODE_TO_COL = {
    "well": ["ROW", "COLUMN", "PLATE_BARCODE"],
    "fov": ["ROW", "COLUMN", "FOV", "PLATE_BARCODE"],
    "tile": ["ROW", "COLUMN", "FOV", "PLATE_BARCODE", "TILE"],
}
COL_TO_AGG_OP = {"PLATE_BARCODE": "first"}

class HookModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_layers: list, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.selected_out = defaultdict(dict)
        self.model = model

        for name, layer in self.model.named_modules():
            if name in self.output_layers:
                layer.register_forward_hook(self.forward_hook(name))

    def forward_hook(self, layer_name: str):
        def hook(module, input, output):
            self.selected_out[layer_name][str(input[0].get_device())] = output

        return hook

    def forward(self, x):
        out = self.model(x)
        return self.selected_out

def main(
    config_path: str,
    input_manifest_list: list,
    tile_coordinates_manifest_list: list,
    checkpoint_path: str,
    output_base: str,
    split_seed: int,
    model_ind: int = 0,
    config_key: str = "featurization",
) -> None:
    """Driver for the featurization
    Parameters:
    ----------
    config_path: str
        Path to featurization config file
    input_manifest_list: list
        Path to the manifests of input ICF normalized files. 
    tile_coordinates_manifest_list: list
        Path to the manifests of tile coordinates files, each corresponding to one ICF h5.
    checkpoint_path: str
        Path to the checkpoint of a saved model
    output_base: str
        Output base directory
    split_seed: str
        split seed
    config_key: str
        Dictionary key to be used for loading config param.
        Default is "featurization".
    """
    logging.info("Begin featurization module")
    os.makedirs(output_base, exist_ok=True)

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    config = full_config[config_key]

    utils.set_seed(full_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_manifest_list = utils.split_manifest_by_plate(
            manifest=input_manifest_list,
            filename_prefix="icf",
            output_dir=output_base,
            barcodes=full_config["barcodes"],
        )

    tile_coordinates_manifest_list = utils.split_manifest_by_plate(
            manifest=tile_coordinates_manifest_list,
            filename_prefix="tile",
            output_dir=output_base,
            barcodes=full_config["barcodes"],
        )

    config["dataloaders"]["dataset"].update(
        dict(
            manifest=input_manifest_list,
            tile_coordinates_manifest=tile_coordinates_manifest_list,
        )
    )
    dataloader = hydra.utils.instantiate(config["dataloaders"])
    tta_ops = hydra.utils.instantiate(config["tta_ops"])

    logging.info(f"Loading trained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = hydra.utils.instantiate(config["model"])
    _ = model(next(iter(dataloader)))
    model.load_state_dict(checkpoint["state_dict"])

    # get logits configutation, if available
    logits = False
    if "featurize_logits" in config:
        logits = config["featurize_logits"]

    logging.info("Generate features...")
    feature, metadata = predict(
        model=model,
        dataloader=dataloader,
        model_ind=int(model_ind),
        device=device,
        logits=logits,
        tta_ops=tta_ops,
    )
    logging.info("Generate features...DONE")

    split_seed = int(split_seed)
    logging.info(f"saving results of model {split_seed}")
    save_features_metadata_df(feature, metadata, output_base, model_ind, split_seed)


def save_features_metadata_df(
    features: Dict[str, List], metadata: List[dict], output_base: str, model_ind: str, split_seed
) -> None:
    """Helper function for saving features and metadata for each model
    Parameters
    ----------
    features : Dict[str, List]
        a dictiory containing 2D Array to be aggregated with shape (N, F) for each features
    metadata : List[dict]
        Metadata of the array with length N. Each element is a
        dictionary containing metadata for each row of the array.
    output_base : str
        Path to where the output will be saved
    model_ind : str
        model index
    split_seed: int
        split seed
    """
    for key in features.keys():
        if len(features[key]) != len(metadata):
            logging.error(
                f"Number of elements in feature ({len(features[key])}) "
                + f"and metadata ({len(metadata)}) are not equal."
            )
            sys.exit(1)

    df = pd.DataFrame(metadata)

    for key in features.keys():
        if features[key].ndim > 2:
            features[key] = features[key].reshape(features[key].shape[0], -1)
        feature_col = [f"{key}_{i}" for i in range(len(features[key].T))]
        df = pd.concat([df, pd.DataFrame(features[key], columns=feature_col)], axis=1)

    df.to_csv(
        os.path.join(output_base, f"metadata_features_model_{split_seed}.csv"),
        index=False,
    )

def predict(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    model_ind: int = 0,
    logits: bool = True,
    verbose: bool = True,
    tta_ops: Optional[List[Callable]] = None,
    device: Optional[torch.device] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    **kwargs,
) -> Tuple[np.ndarray, List[dict]]:
    """Run forward pass of model on images
    The function returns outputs:
    - A np.ndarray with shape (num_image, num_outputs)
    - List of metadata associated with each image
    The function also take in kwargs, which is used as kwargs
    for the model.forward() method.
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    dataloader : torch.utils.data.DataLoader
        DataLoader that returns a batch as dictionary with keys ['inputs', 'metadata']
    logits: bool
        Model output is logit
    tta_ops: Optional[List[Callable]]
        List of augmentation function operating on batches of images with dimensions [B,..., C, H, W]
        Individual functions can be taken from aiml_cell_imaging.featurization.tta
    Returns
    -------
    Tuple[np.ndarray, List[dict]]
        Tuple of (model ouputs, metadata)
    """
    if tta_ops is None or tta_ops == []:
        tta_ops = [lambda x: x]

    model.eval()
    model = model.to(device)

    # model.named_modules is a generator of tuples and that is why we need the extra [0] index to capture the name of the last layer
    last_layer_name = list(model.named_modules())[-1][0]
    output_layers = [last_layer_name] 
    model = HookModelWrapper(model, output_layers).to(device)

    output = defaultdict(list)
    metadata = []

    if logits and last_layer_name not in output_layers:
        logging.error(
            f"to calculate logits, the last layer, which is {last_layer_name} should be added to the output_layers"
        )
        sys.exit(1)

    with torch.no_grad():
        # sample model weights
        if verbose:
            dataloader = tqdm(dataloader, total=len(dataloader))
        for i, batch in enumerate(dataloader):
            x, meta = batch["inputs"], batch["metadata"]
            # adding model index to metadata dict
            meta = [dict(item, MODEL=model_ind) for item in meta]
            x = x.to(device)
            # DataParallel will produce a TypeError when batch_size and num_gpus hits an edge case.
            # See: https://github.com/pytorch/pytorch/issues/15161
            # This behavior typically happens with the last batch of the dataset, depending on its size.
            # The try-except below mitigates this problem. Note that the TypeError is produced from a
            # single worker, so doing this does not skip the last batch
            try:
                output_tta = [to_numpy(model(f(x), **kwargs)) for f in tta_ops]
            except TypeError as e:
                logging.error(str(e))
                continue

            # sort output_tta based on the gpu index and then concatenate across gpus
            output_tta = [
                {k: dict(sorted(d[k].items())) for k in d.keys()}
                for d in output_tta
            ]
            output_tta = [
                {k: np.concatenate(list(d[k].values())) for k in d.keys()}
                for d in output_tta
            ]

            output_tta = {
                k: np.mean(np.stack([d[k] for d in output_tta], axis=-1), axis=-1)
                for k in output_tta[0].keys()
            }

            for key, val in output_tta.items():
                output[key].append(val)
            metadata.extend(meta)

    output = {key: np.concatenate(output[key], axis=0) for key in output.keys()}
    for key in output.keys():
        if output[key].ndim == 1:
            output[key] = np.expand_dims(output[key], axis=-1)

    return output, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="JSON config file", required=True)
    parser.add_argument(
        "-i", "--input-manifest-list", help="Input manifest", required=True
    )
    parser.add_argument(
        "-t",
        "--tile-coordinates-manifest-list",
        help="Tile coordinates file manifest",
        required=True,
    )
    parser.add_argument("--ckpt", help="Model checkpoint", required=False, default=None)
    parser.add_argument(
        "-o", "--output-base", help="Output base directory", required=True
    )
    parser.add_argument("-s", "--seed", help="Checkpoint seed", required=True)
    parser.add_argument(
        "-k", "--config-key", help="Key in config file to use", required=False
    )
    args = parser.parse_args()

    kwargs = {}
    if args.config_key:
        kwargs["config_key"] = args.config_key

    # Force a Slurm job
    try:
        slurm_array_job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    except KeyError:
        logging.error(
            "No SLURM_ARRAY_TASK_ID available, perhaps this is not an array job"
            " - will only run single job"
        )

    main(
        args.config,
        args.input_manifest_list,
        args.tile_coordinates_manifest_list,
        args.ckpt,
        args.output_base,
        args.seed,
        **kwargs,
    )
