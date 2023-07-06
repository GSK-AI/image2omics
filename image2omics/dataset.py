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

import functools
import logging
import random
import sys
from typing import Iterable, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from image2omics.transforms import (
    ComposeAdapter,
    build_data_transformation,
)
from image2omics import dataset_utils, utils

logging.basicConfig(level=logging.INFO)


class PlateDataset(Dataset):
    """A Plate dataset - a collection of Well datasets"""

    def __init__(
        self,
        manifest: str,
        tile_coordinates_manifest: str,
        labels: Optional[str] = None,
        labels_dtype: Optional[torch.dtype] = None,
        transform: Optional[Union[dict, transforms.Compose, ComposeAdapter]] = None,
        channels: Optional[Iterable[str]] = None,
        tile_size: Optional[Iterable[int]] = None,
        max_tiles_per_fov: int = 99999,
        cache_pointers: Optional[bool] = True,
        label_type: str = dataset_utils.SINGLE_LABEL
    ):
        """Constructor
        Parameters:
        -----------
        manifest: str
            A manifest file of all input HDF-5 files for this plate
        tile_coordinates_manifest: str
            A manifest file of all tile coordinates files, each corresponding
            to one input HDF-5 file
        labels: Optional[str]
            Path to CSV file containing controls information, by default None.
            The file should follow the controls plate map format. See User Instructions
            for detailed examples. When set to None, the __getitem__ method will only
            return images from manifest.
        labels_dtype : Optional[torch.dtype]
            torch datatyple for labels, by default None
            This is used to modify labels data type to work with specific torch losses,
            i.e. torch.int64 for CrossEntroyLoss. This argument is only used when
            the argument labels is provided.
        transforms: dict, optional
            Dictionary to be used as kwargs in build_data_transformation
            to create data augmentatioon transformation. When None, no data augmentation
            is used.
        channels: Iterable[str]
            Iterable of channels in the requested order
        tile_size: Iterable[int]
            Tile sizes for cropping images into tiles, by default None
            If None, no tiling is done
        label_type: str
            Label handling strategy. Options are:
            SINGLE_LABEL = "singlelabel"
            CONTINUOUS_LABEL = "multiclass"
            INDEPENDENT_LABEL = "multitask"
        """

        self.num_elements = 0
        self.tile_size = tile_size
        self.processing = None
        self.barcode = None
        self.instrument_type = None
        self.channel_order = channels
        self.manifest = manifest
        self.tile_manifest = tile_coordinates_manifest
        self.max_tiles_per_fov = max_tiles_per_fov    
        self.is_multilabel = (label_type != dataset_utils.SINGLE_LABEL)
        self.labels, self.label_cols = self._map_labels_from_file(labels)
        self.labels2idx = self._create_labels2idx(label_type)
        self.labels_dtype = self._create_labels_dtype(labels_dtype)
        self.cache_pointers = cache_pointers        


        if isinstance(transform, dict):
            self.transform = build_data_transformation(**transform)
        else:
            self.transform = transform

        self._check_channel_ordering(
            self.channel_order
        )  # Get channels unique names, order not guaranteed
        self.channels = utils.get_unique_channel_names(self.channel_order)
        logging.info(f"Channels ordering: {self.channel_order}")
        logging.info(f"Channels unique: {self.channels}")

        # Check all the datasets for matching num FOVs and same barcode
        with open(manifest, "r") as in_f:
            for h5file in in_f:
                self._check_h5_file(h5file.strip())

        self._cache_data()

    def _create_labels_dtype(self, labels_dtype: torch.dtype):
        if self.labels is None or self.is_multilabel:
            return None

        if labels_dtype is not None:
            return labels_dtype

        if self.label_is_continuous():
            return torch.float32
        else:
            return torch.int64            

    def _create_labels2idx(self, label_type: str):
        if self.labels is None:
            return self.labels
        labels2idx = dataset_utils.create_labels2idx(self.labels[self.label_cols], label_type)

        return labels2idx

    def label_is_continuous(self):
        return self._all_is_number(self.labels["LABEL"].unique())

    def _all_is_number(self,list_: Iterable):
        """Returns True if all element in list is a number"""
        if len(list_) <= 0:
            return False

        return all(dataset_utils._is_number(elem) for elem in list_)

    def _map_labels_from_file(self, labels: Optional[str]) -> pd.DataFrame:
        if labels is None:
            return (None, None)
        # Create a flat file with columns ["ROW", "COLUMN", "LABEL"]
        df = controls_parser.map_labels_from_file(labels=labels)
        df = df[df["LABEL"].astype(bool)]

        # Now create the ROW_COLUMN column for fast search
        df = df.set_index(
            df[["ROW", "COLUMN"]].astype(str).agg("_".join, axis=1).values
        )
        lcols = "LABEL"
        if self.is_multilabel:
            if len(df[df["LABEL"].str.contains(dataset_utils.SPLIT_CHAR)]) > 0:
                lcols = []
                df_tmp = df["LABEL"].str.split(dataset_utils.SPLIT_CHAR,expand=True)
                self.num_multilabels = len(df_tmp.columns)
                for i in range(self.num_multilabels):
                    df[f"LABEL_{i}"] = df_tmp[i]
                    lcols.append(f"LABEL_{i}")

        return df, lcols

    def _check_channel_ordering(self, channel_order: Union[list, List[list]]):
        # check for nested list case
        if any(isinstance(c, list) for c in channel_order):
            for c in channel_order:
                if any(isinstance(c_, list) for c_ in c):
                    logging.error(
                        f"Provided channel ordering ({channel_order}) has depth > 2. "
                        + "Currently only supports channel ordering with depth <= 2."
                    )
                    sys.exit(1)

            channel_length = [len(c) for c in channel_order]
            if any(l != channel_length[0] for l in channel_length):
                logging.error(
                    "Provided sublists in channel must have the same length. "
                    + f"Found lengths of {channel_length} instead."
                )
                sys.exit(1)

    def _check_h5_file(self, h5file: str):
        """Check the H5 file and extract the number of FOVs and check it for consistency
        aross the files
        Parameters:
        -----------
        h5file: str
            HDF-5 file for particular Well
        """

        # Use this rather than context as this way has lower overhead
        in_h5 = h5py.File(h5file, mode="r", rdcc_nbytes=0, rdcc_nslots=0)

        # Use full image if no tile size is provided
        for fov in in_h5[f"{self.channels[0]}"].keys():
            if fov == "all_fovs":
                continue
            # because self.min_fov may not exist in some wells!
            img_shape = in_h5[f"{self.channels[0]}/{fov}"].shape
            break

        if self.tile_size is None:
            self.tile_size = img_shape

        # If we have not set it yet, set the barcode from the first file
        if self.barcode is None:
            self.barcode = in_h5.attrs["PLATE_BARCODE"]

        # Barcodes are the same in all files
        assert in_h5.attrs["PLATE_BARCODE"] == self.barcode, (
            f"Incorrect {in_h5.attrs['PLATE_BARCODE']} in {h5file}, expecting"
            f" {self.barcode}"
        )
        in_h5.close()

    def __len__(self) -> int:
        return self.num_elements

    def resample(self, tile_info_fov, fov_ids):
        new_tile_info_fov = None
        for fov in fov_ids:
            fov_indices = np.nonzero(tile_info_fov[:, 0, 1] == fov)[0]
            num_tiles_fov = len(fov_indices)
            if num_tiles_fov > self.max_tiles_per_fov:
                fov_indices = random.sample(fov_indices, self.max_tiles_per_fov)
                if new_tile_info_fov is None:
                    new_tile_info_fov = tile_info_fov[fov_indices]
                else:
                    new_tile_info_fov = np.vstack(
                        (new_tile_info_fov, tile_info_fov[fov_indices])
                    )
        if new_tile_info_fov is None:
            new_tile_info_fov = tile_info_fov
        return new_tile_info_fov, new_tile_info_fov.shape[0]

    def _cache_data(self):
        """Generate a list of tile information a dictionary of image
        h5 cache.
        Each element in the list is a TileCacheEntry object with attributes
        ["fov_index", "row_col", "tile_index", "coordinate"].
        Each key-value pairs in the dictionary maps a well, identified by its
        "row_col", to a h5py.File object.
        """

        tile_coordinates_cache = []
        well_coordinates_cache = {}
        h5_fp_cache = {}
        skipped_wells = []

        with open(self.tile_manifest, "r") as in_f:
            tile_coordinates_files = in_f.readlines()

        current_tile_counter = 0
        well_counter = 0
        for h5_path in sorted(tile_coordinates_files):  # iterate over each well
            tmp_h5 = h5py.File(h5_path.strip(), mode="r")
            row = tmp_h5.attrs["ROW"]
            column = tmp_h5.attrs["COLUMN"]
            row_col = "_".join([str(c) for c in [row, column]])

            if self.labels is not None and row_col not in set(self.labels.index):
                tmp_h5.close()
                continue

            # get all fov numbers
            fov_list = [f for f in tmp_h5.keys() if f != "all_fovs"]
            fov_ids = [int(fov.strip().split("_")[-1]) for fov in fov_list]
            # get tiles from all fovs at once
            # all_fovs is a np.array of shape [N, M, C]
            # where
            # N = total number of tiles in well
            # M = 2, number of coordinates (x,y)
            # C = 3, where C=0: coordinates; C=1: col 1 = FOV, col 2 = tile index
            # for well; C=2: col 1 = well ROW, col 2 = well COL
            # load tile coordinates
            tile_coords = tmp_h5["all_fovs/tile_coords"][...]
            # load fov ids and tile indices
            fovs_indices = tmp_h5["all_fovs/fovs_indices"][...]
            # load well ROWs and COLUMNs
            well_coords = tmp_h5["all_fovs/well_coords"][...]
            tile_info_fov = np.stack((tile_coords, fovs_indices, well_coords), axis=2)
            # only proceed if well has tiles
            num_tiles = tile_info_fov.shape[0]
            if num_tiles <= 0:
                skipped_wells.append(row_col)
                continue

            # resample tiles per fov, if needed
            tile_info_fov, num_tiles = self.resample(tile_info_fov, fov_ids)
            # append to plate tiles coordinate cache np.array
            if len(tile_coordinates_cache) <= 0:
                tile_coordinates_cache = tile_info_fov
            else:
                tile_coordinates_cache = np.vstack(
                    (tile_coordinates_cache, tile_info_fov)
                )
            # create map between well id and indices in the tile cache
            well_coordinates_cache[well_counter] = (
                row_col,
                current_tile_counter,
                current_tile_counter + num_tiles,
            )
            current_tile_counter += num_tiles
            well_counter += 1

            # add file pointer to cache
            if self.cache_pointers:
                h5_fp_cache[row_col] = h5py.File(
                    tmp_h5.attrs["INPUT_POINTER"],
                    mode="r",
                    rdcc_nbytes=0,
                    rdcc_nslots=0,
                )
            else:
                h5_fp_cache[row_col] = tmp_h5.attrs["INPUT_POINTER"]

            tmp_h5.close()

        self.num_elements = len(tile_coordinates_cache)
        self.tile_coordinates_cache = tile_coordinates_cache
        self.well_to_coordinates_cache = well_coordinates_cache
        self.num_wells = len(well_coordinates_cache)
        self.h5_fp_cache = h5_fp_cache

        logging.info(f"Caching coordinates for each tile: Done")
        logging.info(
            f"NUM_IMAGES={self.num_elements},"
            + f" BARCODE={self.barcode},"
            + f" SKIPPED_WELLS={skipped_wells}"
        )

    def __getitem__(self, index: int) -> dict:
        """Returns transformed image and a label"""
        tile_info = self.tile_coordinates_cache[index].astype("int")
        # tile_info[:,0] = (x, y)
        # tile_info[:,1] = (FOV, tile index)
        # tile_info[:,2] = (well ROW, well COL)
        row_col = "_".join([str(c) for c in tile_info[:, 2]])
        h5_file = self.h5_fp_cache[row_col]
        fov_index = tile_info[0, 1]
        x_coord, y_coord = int(tile_info[0, 0]), int(tile_info[1, 0])
        tile_index = tile_info[1, 1]

        with utils.H5Manager(h5_file) as h5_file:
            metadata = {}
            metadata.update(dict(h5_file.attrs))
            metadata["FOV"] = fov_index
            metadata["TILE"] = tile_index
            metadata["CHANNELS"] = self.channels
            metadata["CHANNEL_ORDER"] = str(self.channel_order)

            def _channel2img(channel: str) -> Image:
                """Helper function to convert channel name to corresponding image
                Note that this also adds information to the metadata dictionary outside
                of this function.
                Caching is used here to reduce I/O burden
                """
                channel_dataset_name = f"{channel}/fov_{fov_index}"

                # Add channel metadata
                channel_metadata = dict(h5_file[channel_dataset_name].attrs)
                metadata.update(
                    {f"{channel}_{k}": v for k, v in channel_metadata.items()}
                )

                channel_array = h5_file[channel_dataset_name][
                    x_coord : x_coord + self.tile_size[0],
                    y_coord : y_coord + self.tile_size[1],
                ]
                return channel_array

            def _channeltree2imgtree(channel_tree, img_tree):
                """Recursively build img_tree data structure that follows the same structure
                as channel_tree. Each image image img_tree maps to a channel name in
                channel_tree.
                Note that for the case where channel_tree has a depth of 0 (['a','b','c']),
                this function returns a img_tree of depth 1 ([[a_img,b_img,c_img]]).
                """
                img = []
                for c in channel_tree:
                    if not isinstance(c, list):
                        img.append(_channel2img(c))
                    else:
                        img.append(_channeltree2imgtree(c, []))
                img_tree.extend(img)
                return img_tree

            img_tree = _channeltree2imgtree(self.channel_order, [])
            image = self._merge_channels(img_tree)

            metadata = {
                k: v.decode("utf-8") if isinstance(v, np.bytes_) else v
                for k, v in metadata.items()
            }

            output = {"inputs": image, "metadata": metadata}
            if self.labels is not None:
                row_col = "_".join(
                    [str(metadata[m]) for m in ["ROW", "COLUMN"]]
                )
                if self.is_multilabel:
                    labels_list = self.labels.loc[row_col, self.label_cols].to_list()
                else:
                    labels_list = self.labels.loc[row_col, self.label_cols]
                output["labels"] = torch.tensor(
                    self.labels2idx[labels_list],
                    dtype=self.labels_dtype,
                )

        return output

    def _merge_channels(
        self, images_tuple: Union[List[Image.Image], List[List[Image.Image]]]
    ):
        if not isinstance(images_tuple[0], (list, tuple)):
            images_tuple = [images_tuple]

        images = [np.stack(channels, axis=-1) for channels in images_tuple]
        if self.transform:
            images = [self.transform(img=image) for image in images]

        stack_op = torch.stack if isinstance(images[0], torch.Tensor) else np.stack
        return stack_op(images, axis=0).squeeze(0)

    @staticmethod
    def collate_fn(batch):
        collated_batch = {k: [item[k] for item in batch] for k in batch[0]}
        collated_batch.update(
            {
                k: torch.stack(v)
                for k, v in collated_batch.items()
                if k not in ["metadata"]
            }
        )
        return collated_batch

    def get_labels(self) -> list:
        labels = []
        for i in range(len(self.tile_coordinates_cache)):
            tile_info = self.tile_coordinates_cache[i].astype("int")
            row_col = "_".join([str(c) for c in tile_info[:, 2]])
            labels.append(self.labels2idx[self.labels.loc[row_col, self.label_cols].to_list()])
        return labels


class WellDataset(PlateDataset):
    """A dataset where each element is a bag of tiles from each well
    This dataset is used for multiple instance learning settings.
    """

    def __init__(self, num_tiles_per_well: int = 24, **kwargs):
        """Constructor
        Parameters:
        -----------
        num_tiles_per_well: float
            Number of tiles to return for each __getitem__ call
        kwargs:
            Same arguments as PlateDataset
        """
        super(WellDataset, self).__init__(**kwargs)
        self.num_tiles_per_well = num_tiles_per_well
        self.num_elements = len(self.h5_fp_cache)

    def __getitem__(self, index: int) -> dict:
        """Returns transformed image and a label"""
        image_bag = []

        (row_col, init_index, end_index) = self.well_to_coordinates_cache[index]
        all_tile_indices = [n for n in range(init_index, end_index)]
        num_tiles = len(all_tile_indices)
        if num_tiles > self.num_tiles_per_well:
            all_tile_indices = random.sample(all_tile_indices, self.num_tiles_per_well)
        elif num_tiles < self.num_tiles_per_well:
            all_tile_indices = random.choices(
                all_tile_indices, k=self.num_tiles_per_well
            )

        all_tile_info = self.tile_coordinates_cache[all_tile_indices].astype("int")

        h5_file = self.h5_fp_cache[row_col]
        with utils.H5Manager(h5_file) as h5_file:
            for tile_info in all_tile_info:
                # tile_info[:,0] = (x, y)
                # tile_info[:,1] = (FOV, tile index)
                # tile_info[:,2] = (well ROW, well COL)
                fov_index = tile_info[0, 1]
                x_coord, y_coord = int(tile_info[0, 0]), int(tile_info[1, 0])
                tile_index = tile_info[1, 1]

                metadata = {}
                metadata.update(dict(h5_file.attrs))
                metadata["FOV"] = fov_index
                metadata["TILE"] = tile_index
                metadata["CHANNELS"] = self.channels
                metadata["CHANNEL_ORDER"] = str(self.channel_order)

                def _channel2img(channel: str) -> Image:
                    """Helper function to convert channel name to corresponding image
                    Note that this also adds information to the metadata dictionary outside
                    of this function.
                    Caching is used here to reduce I/O burden
                    """
                    channel_dataset_name = f"{channel}/fov_{fov_index}"

                    # Add channel metadata
                    channel_metadata = dict(h5_file[channel_dataset_name].attrs)
                    metadata.update(
                        {f"{channel}_{k}": v for k, v in channel_metadata.items()}
                    )

                    channel_array = h5_file[channel_dataset_name][
                        x_coord : x_coord + self.tile_size[0],
                        y_coord : y_coord + self.tile_size[1],
                    ]
                    return channel_array

                def _channeltree2imgtree(channel_tree, img_tree):
                    """Recursively build img_tree data structure that follows the same structure
                    as channel_tree. Each image image img_tree maps to a channel name in
                    channel_tree.
                    Note that for the case where channel_tree has a depth of 0 (['a','b','c']),
                    this function returns a img_tree of depth 1 ([[a_img,b_img,c_img]]).
                    """
                    img = []
                    for c in channel_tree:
                        if not isinstance(c, list):
                            img.append(_channel2img(c))
                        else:
                            img.append(_channeltree2imgtree(c, []))
                    img_tree.extend(img)
                    return img_tree

                img_tree = _channeltree2imgtree(self.channel_order, [])
                image = self._merge_channels(img_tree)
                image_bag.append(image)

            stack_op = (
                torch.stack if isinstance(image_bag[0], torch.Tensor) else np.stack
            )

            image_bag = stack_op(image_bag, axis=0).squeeze(0)
            metadata = {
                k: v.decode("utf-8") if isinstance(v, np.bytes_) else v
                for k, v in metadata.items()
            }

            output = {"inputs": image_bag, "metadata": metadata}
            if self.labels is not None:
                row_col = "_".join(
                    [str(metadata[m]) for m in ["ROW", "COLUMN"]]
                )
                if self.is_multilabel:
                    labels_list = self.labels.loc[row_col, self.label_cols].to_list()
                else:
                    labels_list = self.labels.loc[row_col, self.label_cols]
                output["labels"] = torch.tensor(
                    self.labels2idx[labels_list],
                    dtype=self.labels_dtype,
                )

        return output

    def get_labels(self) -> list:
        return [
            self.labels2idx[self.labels.loc[row_col,self.label_cols]]
            for row_col in self.h5_fp_cache.keys()
        ]


class MultiPlateDataset(Dataset):
    """Dataset for multiple plates"""

    def __init__(
        self,
        manifest: Union[str, list],
        tile_coordinates_manifest: Union[str, list],
        labels: Optional[Union[str, list]] = None,
        dataset_obj: Dataset = PlateDataset,
        **platedataset_kwargs,
    ):
        """Constructor
        Parameters are the same as PlateDataset, except manifest,
        tile_coordinates_manifest and labels all become lists
        Parameters:
        -----------
        manifest: Union[str, list]
            Path to the image manifest file of one plate, or a list of manifest files,
            each containing all input HDF-5 files for one plate
        tile_coordinates_manifest: Union[str, list]
            Path to to the tile coordinates manifest file of one plate,
            or a list of tile coordinates manifest files for each plate.
            In the latter case, the order of plates should be the same as in manifest
        labels: Optional[Union[str, list]]
            Path to the CSV file that contains control label information of one plate,
            or a list of paths to such CSV file for each plate, by default is None.
            The CSV file should follow the controls plate map format.
            See User Instructions for detailed examples.
            When set to None, the __getitem__ method will return images from manifest
            without control labels.
        """

        # convert to lists
        create_labels2idx = True
        if isinstance(manifest, str):
            manifest = [manifest]
        if isinstance(tile_coordinates_manifest, str):
            tile_coordinates_manifest = [tile_coordinates_manifest]
        if isinstance(labels, str):
            labels = [labels]
        elif labels is None:
            create_labels2idx = False
            labels = [None for _ in range(len(manifest))]

        # instantiate the PlateDatasets
        self.plate_dataset_list = []
        for plate_manifest, plate_tile_manifest, plate_labels in zip(
            manifest, tile_coordinates_manifest, labels
        ):
            self.plate_dataset_list.append(
                dataset_obj(
                    manifest=plate_manifest,
                    tile_coordinates_manifest=plate_tile_manifest,
                    labels=plate_labels,
                    **platedataset_kwargs,
                )
            )

        # replace the labels2idx (if exists) of each PlateDataset with a global one
        # get the label2idx singleton
        self.labels2idx = None
        if create_labels2idx:
            self.labels2idx = dataset_utils.create_labels2idx(None,None)

        # a list of the largest data index of each PlateDataset when they are
        # concatenated sequentially
        # for example: if plate_size_list is [1000,950,1100],
        # self.plate_largest_index_list would be [999,1949,3049]
        plate_size_list = [len(d) for d in self.plate_dataset_list]
        self.plate_largest_index_list = np.cumsum(plate_size_list) - 1


    def __len__(self) -> int:
        total_length = self.plate_largest_index_list[-1] + 1
        return total_length

    def __getitem__(self, index: int) -> dict:
        """Get the data from the corresponding PlateDataset"""
        for i, largest_index in enumerate(self.plate_largest_index_list):
            if index <= largest_index:
                if i == 0:
                    local_index = index
                else:
                    local_index = index - self.plate_largest_index_list[i - 1] - 1
                return self.plate_dataset_list[i][local_index]
        raise IndexError(
            f"Index {index} out of range. Largest possible index is"
            + f" {self.plate_largest_index_list[-1]+1}"
        )

    @staticmethod
    def collate_fn(batch):
        collated_batch = {k: [item[k] for item in batch] for k in batch[0]}
        collated_batch.update(
            {
                k: torch.stack(v)
                for k, v in collated_batch.items()
                if k not in ["metadata"]
            }
        )
        return collated_batch

    def get_labels(self):
        return [
            label
            for plate_dataset in self.plate_dataset_list
            for label in plate_dataset.get_labels()
        ]

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, embed_size, data_path):
        df = pd.read_csv(data_path)
        # store the inputs and outputs
        self.X = df[[f"FEAT_{i}" for i in range(embed_size)]].values.astype('float32')
        self.y = df['target'].values.astype('float32')
        self.metadata = df[['PLATE_BARCODE','COLUMN','ROW','label']]
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        output = {"inputs": self.X[idx], "metadata": self.metadata.iloc[idx].to_dict(), "labels": self.y[idx]}
        return output