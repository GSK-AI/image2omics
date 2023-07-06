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

import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import torch


def set_seed(random_seed: int) -> None:
    """Sets seed and turns pytorch non-deterministic (as far as one can)"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def split_manifest_by_plate(
    manifest: str, filename_prefix: str, output_dir: str, barcodes: List[str]
) -> list:
    """Split the h5 manifest of multiple plates to several manifests,
    each corresponding to one plate

    Parameters
    ----------
    manifest : str
        path to the manifest file containing all plates
    filename_prefix : str
        filename prefix that will be added to the single plate manifests,
        such as "icf" or "tile"
    output_dir : str
    barcodes: List[str]

    Returns
    -------
    list
        a list of single plate manifest files
    """
    with open(manifest, "r") as f:
        h5_files = [line.strip() for line in f.readlines()]

    plate_manifests_dict = defaultdict(list)
    for h5_file in h5_files:
        plate_barcode = get_info_from_h5_path(h5_file)["PLATE_BARCODE"]
        if plate_barcode in barcodes:
            plate_manifests_dict[plate_barcode].append(h5_file)

    # Return a list of manifest with same orders as barcodes
    plate_manifests_list = []
    for plate_barcode in barcodes:
        h5_file_list = plate_manifests_dict[plate_barcode]
        plate_manifest_file = os.path.join(
            output_dir, f"{filename_prefix}_manifest_{plate_barcode}.txt"
        )

        with open(plate_manifest_file, "w") as f:
            f.write("\n".join(h5_file_list))
        plate_manifests_list.append(plate_manifest_file)
    return plate_manifests_list

def get_info_from_h5_path(
    h5_path: str, attrs: list = ["ROW", "COLUMN", "PLANE", "PLATE_BARCODE"]
) -> dict:
    """Get information from well image h5 path

    Information may include row, column, plane, timestamp and plate barcode

    Parameters
    ----------
    h5_path : str
        path to the h5 file
    attrs: list
        list of attributes to read from hdf5 file

    Returns
    -------
    dict
        row, column, plane, timestamp and plate barcode of that well as a dict
    """
    h5 = h5py.File(h5_path, "r")
    info_dict = {}
    for attr in attrs:
        if attr in h5.attrs:
            info_dict[attr] = h5.attrs[attr]
        else:
            raise ValueError(f"Unknown attribute {attr}")

    h5.close()

    return info_dict

class H5Manager(object):
    def __init__(self, h5file):
        self.h5_file = h5file
        self.cached = isinstance(h5file, h5py.File)

    def __enter__(self):
        if not self.cached:
            self.h5_file = h5py.File(
                self.h5_file, mode="r", rdcc_nbytes=0, rdcc_nslots=0
            )
        return self.h5_file

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self.cached:
            self.h5_file.close()

def get_unique_channel_names(channels: Union[List, List[List]]) -> List:
    """Get unique channel names from the channels provided by users

    This function is used to parse nested list structures of
    channel names and return a list of unique and
    alphabetically-sorted channel names

    Parameters
    ----------
    channels : Union[List, List[List]]
        User-provided channel names

    Returns
    -------
        unique:
            list of unique channel names
    """

    def flatten(ll):
        flat_list = []
        for i in ll:
            if not isinstance(i, list):
                flat_list.append(i)
            else:
                flat_list.extend(flatten(i))
        return flat_list

    all_channels = flatten(channels)
    return sorted(list(set(all_channels)))

class Singleton(type):
    """Singleton metaclass

    make sure there's only one instance of each class"""

    single_instance_dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.single_instance_dict:
            cls.single_instance_dict[cls] = super().__call__(*args, **kwargs)
        return cls.single_instance_dict[cls]


