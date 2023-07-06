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
import glob
import os
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np


def main(data_path: str) -> None:
    tile_path = os.path.join(data_path, "TILES")
    icf_path = os.path.join(data_path, "ICF")

    # changing the pointer to the icf files based on the path to the data
    update_tile_pointer(icf_path, tile_path)
    # creating icf manifest
    icf_manifest_path = os.path.join(data_path, "icf_manifest.txt")
    create_manifests(icf_path, icf_manifest_path)
    # creating tile manifest
    tile_manifest_path = os.path.join(data_path, "tile_manifest.txt")
    create_manifests(tile_path, tile_manifest_path)


def update_tile_pointer(icf_path: str, tile_path: str) -> None:
    """changing the pointer to the icf files based on the path to the data

    Parameters
    ----------
    icf_path : str
        Path to the manifests of input ICF normalized files.
    tile_path : str
        Path to the manifests of tile coordinates files, each corresponding to one ICF h5.
    """

    h5_list = glob.glob(os.path.join(tile_path, "*.h5"))
    for f in h5_list:
        h5 = h5py.File(f, "r+")
        curr_path = h5.attrs["INPUT_POINTER"].decode("UTF-8")
        basename = os.path.basename(os.path.normpath(curr_path))
        new_path = os.path.join(icf_path, basename)
        h5.attrs["INPUT_POINTER"] = np.string_(new_path)
        h5.close()


def create_manifests(h5_list_path: list, manifest_path: str) -> None:
    """creating manifest, each line is the path to one tile/icf file

    Parameters
    ----------
    h5_list_path : list
        List of all *.h5 files for the manifest
    manifest_path : str
        Path to where the manifest is saved
    """
    h5_list = glob.glob(os.path.join(h5_list_path, "*.h5"))
    with open(manifest_path, "w") as temp:
        for f in h5_list:
            temp.write(os.path.join(h5_list_path, f) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", help="Path to the data", required=True)

    args = parser.parse_args()

    main(
        args.data_dir,
    )
