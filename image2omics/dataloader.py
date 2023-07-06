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

from collections.abc import Sequence
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data.dataloader import _utils
from torchvision import transforms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = False,
    shuffle: bool = False,
    collate_fn: Callable = _utils.collate.default_collate,
    sampler: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """
    Function for setting up the loader for the feature extraction = load data from manifest of well HDF-5 files encapsulated
    in CellPainting Dataset Class

    Paremeters:
    -----------
    dataset: torch.utils.data.Dataset
        Dataset object
    batch_size: int
        Size of each data batch
    num_workers: int, optional
        How many workers for the data pre-fetching
    pin_memory: bool, optional
        Pin CPU memory to prevent repeated allocation for faster transfer to the GPUs
    shuffle: bool, optional
        Shuffle dataset during sampling, by default False
    collate_fn: Callable
        A function for collating elements of dataset into batches, by default _utils.collate.default_collate
        Custom collate_fn is needed for cases where dataset.__getitem__ returns a non-torch-friendly object
        such as List[str]. See https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
        for info on how to write custom collate fn.
    sampler: Optional[Callable]
        A function or un-instantianted object that returns a torch.utils.data.sampler.Sampler object after instantiated
        with sampler(dataset).

    Returns
    -------
    torch.utils.data.DataLoader:
        Instance on torch DataLoader
    """
    if sampler is not None:
        sampler = sampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        shuffle = False


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    return dataloader
