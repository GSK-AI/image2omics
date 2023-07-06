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

import logging
import sys
from collections.abc import Sequence
from typing import Callable, List, Optional, Union

import albumentations as A
import albumentations.augmentations.functional as F
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchvision import transforms

#from aiml_cell_imaging.utils.utils import convert_image


class ComposeAdapter(A.Compose):
    """Adapter for Albumentations' Compose.
    This object mimicks the behavior of torchvision's Compose.
    """

    def __init__(self, **kwargs):
        super(ComposeAdapter, self).__init__(**kwargs)

    def __call__(self, img, **kwargs):
        return super(ComposeAdapter, self).__call__(image=img, **kwargs)["image"]


class NormalizePerChannel(ImageOnlyTransform):
    """Perform per-channel normalization.

    Assume images are in channels-last mode (H,W,C)
    """

    def __init__(self, always_apply=False, p=1.0):
        super(NormalizePerChannel, self).__init__(always_apply, p)

    def apply(self, image, **params):
        if len(image.shape) not in [2, 3]:
            logging.error(
                f"Image has {len(image.shape)} dimensions. Expected 2 (HC) or 3 (HWC) dimensions."
            )
            sys.exit(1)

        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)

        pixel_mean = image.mean((0, 1))
        pixel_std = image.std((0, 1)) + 1e-8
        image = (image - pixel_mean.reshape(1, 1, -1)) / pixel_std.reshape(1, 1, -1)
        return image

    def get_transform_init_args_names(self):
        return ()


class ConvertImage:
    """Wrapper for old convert_image method to be used as a torchvision Transform"""

    def __call__(self, channel_array):
        image = convert_image(channel_array=channel_array)
        return image

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MultiplicativeNoiseWithoutClipping(A.MultiplicativeNoise):
    """Wrapper around Albumentations' MultiplicativeNoise without default clipping

    This can be used for adding noises after normalization.
    See https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py
    for arguments.
    """

    def __init__(self, **kwargs):
        super(MultiplicativeNoiseWithoutClipping, self).__init__(**kwargs)

    def apply(self, img: np.ndarray, multiplier=np.array([1]), **kwargs):
        return _multiply_without_clipping(img, multiplier)


def _multiply_without_clipping(img, multiplier):
    """Multiplying using unwrapped functions from albumentations"""
    multiply_op = F._multiply_non_uint8
    if img.dtype == np.uint8:
        multiply_op = F._multiply_uint8
        if len(multiplier.shape) == 1:
            multiply_op = F._multiply_uint8_optimized

    return multiply_op.__wrapped__(img, multiplier).astype(img.dtype)


def build_data_transformation(
    resize: Union[Sequence, int, None] = None,
    center_crop: Union[Sequence, int, None] = None,
    flip_horiz_prob: float = 0.0,
    flip_vert_prob: float = 0.0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
    saturation_jitter: float = 0.0,
    hue_jitter: float = 0.0,
    to_pilimage: bool = False,
    to_tensor: bool = True,
    translate: List[float] = [0, 0],
    degrees: Union[int, List[int]] = 0,
    mean: Optional[Sequence] = None,
    std: Optional[Sequence] = None,
    **kwargs,
) -> transforms.Compose:
    """Build data transformation operations based on input parameters from a config JSON file

    parameters
    ------
    resize
        If not None, transforms.Resize() will be added, with the value of resize as its input.
        If None, no resizing will be applied
    center_crop
        If not None, transforms.CenterCrop() will be added, with the value of resize as its input.
        If None, no center cropping will be applied
    flip_horiz_prob : float
        Horizontal flip probability
    flip_vert_prob : float
        Vertical flip probability
    brightness_jitter : float
        How much to jitter brightness [0,1]
    contrast_jitter : float
        How much to jitter contrast [0,1]
    saturation_jitter : float
        How much to jitter saturation [0,1]
    hue_jitter : float
        How much to jitter hue [0,1]
    to_tensor : bool
        Boolean, if True, transforms.ToTensor() will be added
    translate : List[float]
        Fraction of total horizontal and vertical translation, by default [0, 0]
    degrees: Union[int, List[int]]
        Range of degrees to rotate image. When int is provided, range from (-degree, +degree), by default 0
    mean
        A list of mean values for each image channel for normalization.
    std
        A list of std values for each image channel for normalization.
        When both mean and std are not None, normalization will be added to the transformation list.

    Returns
    -------
    transforms.Compose
        A composed series of data transformations
    """
    transforms_list = []
    if to_pilimage:
        transforms_list.append(transforms.ToPILImage())
    if resize is not None:
        transforms_list.append(transforms.Resize(resize))
    if center_crop is not None:
        transforms_list.append(transforms.CenterCrop(center_crop))
    if flip_horiz_prob > 0:
        transforms_list.append(transforms.RandomHorizontalFlip(p=flip_horiz_prob))
    if flip_vert_prob > 0:
        transforms_list.append(transforms.RandomVerticalFlip(p=flip_vert_prob))
    if degrees != 0 or translate != [0, 0]:
        transforms_list.append(
            transforms.RandomAffine(degrees=degrees, translate=translate)
        )

    jitter_kwargs = {
        k.replace("_jitter", ""): v for k, v in locals().items() if "_jitter" in k
    }
    if any(jitter_prob > 0 for jitter_prob in jitter_kwargs.values()):
        transforms_list.append(transforms.ColorJitter(**jitter_kwargs))
    if to_tensor:
        transforms_list.append(transforms.ToTensor())
    if (mean is not None) and (std is not None):
        transforms_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transforms_list)
