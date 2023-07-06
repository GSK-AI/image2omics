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

import torch


def unmodified(x: torch.Tensor):
    """Return unmodified batch of images"""
    return x


def flip_vertical(x: torch.Tensor):
    """Return a batch of images flipped along the second-to-last dimension

    Assumes input is in channel-first format and has dimension [B, C, H, W]
    """
    return x.flip(-2)


def flip_horizontal(x: torch.Tensor):
    """Return a batch of images flipped along the last dimension

    Assumes input is in channel-first format and has dimension [B, C, H, W]
    """
    return x.flip(-1)


def transpose(x: torch.Tensor):
    """Return a batch of images with H and W transposed

    Assumes input is in channel-first format and has dimension [B, C, H, W]
    """
    return x.transpose(-2, -1)


DEFAULT_TTA_OPS = [unmodified, flip_vertical, flip_horizontal, transpose]
IDENTITY_TTA_OPS = [unmodified]
