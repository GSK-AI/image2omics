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
from typing import Dict, Iterable, List, Optional, Type, Union

import torch
from torch import nn

AGGREGATION_LAYERS = ["none", "mean", "max", "concat"]


def get_aggregation_layer(
    aggregation: Optional[Union[str, List[dict]]] = None, **kwargs
) -> nn.Module:
    if kwargs is None:
        kwargs = {}
    if aggregation is None:
        aggregation = "none"
    if aggregation in ("mean", "max", "none"):
        return SimpleAggregationLayer(aggregation)
    if aggregation == "concat":
        return ConcatAggregationLayer()
    raise AssertionError("Aggregation layer %s is unknown.", aggregation)


class SimpleAggregationLayer(nn.Module):
    """Simple aggregation layer over different tiles of the same slide,
    either using mean or max of the different tiles.
    """

    def __init__(self, name: str):
        super(SimpleAggregationLayer, self).__init__()
        self.name = name
        if name == "mean":
            self._aggregate = lambda x: torch.mean(x, dim=1)
        elif name == "max":
            self._aggregate = lambda x: torch.max(x, dim=1)[0]
        elif name == "none":
            self._aggregate = lambda x: x
        else:
            raise KeyError("Aggregation layer %s is unknown.", name)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2 and self.name != "none":
            x = torch.unsqueeze(x, dim=1)
        return self._aggregate(x)


class ConcatAggregationLayer(nn.Module):
    """Concatenation aggregation layer that works with torch tensor and
    list of torch tensors as inputs.

    When a torch.Tensor is provided, expect number of dimensions >= 3.
    When a a list of torch.Tensor is provided, expect number of dimensions >= 2.
    """

    def __init__(self):
        super(ConcatAggregationLayer, self).__init__()

    def _dim_is_correct(self, x):
        if isinstance(x, torch.Tensor):
            return x.dim() >= 3
        else:
            return x[0].dim() >= 2

    def forward(self, x: Union[torch.Tensor, Iterable[torch.Tensor]]):
        if not self._dim_is_correct(x):
            logging.error(
                "ConcatAggregationLayer expects inputs to have number of dimensions >= 3 for torch.Tensors "
                " and >= 2 for Iterable[torch.Tensor]."
            )
            sys.exit(1)

        if isinstance(x, torch.Tensor):
            x = [
                x[:, i] for i in range(x.size(1))
            ]  # Create a list of tensor to be concatenated

        return torch.cat(x, dim=1)
