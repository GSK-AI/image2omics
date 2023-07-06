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

from typing import Dict, Iterable, List, Optional, Type, Union

import torch
from torch import nn


class FlattenLayer(nn.Module):
    """Simple layer that flattens the last indices. Needed when restructuring some
    of the torchvision models.
    """

    def __init__(self, output_dim: int):
        super(FlattenLayer, self).__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[: self.output_dim - 1] + (-1,))


class SqueezeLayer(nn.Module):
    """Simple layer that squeezes the last indices. Needed when restructuring some
    of the torchvision models.
    """

    def __init__(self, dims: Union[int, Iterable[int]]):
        super(SqueezeLayer, self).__init__()
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dim in self.dims:
            x = x.squeeze(dim)
        return x


class MultiLayerPerceptron(nn.Module):
    """Standard multi-layer perceptron with non-linearity and potentially dropout.
    Parameters
    ----------
    num_input_features : int
        input dimension
    num_classes : int, optional
        Number of output classes. If not specified (or None), MLP does not have a final layer.
    hidden_layer_dimensions : List[int], optional
        list of hidden layer dimensions. If not provided, class is a linear model
    nonlin : Union[str, nn.Module]
        name of a nonlinearity in torch.nn, or a pytorch Module. default is relu
    p_dropout : float
        dropout probability for dropout layers. default is 0.0
    num_tasks : int, optional
        if specified, outputs for several tasks
        shape of output tensor changes from
        (batch_size, num_classes) to (batch_size, num_tasks, num_classes)
    detach: bool, (default=False)
        if set to True, detaches the inputs so that backprop does not affects
        previous layers.
    """

    def __init__(
        self,
        num_input_features: int,
        num_classes: Optional[int] = None,
        hidden_layer_dimensions: Optional[List[int]] = None,
        nonlin: Union[str, nn.Module] = "ReLU",
        p_dropout: float = 0.0,
        num_tasks: Optional[int] = None,
        detach: bool = False,
    ):
        super(MultiLayerPerceptron, self).__init__()
        if hidden_layer_dimensions is None:
            hidden_layer_dimensions = []
        if isinstance(nonlin, str):
            nonlin = getattr(torch.nn, nonlin)()
        layer_inputs = [num_input_features] + hidden_layer_dimensions
        modules = []
        self.detach = detach
        for i in range(len(hidden_layer_dimensions)):
            modules.extend(
                [
                    nn.Dropout(p=p_dropout),
                    nn.Linear(layer_inputs[i], layer_inputs[i + 1]),
                ]
            )
            if i < (len(hidden_layer_dimensions) - 1):
                modules.append(nonlin)

        self.module = nn.Sequential(*modules)
        if num_classes is None:
            self.has_final_layer = False
        else:
            self.has_final_layer = True
            if num_tasks is None:
                if num_classes > 1:
                    self.output_shape = (num_classes,)
                else:
                    self.output_shape = ()
                output_size = num_classes
            else:
                self.output_shape = (num_tasks, num_classes)
                output_size = num_tasks * num_classes
            self.final_nonlin = nonlin
            self.final_dropout = nn.Dropout(p=p_dropout)
            self.final = nn.Linear(layer_inputs[-1], output_size)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass up to penultimate layer"""
        if self.detach:
            inputs = torch.detach(inputs)
        outputs = self.module(inputs)
        return outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.detach:
            inputs = torch.detach(inputs)
        outputs = self.module(inputs)
        if self.has_final_layer:
            outputs = self.final_nonlin(outputs)
            outputs = self.final_dropout(outputs)
            outputs = self.final(outputs)
            outputs = torch.reshape(outputs, outputs.shape[:-1] + self.output_shape)
        return outputs