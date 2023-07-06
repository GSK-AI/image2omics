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

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from image2omics.models.aggregation import get_aggregation_layer
from image2omics.models.modules import MultiLayerPerceptron
from image2omics.models.pretrained import get_imagenet_feature_extractor
import logging

class PrebuildImageClassifier(nn.Module):
    """Classifier created as simple wrapper of torchvision prebuilt architecture.

    Parameters
    ----------
    model_name : str
        Name of prebuilt architecture, must be in the namespace of torchvision.models
    freeze_weights : bool, optional
        Freeze weights of prebuilt feature extractor architecture
        default is False
    use_pretrained : bool, optional
        Use weights from model trained with ImageNet
        default is True
    tasks : Dict[str, dict], optional
        if given, model has a head for each specified task
        keys are the different tasks, values are kwargs for each individual head
        (and overwrite global kwargs)
    mlp_kwargs : Dict[str, str], optional
        kwargs passed on to the MultiLayerPerceptron class
    """

    def __init__(
        self,
        model_name: str,
        num_channels: Optional[int] = None,
        freeze_weights: bool = False,
        use_pretrained: bool = True,
        num_classes: int = 1,
        mlp_kwargs: dict = {},
        aggregation: Optional[str] = None,
        aggregation_kwargs: dict = {},
    ):
        super(PrebuildImageClassifier, self).__init__()
        self.feature_extractor_kwargs = {
            "model_name": model_name,
            "freeze_weights": freeze_weights,
            "use_pretrained": use_pretrained,
        }

        if num_channels is None:
            self.add_module("base", None)
        else:
            self.base, base_output_features = get_imagenet_feature_extractor(
                **self.feature_extractor_kwargs, num_channels=num_channels
            )
        self.aggregate = get_aggregation_layer(
            aggregation=aggregation, **aggregation_kwargs
        )
        self.mlp_kwargs = mlp_kwargs
        self.mlp_kwargs["num_classes"] = num_classes
        # Need to register module as part of the variable scope of the nn.Module object.
        # Just setting self.head = None is not sufficient and leads to errors when loading
        # models from state dict.
        self.add_module("head", None)

    def forward(
        self, data: Dict[str, torch.Tensor], embed: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass

        Parameters
        ----------
        data : Dict of torch.Tensors. Must at least contain a key "inputs" with an
            Input tensor, which must have shape [*, 3, height, width]
        embed : bool
            Stop the forward pass at the penultimate layer and return the outputs, by default False

        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            if single task or task is provided:
                torch.Tensor
                    Model output with shape [*, num_classes]
            else:
                Dict[str, torch.Tensor]
                    keys are tasks, values are model output with shape [*, num_classes]
        """
        if isinstance(data, dict):
            x = data["inputs"]
        elif isinstance(data, torch.Tensor):
            x = data
        else:
            raise TypeError("Type %s not valid input to model", type(data))
        external_shape = x.shape[:-3]
        # Reshaping necessary since pre-defined models can only deal with 4-dim tensors
        if x.dim() > 4:
            x = x.reshape((-1,) + x.shape[-3:])

        if self.base is None:
            self.base, base_output_features = get_imagenet_feature_extractor(
                **self.feature_extractor_kwargs, num_channels=x.shape[-3]
            )
            self.base = self.base.to(x)

        outputs = self.base(x)

        # undo original reshaping
        outputs = outputs.reshape(external_shape + (outputs.shape[-1],))

        # aggregate
        outputs = self.aggregate(outputs)

        if self.head is None:
            self.head = MultiLayerPerceptron(outputs.shape[-1], **self.mlp_kwargs).to(
                outputs
            )

        # return layers before penultimate output layer
        if embed:
            return self.head.embed(outputs)
        outputs = self.head(outputs)
        return outputs

class MLP(nn.Module):
    def __init__(self, embed_size, num_classes, mlp_kwargs):
        super().__init__()
        self.MLP = MultiLayerPerceptron(embed_size, num_classes, **mlp_kwargs)

    def forward(self, x):
        x = self.MLP(x)
        return x
