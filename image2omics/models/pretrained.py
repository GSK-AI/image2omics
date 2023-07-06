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
from typing import Tuple

import torch.nn as nn
from efficientnet_pytorch import VALID_MODELS, EfficientNet
from torchvision import models, transforms

from image2omics.models.modules import FlattenLayer, SqueezeLayer

AVAILABLE_MODELS = [
    "alexnet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
    "mobilenet_v2",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
] + list(VALID_MODELS)

# Some of the below models don't currently work as feature extractors and need extra work
AVAILABLE_MODELS_ALL = [
    "alexnet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "googlenet",
    "inception_v3",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
    "mobilenet_v2",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
] + list(VALID_MODELS)


class GSKEfficientNet(EfficientNet):
    def __init__(self, *args, **kwarargs):
        super(GSKEfficientNet, self).__init__(*args, **kwarargs)
        self.include_classification_head = False

    def forward(self, input):
        x = self.extract_features(input)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        if self.include_classification_head:
            x = self._fc(x)

        return x


def _validate_model_name(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model name {model_name}, please provide one of {AVAILABLE_MODELS}"
        )


def _reset_input_channels(model: nn.Module, num_channels: int) -> nn.Module:
    model_name = model.torchvision_name
    if any(name in model_name for name in ["resnet", "resnext"]):
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    elif any(name in model_name for name in ["densenet"]):
        out_channels = model.features.conv0.out_channels
        model.features.conv0 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    elif any(name in model_name for name in ["efficientnet"]):
        output_channels = model._conv_stem.out_channels
        model._conv_stem = nn.Conv2d(
            in_channels=num_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    else:
        raise ValueError(
            f"_reset_input_channels does not support model {model_name}. Choose models from familes of resnet, resnext, densenet, and efficientnet."
        )

    return model


def get_imagenet_feature_extractor(
    model_name: str,
    use_pretrained: bool = True,
    freeze_weights: bool = False,
    num_channels: int = 3,
) -> Tuple[nn.Module, int]:
    """Removes final layer of model and returns set of remaining modules as feature extractor.

    Parameters
    ----------
    model_name : str
        Name of torchvision model, see list AVAILABLE_MODELS
    freeze_weights : bool (optional)
        whether the weights (apart from the final layer) are frozen
    use_pretrained : bool (optional)
        whether the weights pretrained on ImageNet are used as initialization
    num_channels : int
        Number of input channels. If num_channels != 3, the first convolution layer
        will be reset to accomodate the correct num_channels input.

    Returns
    -------
    Tuple[nn.Module, int]
        imagenet feature extractor, number of output feature dimension
    """
    if model_name in VALID_MODELS:
        include_classification_head = False
    else:
        include_classification_head = True

    model, num_final_features, _ = get_imagenet_architecture(
        model_name=model_name,
        num_classes=1,
        freeze_weights=freeze_weights,
        use_pretrained=use_pretrained,
        include_classification_head=include_classification_head,
    )

    if num_channels != 3:
        try:
            model = _reset_input_channels(model=model, num_channels=num_channels)
        except ValueError as e:
            raise ValueError(f"{str(e)} Or use default num_channels=3.")

    if model_name in VALID_MODELS:
        model._fc = None
        return model, num_final_features

    modules = list(model.children())[:-1]

    if needs_flattening_before_final(model_name):
        modules.append(FlattenLayer(2))
    elif needs_pooling_before_final(model_name):
        modules.append(nn.AdaptiveAvgPool2d(output_size=1))
    last_module = list(model.children())[-1]
    # if the last layer has children, we add everything except the last child
    if len(list(last_module.children())) > 0:
        modules += list(last_module.children())[:-1]
    if needs_squeeze_before_final(model_name):
        modules.append(SqueezeLayer((-2, -1)))
    model = nn.Sequential(*modules)
    return model, num_final_features


def get_named_model(model_name: str, pretrained: bool) -> nn.Module:
    """
    Return a torchvision model with a "torchvision_name" atrribute, equal to the name of
    the model.

    Parameters
    ----------
    model_name :
        Name of torchvision model, see list AVAILABLE_MODELS
    pretrained :
        whether the weights pretrained on ImageNet are used as initialization

    Returns
    -------
    nn.Module
        The pytorch model

    Raises
    ------
    ValueError
        If model_name is not valid

    Example:
    >>> model = get_named_model(model_name='alexnet', pretrained=False)
    >>> model.torchvision_name
    'alexnet'
    """
    _validate_model_name(model_name)
    if model_name in VALID_MODELS:
        if pretrained:
            model = GSKEfficientNet.from_pretrained(model_name)
        else:
            model = GSKEfficientNet.from_name(model_name)
    else:
        model = getattr(models, model_name)(pretrained=pretrained)
    model.torchvision_name = model_name
    return model


def output_input_dims(named_model: nn.Module):
    """
    Return the expected height and width of input images, and the dim of final layer
    activations, given a model with a 'torchvision_name' attribute.

    Parameters
    ----------
    named_model :
        A module having a 'torchvision_name'

    Returns
    -------
    int
        number of final layer features
    int
        expected height and width of input images

    Raises
    ------
    ValueError
        If model_name is not valid
    ValueError
        If model_name is valid but not covered by this function
    """
    if not hasattr(named_model, "torchvision_name"):
        raise ValueError(
            "model must have a torchvision_name attribute to use with output_input_dims"
        )

    model_name = named_model.torchvision_name

    if any(name in model_name for name in ["resnet", "resnext", "shufflenet"]):
        num_final_features = named_model.fc.in_features
        input_size = 224

    elif any(name in model_name for name in ["alexnet", "vgg", "mobilenet", "mnas"]):
        num_final_features = named_model.classifier[-1].in_features
        input_size = 224

    elif "densenet" in model_name:
        num_final_features = named_model.classifier.in_features
        input_size = 224

    elif "squeezenet" in model_name:
        num_final_features = 1
        input_size = 224

    elif "inception" in model_name:
        num_final_features = named_model.fc.in_features
        input_size = 299

    elif "googlenet" in model_name:
        num_final_features = named_model.fc.in_features
        input_size = 224

    elif model_name in VALID_MODELS:
        input_size = named_model.get_image_size(model_name)
        num_final_features = named_model._fc.in_features

    else:
        raise ValueError(
            f"Model {model_name} is available but not covered by output_input_dims logic."
            "This is a problem with the code, not your input."
        )

    return num_final_features, input_size


def get_imagenet_architecture(
    model_name: str,
    num_classes: int,
    use_pretrained: bool = True,
    freeze_weights: bool = False,
    include_classification_head: bool = True,
) -> Tuple[nn.Module, int, int]:
    """
    Loads one of the ImageNet models given in the list AVAILABLE_MODELS,
    potentially with pretrained weights.

    Parameters
    ----------
    model_name : str
        Name of torchvision model, see list AVAILABLE_MODELS
    num_classes : int
        Number of final outputs of the model, i.e. the number of classes in classification
    freeze_weights : bool (optional)
        whether the weights (apart from the final layer) are frozen
    use_pretrained : bool (optional)
        whether the weights pretrained on ImageNet are used as initialization
    include_classification_head : bool (optional)
        Set true to include the classification head as default, note this option only has an effect on
        EfficientNet, all other models will include a classification head regardless of this switch

    Returns
    -------
    nn.Module
        The pytorch model
    int
        number of final layer features
    int
        the expected height and width of input images

    Raises
    ------
    ValueError
        If model_name is not valid
    ValueError
        If model_name is valid but not covered by this function
    """
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    _validate_model_name(model_name)

    model = get_named_model(model_name, pretrained=use_pretrained)
    if freeze_weights:
        freeze_parameters(model)

    num_final_features, input_size = output_input_dims(model)
    if any(name in model_name for name in ["resnet", "resnext", "shufflenet"]):
        model.fc = nn.Linear(num_final_features, num_classes)
    elif any(name in model_name for name in ["alexnet", "vgg", "mobilenet", "mnas"]):
        model.classifier[-1] = nn.Linear(num_final_features, num_classes)

    elif "densenet" in model_name:
        model.classifier = nn.Linear(num_final_features, num_classes)

    elif "squeezenet" in model_name:
        # Currently it seems this model will not work with slidebased networks
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = num_classes

    elif "inception" in model_name:
        logging.warning(
            "Inception v3 expects (299,299) sized images and has auxiliary output"
        )
        # Handle the auxilary net
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        # Handle the primary net
        model.fc = nn.Linear(num_final_features, num_classes)

    elif "googlenet" in model_name:
        logging.warning("GoogLenet has auxiliary outputs")
        # Handle the auxilary nets
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
        # Handle the primary net
        model.fc = nn.Linear(num_final_features, num_classes)

    elif model_name in VALID_MODELS:
        if include_classification_head:
            model.include_classification_head = True
            model._fc = nn.Linear(num_final_features, num_classes)

    else:
        raise ValueError(
            f"Model {model_name} is available but not covered by get_imagenet_architecture logic."
            "This is a problem with the code, not your input."
        )

    return model, num_final_features, input_size


def get_imagenet_transform() -> transforms.Normalize:
    """
    Returns the torchvision normalization transform that all torchvision models
    with weights pretrained on ImageNet use.

    Returns
    -------
    torchvision.transforms.Normalize
    """
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def needs_pooling_before_final(model_name: str) -> bool:
    """
    Returns True if the model needs an additional pooling layer after separating
    the last child module out of the model.

    Parameters
    ----------
    model_name : str
        name of the model

    Returns
    -------
    bool
        whether model needs pooling before final layer
    """
    return any(
        model in model_name
        for model in ["densenet", "mobilenet", "mnasnet", "shufflenet"]
    )


def needs_flattening_before_final(model_name: str) -> bool:
    """
    Returns True if the model needs an additional flattening layer after separating
    the last child module out of the model.

    Parameters
    ----------
    model_name : str
        name of the model

    Returns
    -------
    bool
        whether model needs flattening before final layer
    """
    return any(model in model_name for model in ["alexnet", "vgg"])


def needs_squeeze_before_final(model_name: str) -> bool:
    """
    Returns True if the model needs an additional squeezing layer after separating
    the last child module out of the model.

    Parameters
    ----------
    model_name : str
        name of the model

    Returns
    -------
    bool
        whether model needs squeezing before final layer
    """
    return any(
        model in model_name
        for model in [
            "densenet",
            "mobilenet",
            "mnasnet",
            "resnet",
            "resnext",
            "shufflenet",
        ]
    )


def freeze_parameters(model: nn.Module) -> None:
    """
    Parameters
    ----------

    model : torch.nn.Module
        pytorch module for which to freeze weights

    Returns
    -------
    None
    """
    for param in model.parameters():
        param.requires_grad = False
