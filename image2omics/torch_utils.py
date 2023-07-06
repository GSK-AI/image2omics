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
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist

if dist.is_available():
    from torch.distributed import ReduceOp

Tensorlike = Union[np.ndarray, torch.Tensor]
TensorlikeOrTupleThereof = Union[Tensorlike, Tuple[Tensorlike, ...]]
TensorOrTupleThereof = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
TensorOrContainer = Union[torch.Tensor, Dict, List, Tuple]
TensorOrDictThereof = Union[torch.Tensor, Dict[str, torch.Tensor]]
TensorOrTupleThereofOrDictThereof = Union[
    TensorOrTupleThereof, Dict[str, TensorOrTupleThereof]
]
ArrayOrTupleThereof = Union[np.ndarray, Tuple[np.ndarray, ...]]
ArrayOrTupleThereofOrDictThereof = Union[
    ArrayOrTupleThereof, Dict[str, ArrayOrTupleThereof]
]
ArrayOrContainer = Union[np.ndarray, Dict, List, Tuple]
TensorlikeOrContainer = Union[Tensorlike, Dict, List, Tuple]
Device = Union[str, torch.device]


def recursive_on_containers(function: Callable) -> Callable:
    """Wrapped function acts recursively on containers.
    Can deal with lists, tuples and dicts.
    Parameters
    ----------
    function: Callable
    Returns
    -------
    Callable
    """

    @wraps(function)
    def function_on_container(x, *args, **kwargs):
        if isinstance(x, list):
            return [function_on_container(item, *args, **kwargs) for item in x]
        if isinstance(x, tuple):
            return tuple(function_on_container(item, *args, **kwargs) for item in x)
        if isinstance(x, dict):
            return {k: function_on_container(v, *args, **kwargs) for k, v in x.items()}
        return function(x, *args, **kwargs)

    return function_on_container


@recursive_on_containers
def to_tensor(
    x: Tensorlike, device: Optional[Device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Sends x to the specified device.
    If x is a numpy array, it gets cast to a pytorch tensor on the device.
    If x is a container, the function is applied recursively to the items/values
    Parameters
    ----------
    x : TensorlikeOrContainer
        object to send to device
    device : Union[str, torch.device], optional
        specifies device
    dtype: torch.dtype, optional
        specifies dtype of Tensor
    Returns
    -------
    TensorOrContainer
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=dtype)


@recursive_on_containers
def to_numpy(x: Tensorlike) -> np.ndarray:
    """Sends x to the specified device.
    If x is a numpy array, it gets cast to a pytorch tensor on the device.
    If x is a dictionary, the function is applied recursively to the values in the dictionary.
    Parameters
    ----------
    x : TensorlikeOrContainer
        object to send to numpy
    Returns
    -------
    ArrayOrContainer
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.data.to("cpu").numpy()


@recursive_on_containers
def detach(x: torch.Tensor) -> torch.Tensor:
    return torch.detach(x)


@recursive_on_containers
def get_shape(x: torch.Tensor) -> torch.Size:
    return x.shape


@recursive_on_containers
def mask_select(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask]


def backward(x: TensorOrContainer, **kwargs):
    """Computes the backward pass for a tensor or the backward pass on the sum for a set of
    tensors in containers.
    Parameters
    ----------
    x : Union[torch.tensor, Dict[any, torch.Tensor], List[torch.Tensor],
                      Set[torch.Tensor], Tuple[torch.Tensor]]
        a tensor or a container of tensors.
    """

    def _sum_all(x: TensorOrContainer) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, (list, set, tuple)):
            return sum(_sum_all(item) for item in x)
        elif isinstance(x, dict):
            return sum(_sum_all(item) for item in x.values())

    x = _sum_all(x)
    if isinstance(x, torch.Tensor):
        x.backward(**kwargs)


def dict_scaler_all_reduce_mean(x: Dict, device: torch.device) -> Dict:
    """Convenience function for aggregate metrics for multi-gpu training
    Doing a all reduce with mean for dict of scalers values
    Parameters
    ----------
    x :
        With keys are strings and values scaler. keys are the metric name and
        values are single scaler
    device :
        torch.device
    Returns
    -------
    A Dict either the same as input or aggregated metrics
    """
    if dist.is_available():
        try:
            num_replicas = dist.get_world_size()
            for k, v in x.items():
                if isinstance(v, dict):
                    x[k] = dict_scaler_all_reduce_mean(v, device)
                elif isinstance(v, (float, int)):
                    v = torch.tensor([v], device=device)
                    dist.all_reduce(v, op=ReduceOp.SUM)
                    x[k] = v.cpu().item() / float(num_replicas)
                else:
                    logging.error("data not scalar. Found type %s", type(v))
                    raise TypeError
            return x
        # Two exceptions below are for PyTorch backward compatibility.
        # In PyTorch 1.9, the expected error from get_world_size() is RuntimeError
        # while it has previously been AssertionError.
        except (AssertionError, RuntimeError):
            logging.warning(
                "Distributed env not initialized, returning the metrics from lead process"
            )
            return x
    else:
        return x


def to_list(a: Union[List, Tensorlike]):
    """Convenience function: Converts a tensorlike object to a list"""
    if isinstance(a, list):
        return a
    return to_numpy(a).tolist()
