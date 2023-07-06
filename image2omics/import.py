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

from importlib import import_module
import builtins

def import_from(path: str):
    """Import object from module according to the specified task.
    If import fails, the function recursively checks each import along the path
    to find the import error.
    Copied and modified from hydra 1.0 codebase
    https://github.com/facebookresearch/hydra/tree/1.0_branch
    Parameters
    ----------
    path
        dot-separated import path
        examples:
        - "omegaconf.OmegaConf"
        - "torch.relu"
        - "torch.nn.Linear"
    Returns
    -------
    imported object
    """
    # Copied from hydra codebase. TODO: clean up.
    if path == "":
        raise ImportError("Empty path")

    parts = [part for part in path.split(".") if part]
    module = None
    for n in reversed(range(len(parts))):
        try:
            mod = ".".join(parts[:n])
            module = import_module(mod)
        except Exception as e:
            if n == 0:
                raise ImportError(f"Error loading module '{path}'") from e
            continue
        if module is not None:
            break
    if module is not None:
        obj = module
    else:
        obj = builtins
    for part in parts[n:]:
        mod = mod + "." + part
        if not hasattr(obj, part):
            try:
                import_module(mod)
            except Exception as e:
                raise ImportError(
                    f"Encountered error: `{e}` when loading module '{path}'"
                ) from e
        obj = getattr(obj, part)
    return obj