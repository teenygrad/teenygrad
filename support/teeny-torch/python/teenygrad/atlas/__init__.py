#
# Copyright (c) 2026 Teenygrad.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Teenygrad Atlas backend for TorchDynamo integration.

Module description.
"""

import importlib.metadata
from typing import Any

import torch

from .module import AtlasModule

REQUIRED_TORCH = "2.7.0"

try:
    torch_version = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError as exc:
    raise RuntimeError(
        f"torch>={REQUIRED_TORCH} is required but not installed"
    ) from exc

if tuple(map(int, torch_version.split("."))) < tuple(
    map(int, REQUIRED_TORCH.split("."))
):
    raise RuntimeError(f"Requires torch>={REQUIRED_TORCH}, found {torch_version}")


def atlas(gm: torch.fx.GraphModule, example_inputs: list[Any] | None = None):
    """
    To Do
    """
    module = AtlasModule(gm, example_inputs)
    return module
