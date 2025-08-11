#
# Copyright (c) 2025 Teenygrad. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Teenygrad Atlas backend for TorchDynamo integration.

Module description.
"""

import importlib.metadata

import torch

from .module import AtlasModule

REQUIRED_TORCH = "2.7.0"

try:
    torch_version = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError as exc:
    raise RuntimeError(f"torch>={REQUIRED_TORCH} is required but not installed") from exc

if tuple(map(int, torch_version.split("."))) < tuple(map(int, REQUIRED_TORCH.split("."))):
    raise RuntimeError(f"Requires torch>={REQUIRED_TORCH}, found {torch_version}")


def atlas(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor] | None = None):
    """
    To Do
    """
    module = AtlasModule(gm, example_inputs)
    return module


# pylint: disable=protected-access
torch._dynamo.register_backend(atlas, "atlas")
