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

from ..graph import serialize_fx_graph
from ..teenygrad import atlas_compile  # type: ignore noqa: F401

REQUIRED_TORCH = "2.7.0"

try:
    torch_version = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError as exc:
    raise RuntimeError(f"torch>={REQUIRED_TORCH} is required but not installed") from exc

if tuple(map(int, torch_version.split("."))) < tuple(map(int, REQUIRED_TORCH.split("."))):
    raise RuntimeError(f"Requires torch>={REQUIRED_TORCH}, found {torch_version}")


def atlas(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor] | None = None):
    """
    This is a dummy implementation of the atlas backend for TorchDynamo.
    It is used to test the Teenygrad-Torch interop.

    It is not a real atlas backend, it is just a placeholder to allow the
    Teenygrad-Torch interop to be tested.

    It is not a real atlas backend, it is just a placeholder to allow the
    Teenygrad-Torch interop to be tested.
    """
    graph = serialize_fx_graph(gm, example_inputs)
    result = atlas_compile(graph)
    print(result)
    return gm.forward  # Return original function


# pylint: disable=protected-access
torch._dynamo.register_backend(atlas, "atlas")
