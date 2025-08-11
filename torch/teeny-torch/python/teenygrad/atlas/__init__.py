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
import traceback

import torch

from ..graph import serialize_fx_graph, verify_buffer_integrity

# Import the compiled Rust extension
try:
    from .. import teenygrad
    atlas_compile = teenygrad.atlas_compile  # pylint: disable=c-extension-no-member
except ImportError:
    # Fallback for development/testing
    def atlas_compile(buffer):
        """Dummy implementation of atlas_compile"""
        raise RuntimeError("atlas_compile not available - Rust extension not loaded")

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

    # Debug logging to help diagnose buffer size issues
    print(f"Debug: Serialized graph size: {len(graph)} bytes")
    print(f"Debug: Graph has {len(list(gm.graph.nodes))} nodes")
    if example_inputs:
        print(f"Debug: Example inputs count: {len(example_inputs)}")

    # Verify buffer integrity before passing to Rust
    if not verify_buffer_integrity(graph):
        raise RuntimeError("Generated buffer failed integrity verification - cannot proceed to Rust")

    try:
        result = atlas_compile(graph)
    except Exception as e:
        print(f"Debug: Atlas compilation error: {e}")
        print(f"Debug: Error type: {type(e)}")
        print(f"Debug: Traceback: {traceback.format_exc()}")
        raise

    print(f"Debug: Atlas compilation result: {result}")
    return gm.forward  # Return original function


# pylint: disable=protected-access
torch._dynamo.register_backend(atlas, "atlas")
