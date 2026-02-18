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

from typing import Any

import torch
import torch.nn as nn

from ..graph import serialize_fx_graph
from ..teenygrad import atlas_compile  # pylint: disable=c-extension-no-member


class AtlasModule(nn.Module):
    """Represents a Teenygrad module."""

    gm: torch.fx.GraphModule
    example_inputs: list[Any] | None
    compiled_module: torch.nn.Module | None

    def __init__(
        self, gm: torch.fx.GraphModule, example_inputs: list[Any] | None = None
    ):
        """Initialize the module."""
        super().__init__()
        self.gm = gm
        self.example_inputs = example_inputs
        self._compile(example_inputs)

    def forward(self, *args, **kwargs):
        """Forward pass."""
        self._print_args(*args, **kwargs)
        return self.gm.forward(*args, **kwargs)

    def _compile(self, args: list[Any] | None):
        """Compile the module."""
        print("Compiling module...")
        fxgraph = serialize_fx_graph(self.gm, args)
        self.compiled_module = atlas_compile(fxgraph)

    def _print_args(self, *args, **kwargs):
        """Print the argument."""
        # print("ARGS:", args)
        print("KWARGS:", kwargs)
