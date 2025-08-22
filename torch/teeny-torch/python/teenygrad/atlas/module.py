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

    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[Any] | None = None):
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
