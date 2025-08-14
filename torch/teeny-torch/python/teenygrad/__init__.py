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

# Import the native extension module teenygrad (teenygrad.cpython-*.so)
# and expose the atlas_compiler function at the package level.

from typing import Any

import torch

from .atlas import atlas  # type: ignore noqa: F401


def teenygrad(gm: torch.fx.GraphModule, example_inputs: list[Any] | None = None):
    """
    Forwards to atlas.
    """
    return atlas(gm, example_inputs)


# pylint: disable=protected-access
torch._dynamo.register_backend(teenygrad, "teenygrad")

__all__ = ["teenygrad"]
