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
