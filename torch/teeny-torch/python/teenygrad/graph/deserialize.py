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

"""Deserialize FX graphs from flatbuffers"""

import torch

from .FXGraph.Graph import Graph as FBGraph


def deserialize_fx_graph(buffer: bytes) -> torch.fx.GraphModule:
    """Deserialize a flatbuffer to a torch.fx.GraphModule."""
    # Read the flatbuffers Graph object from the bytes buffer
    # Flatbuffers expects a bytearray or bytes-like object
    if not isinstance(buffer, (bytes, bytearray)):
        raise TypeError("Buffer must be bytes or bytearray")
    graph_fb = FBGraph.GetRootAs(buffer, 0)

    # NOTE: This is a stub. To actually reconstruct a torch.fx.GraphModule,
    # you would need to walk the flatbuffer graph_fb and rebuild the FX graph.
    # For now, just return the flatbuffer object for debugging.
    return graph_fb
