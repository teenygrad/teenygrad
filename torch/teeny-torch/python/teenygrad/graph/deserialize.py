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
