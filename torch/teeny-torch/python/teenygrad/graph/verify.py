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

"""Verify the integrity of serialized FX graphs"""

import traceback

from .FXGraph.Graph import Graph as FBGraph


def verify_buffer_integrity(buffer: bytes) -> bool:
    """Verify that a serialized flatbuffer is valid and can be deserialized."""
    try:
        # Try to deserialize the buffer to verify it's valid
        graph = FBGraph.GetRootAs(buffer, 0)

        # Check if we can access basic fields
        if graph.NodesLength() < 0:
            print("No nodes found in the graph")
            return False

        # Try to access a few nodes to ensure the structure is sound
        for i in range(min(graph.NodesLength(), 5)):  # Check first 5 nodes
            node = graph.Nodes(i)
            if node is None:
                print("Node is None")
                return False
            # Try to access basic node fields
            if node.Name() is None or node.Target() is None:
                print("Node name or target is None")
                return False

        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        print("Exception occurred while verifying buffer integrity")
        traceback.print_exc()
        print(e)
        return False
