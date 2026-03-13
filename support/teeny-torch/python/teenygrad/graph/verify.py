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
