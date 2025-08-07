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

import flatbuffers  # type: ignore

import torch

from .FXGraph import Graph, KeyValue, Node, OpType


def serialize_fx_graph(gm: torch.fx.GraphModule) -> bytes:
    """Serialize a torch.fx.GraphModule to a flatbuffer."""
    builder = flatbuffers.Builder(1024)

    # Map nodes for user tracking
    # node_map = {node.name: node for node in gm.graph.nodes}

    # Serialize nodes in reverse order
    node_offsets = []
    for node in reversed(gm.graph.nodes):
        # Serialize users
        users = [builder.CreateString(u.name) for u in node.users]
        Node.StartUsersVector(builder, len(users))
        for user in reversed(users):
            builder.PrependUOffsetTRelative(user)
        users_vec = builder.EndVector()

        # Serialize args/kwargs
        args = [builder.CreateString(str(a)) for a in node.args]
        Node.StartArgsVector(builder, len(args))
        for arg in reversed(args):
            builder.PrependUOffsetTRelative(arg)
        args_vec = builder.EndVector()

        kwargs = []
        for k, v in node.kwargs.items():
            key = builder.CreateString(k)
            val = builder.CreateString(str(v))
            KeyValue.Start(builder)
            KeyValue.AddKey(builder, key)
            KeyValue.AddValue(builder, val)
            kwargs.append(KeyValue.End(builder))

        Node.StartKwargsVector(builder, len(kwargs))
        for kv in reversed(kwargs):
            builder.PrependUOffsetTRelative(kv)
        kwargs_vec = builder.EndVector()

        # Build node
        name = builder.CreateString(node.name)
        target = builder.CreateString(str(node.target))

        Node.Start(builder)
        Node.AddName(builder, name)
        Node.AddOp(builder, getattr(OpType, node.op))
        Node.AddTarget(builder, target)
        Node.AddArgs(builder, args_vec)
        Node.AddKwargs(builder, kwargs_vec)
        Node.AddUsers(builder, users_vec)
        node_offsets.append(Node.End(builder))

    # Create node vector
    Graph.StartNodesVector(builder, len(node_offsets))
    for offset in reversed(node_offsets):
        builder.PrependUOffsetTRelative(offset)
    nodes_vec = builder.EndVector()

    # Serialize inputs/outputs
    inputs = [builder.CreateString(n.name)
              for n in gm.graph.nodes if n.op == "placeholder"]
    Graph.StartInputNamesVector(builder, len(inputs))
    for name in reversed(inputs):
        builder.PrependUOffsetTRelative(name)
    inputs_vec = builder.EndVector()

    outputs = [builder.CreateString(n.name)
               for n in gm.graph.nodes if n.op == "output"]
    Graph.StartOutputNamesVector(builder, len(outputs))
    for name in reversed(outputs):
        builder.PrependUOffsetTRelative(name)
    outputs_vec = builder.EndVector()

    # Build final graph
    Graph.Start(builder)
    Graph.AddNodes(builder, nodes_vec)
    Graph.AddInputNames(builder, inputs_vec)
    Graph.AddOutputNames(builder, outputs_vec)
    graph_offset = Graph.End(builder)

    builder.Finish(graph_offset)
    return bytes(builder.Output())
