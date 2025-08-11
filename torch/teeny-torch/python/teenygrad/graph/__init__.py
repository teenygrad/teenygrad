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

"""Test serialization of FX graphs to flatbuffers"""

import traceback

import flatbuffers  # type: ignore

import torch

# Import the generated flatbuffer classes and functions
from .FXGraph.Device import Device
from .FXGraph.DType import DType
from .FXGraph.ExampleInputs import (
    ExampleInputsAddInputs,
    ExampleInputsEnd,
    ExampleInputsStart,
    ExampleInputsStartInputsVector,
)
from .FXGraph.Graph import Graph as FBGraph
from .FXGraph.Graph import (
    GraphAddExampleInputs,
    GraphAddInputNames,
    GraphAddNodes,
    GraphAddOutputNames,
    GraphEnd,
    GraphStart,
    GraphStartInputNamesVector,
    GraphStartNodesVector,
    GraphStartOutputNamesVector,
)
from .FXGraph.KeyValue import (
    KeyValueAddKey,
    KeyValueAddValue,
    KeyValueEnd,
    KeyValueStart,
)
from .FXGraph.Node import (
    NodeAddArgs,
    NodeAddKwargs,
    NodeAddName,
    NodeAddOp,
    NodeAddTarget,
    NodeAddUsers,
    NodeEnd,
    NodeStart,
    NodeStartArgsVector,
    NodeStartKwargsVector,
    NodeStartUsersVector,
)
from .FXGraph.OpType import OpType
from .FXGraph.Shape import (
    ShapeAddDims,
    ShapeEnd,
    ShapeStart,
    ShapeStartDimsVector,
)
from .FXGraph.Tensor import (
    TensorAddDevice,
    TensorAddDtype,
    TensorAddShape,
    TensorEnd,
    TensorStart,
)


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


def serialize_fx_graph(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor] | None = None) -> bytes:
    """Serialize a torch.fx.GraphModule to a flatbuffer."""
    # Calculate buffer size dynamically based on graph complexity
    # Base size + per-node overhead + per-tensor overhead
    base_size = 1024
    node_count = len(list(gm.graph.nodes))
    tensor_count = len(example_inputs) if example_inputs else 0

    # Estimate buffer size: base + nodes + tensors + safety margin
    estimated_size = base_size + (node_count * 256) + (tensor_count * 512) + 2048

    # Ensure minimum size and cap at reasonable maximum
    buffer_size = max(1024, min(estimated_size, 65536))

    builder = flatbuffers.Builder(buffer_size)

    try:
        # Build all nested objects first, then assemble the final structure
        # This ensures proper offset calculations

        # 1. Build nodes first (most complex part)
        node_offsets = []
        for node in gm.graph.nodes:
            if not hasattr(node, 'name') or not hasattr(node, 'target'):
                continue

            # Build users vector (empty for now to avoid circular references)
            NodeStartUsersVector(builder, 0)
            users_vec = builder.EndVector()

            # Build args vector
            args = []
            if hasattr(node, 'args'):
                args = [builder.CreateString(str(a)) for a in node.args if a is not None]

            NodeStartArgsVector(builder, len(args))
            for arg in reversed(args):
                builder.PrependUOffsetTRelative(arg)
            args_vec = builder.EndVector()

            # Build kwargs vector
            kwargs = []
            if hasattr(node, 'kwargs'):
                for k, v in node.kwargs.items():
                    if k is not None and v is not None:
                        key = builder.CreateString(str(k))
                        val = builder.CreateString(str(v))
                        KeyValueStart(builder)
                        KeyValueAddKey(builder, key)
                        KeyValueAddValue(builder, val)
                        kwargs.append(KeyValueEnd(builder))

            NodeStartKwargsVector(builder, len(kwargs))
            for kv in reversed(kwargs):
                builder.PrependUOffsetTRelative(kv)
            kwargs_vec = builder.EndVector()

            # Build the node
            name = builder.CreateString(str(node.name))
            target = builder.CreateString(str(node.target))

            op_type = OpType.placeholder  # default
            if hasattr(node, 'op') and hasattr(OpType, node.op):
                op_type = getattr(OpType, node.op)

            NodeStart(builder)
            NodeAddName(builder, name)
            NodeAddOp(builder, op_type)
            NodeAddTarget(builder, target)
            NodeAddArgs(builder, args_vec)
            NodeAddKwargs(builder, kwargs_vec)
            NodeAddUsers(builder, users_vec)
            node_offsets.append(NodeEnd(builder))

        if not node_offsets:
            raise ValueError("No valid nodes found for serialization")

        # 2. Build node vector
        GraphStartNodesVector(builder, len(node_offsets))
        for offset in reversed(node_offsets):  # Use reversed for proper flatbuffer construction
            builder.PrependUOffsetTRelative(offset)
        nodes_vec = builder.EndVector()

        # 3. Build input/output name vectors
        inputs = [builder.CreateString(n.name)
                  for n in gm.graph.nodes if n.op == "placeholder"]
        GraphStartInputNamesVector(builder, len(inputs))
        for name in reversed(inputs):
            builder.PrependUOffsetTRelative(name)
        inputs_vec = builder.EndVector()

        outputs = [builder.CreateString(n.name)
                   for n in gm.graph.nodes if n.op == "output"]
        GraphStartOutputNamesVector(builder, len(outputs))
        for name in reversed(outputs):
            builder.PrependUOffsetTRelative(name)
        outputs_vec = builder.EndVector()

        # 4. Build example inputs if provided
        example_inputs_offset = None
        if example_inputs and len(example_inputs) > 0:
            # Build tensor objects first
            tensor_offsets = []
            for tensor in example_inputs:
                # Build shape
                shape = list(tensor.shape)
                ShapeStartDimsVector(builder, len(shape))
                for dim in reversed(shape):
                    builder.PrependUint32(dim)
                shape_dims_vec = builder.EndVector()

                ShapeStart(builder)
                ShapeAddDims(builder, shape_dims_vec)
                shape_offset = ShapeEnd(builder)

                # Map device and dtype
                device_str = str(tensor.device)
                if 'cuda' in device_str:
                    device_enum = Device.CUDA
                elif 'xpu' in device_str:
                    device_enum = Device.XPU
                else:
                    device_enum = Device.CPU

                dtype_mapping = {
                    torch.float32: DType.FLOAT32,
                    torch.float64: DType.FLOAT64,
                    torch.int32: DType.INT32,
                    torch.int64: DType.INT64,
                    torch.uint8: DType.UINT8,
                    torch.int8: DType.INT8,
                    torch.bool: DType.BOOL,
                    torch.bfloat16: DType.BFLOAT16,
                    torch.float16: DType.FLOAT16,
                }
                dtype_enum = dtype_mapping.get(tensor.dtype, DType.FLOAT32)

                # Build tensor
                TensorStart(builder)
                TensorAddShape(builder, shape_offset)
                TensorAddDtype(builder, dtype_enum)
                TensorAddDevice(builder, device_enum)
                tensor_offsets.append(TensorEnd(builder))

            # Build inputs vector
            ExampleInputsStartInputsVector(builder, len(tensor_offsets))
            for offset in reversed(tensor_offsets):
                builder.PrependUOffsetTRelative(offset)
            inputs_vec_tensors = builder.EndVector()

            # Build ExampleInputs
            ExampleInputsStart(builder)
            ExampleInputsAddInputs(builder, inputs_vec_tensors)
            example_inputs_offset = ExampleInputsEnd(builder)

        # 5. Build the final Graph structure
        GraphStart(builder)
        if example_inputs_offset is not None:
            GraphAddExampleInputs(builder, example_inputs_offset)
        GraphAddNodes(builder, nodes_vec)
        GraphAddInputNames(builder, inputs_vec)
        GraphAddOutputNames(builder, outputs_vec)

        graph_offset = GraphEnd(builder)

        # 6. Finish and validate
        if graph_offset <= 0:
            raise ValueError(f"Invalid graph offset: {graph_offset}")

        builder.Finish(graph_offset)

        # 7. Get output and validate
        output = builder.Output()
        if len(output) <= 0:
            raise ValueError("Generated buffer is empty")

        return bytes(output)

    except Exception as e:
        raise RuntimeError(f"Failed to serialize graph: {e}") from e


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
