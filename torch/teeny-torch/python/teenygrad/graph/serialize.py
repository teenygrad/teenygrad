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

"""Serialize FX graphs to flatbuffers"""

import json
from typing import Any

import flatbuffers  # type: ignore

import torch

# Import the generated flatbuffer classes and functions
from .FXGraph.DType import DType

# Import the required classes and functions
from .FXGraph.ExampleInput import (
    ExampleInputAddType,
    ExampleInputAddValue,
    ExampleInputAddValueType,
    ExampleInputEnd,
    ExampleInputStart,
)
from .FXGraph.ExampleInputs import (
    ExampleInputsAddInputs,
    ExampleInputsEnd,
    ExampleInputsStart,
    ExampleInputsStartInputsVector,
)
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
from .FXGraph.InputType import InputType
from .FXGraph.InputValue import InputValue
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
from .FXGraph.SymInt import SymIntAddId, SymIntEnd, SymIntStart
from .FXGraph.Tensor import (
    TensorAddDevice,
    TensorAddDtype,
    TensorAddShape,
    TensorEnd,
    TensorStart,
)


def serialize_fx_graph(gm: torch.fx.GraphModule, example_inputs: list[Any] | None = None) -> bytes:
    """Serialize a torch.fx.GraphModule to a flatbuffer."""

    print_fx_graph(gm, example_inputs)

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
        example_inputs_offset = serialize_example_inputs(builder, example_inputs)

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


def print_fx_graph(gm: torch.fx.GraphModule, example_inputs: list[Any] | None = None):
    """Print the fx graph and example inputs to a file in /tmp for debugging"""

    def serialize_fx_node(node):
        return {
            "name": getattr(node, "name", None),
            "op": getattr(node, "op", None),
            "target": str(getattr(node, "target", None)),
            "args": [str(a) for a in getattr(node, "args", [])],
            "kwargs": {str(k): str(v) for k, v in getattr(node, "kwargs", {}).items()},
            "users": [str(u) for u in getattr(node, "users", [])],
        }

    graph_dict = {
        "nodes": [serialize_fx_node(node) for node in gm.graph.nodes],
    }

    def serialize_example_input(inp):
        if hasattr(inp, "shape") and hasattr(inp, "dtype"):
            return {
                "type": "tensor",
                "shape": list(inp.shape),
                "dtype": str(inp.dtype),
                "device": str(inp.device) if hasattr(inp, "device") else None,
            }
        elif hasattr(inp, "__class__") and inp.__class__.__name__ == "SymInt":
            return {
                "type": "symint",
                "value": str(inp),
            }
        else:
            return str(inp)

    print("--------------------------------")
    print(json.dumps({
        "graph": graph_dict,
        "example_inputs": [serialize_example_input(inp) for inp in example_inputs] if example_inputs else [],
    }, indent=2, sort_keys=True))
    print("--------------------------------")


def serialize_example_inputs(builder: flatbuffers.Builder, example_inputs: list[Any] | None) -> int | None:
    """Serialize example inputs to a flatbuffer.

    Returns the offset to the ExampleInputs table, or None if no inputs.
    """
    if not example_inputs or len(example_inputs) == 0:
        return None

    sample_input_offsets = []
    for arg in example_inputs:
        if isinstance(arg, torch.Tensor):
            # Build shape
            shape = list(arg.shape)
            ShapeStartDimsVector(builder, len(shape))
            for dim in reversed(shape):
                builder.PrependUint32(dim)
            shape_dims_vec = builder.EndVector()

            ShapeStart(builder)
            ShapeAddDims(builder, shape_dims_vec)
            shape_offset = ShapeEnd(builder)

            # Map device and dtype
            device_str = builder.CreateString(str(arg.device))

            dtype_mapping = {
                torch.float64: DType.FLOAT64,
                torch.float32: DType.FLOAT32,
                torch.int64: DType.INT64,
                torch.int32: DType.INT32,
                torch.int8: DType.INT8,
                torch.uint8: DType.UINT8,
                torch.bool: DType.BOOL,
                torch.bfloat16: DType.BFLOAT16,
                torch.float16: DType.FLOAT16,
            }
            dtype_enum = dtype_mapping.get(arg.dtype)
            if dtype_enum is None:
                raise ValueError(f"Unsupported dtype: {arg.dtype}")

            # Build tensor
            TensorStart(builder)
            TensorAddShape(builder, shape_offset)
            TensorAddDtype(builder, dtype_enum)
            TensorAddDevice(builder, device_str)
            tensor_offset = TensorEnd(builder)

            # Build ExampleInput as a tensor
            ExampleInputStart(builder)
            ExampleInputAddType(builder, InputType.Tensor)
            ExampleInputAddValueType(builder, InputValue.tensor)
            ExampleInputAddValue(builder, tensor_offset)
            sample_input_offsets.append(ExampleInputEnd(builder))

        elif isinstance(arg, torch.SymInt):
            # Build SymInt with string ID
            symint_id = builder.CreateString(str(arg))

            SymIntStart(builder)
            SymIntAddId(builder, symint_id)
            symint_offset = SymIntEnd(builder)

            # Build ExampleInput as a symint
            ExampleInputStart(builder)
            ExampleInputAddType(builder, InputType.SymInt)
            ExampleInputAddValueType(builder, InputValue.symint)
            ExampleInputAddValue(builder, symint_offset)
            sample_input_offsets.append(ExampleInputEnd(builder))

        else:
            raise ValueError(f"Expected tensor or SymInt, got {type(arg)}")

    # Build ExampleInputs vector
    ExampleInputsStartInputsVector(builder, len(sample_input_offsets))
    for offset in reversed(sample_input_offsets):
        builder.PrependUOffsetTRelative(offset)
    inputs_vec = builder.EndVector()

    # Build ExampleInputs
    ExampleInputsStart(builder)
    ExampleInputsAddInputs(builder, inputs_vec)
    example_inputs_offset = ExampleInputsEnd(builder)

    return example_inputs_offset
