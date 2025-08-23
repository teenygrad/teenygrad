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

from .FXGraph.CallFunction import (
    CallFunctionAddName,
    CallFunctionEnd,
    CallFunctionStart,
)
from .FXGraph.CallMethod import CallMethodAddName, CallMethodEnd, CallMethodStart
from .FXGraph.CallModule import CallModuleAddName, CallModuleEnd, CallModuleStart

# Import the generated flatbuffer classes and functions
from .FXGraph.DType import DType
from .FXGraph.ExampleInputs import (
    ExampleInputsAddInputs,
    ExampleInputsEnd,
    ExampleInputsStart,
    ExampleInputsStartInputsVector,
)
from .FXGraph.ExampleInputWrapper import (
    ExampleInputWrapperAddValue,
    ExampleInputWrapperEnd,
    ExampleInputWrapperStart,
)
from .FXGraph.GetAttr import GetAttrAddName, GetAttrEnd, GetAttrStart
from .FXGraph.Graph import (
    GraphAddExampleInputs,
    GraphAddNodes,
    GraphEnd,
    GraphStart,
    GraphStartNodesVector,
)
from .FXGraph.Node import Node
from .FXGraph.NodeWrapper import (
    NodeWrapperAddNode,
    NodeWrapperAddNodeType,
    NodeWrapperEnd,
    NodeWrapperStart,
)
from .FXGraph.Output import OutputAddName, OutputEnd, OutputStart
from .FXGraph.Placeholder import Placeholder
from .FXGraph.PlaceholderWrapper import (
    PlaceholderWrapperAddName,
    PlaceholderWrapperAddTarget,
    PlaceholderWrapperAddUsers,
    PlaceholderWrapperAddValue,
    PlaceholderWrapperAddValueType,
    PlaceholderWrapperEnd,
    PlaceholderWrapperStart,
    PlaceholderWrapperStartUsersVector,
)
from .FXGraph.Shape import ShapeAddDims, ShapeEnd, ShapeStart, ShapeStartDimsVector
from .FXGraph.SymInt import SymIntAddValue, SymIntEnd, SymIntStart
from .FXGraph.Tensor import (
    TensorAddDevice,
    TensorAddDtype,
    TensorAddRequiresGrad,
    TensorAddShape,
    TensorAddStride,
    TensorEnd,
    TensorStart,
    TensorStartStrideVector,
)
from .FXGraph.ValInt import ValIntAddValue, ValIntEnd, ValIntStart


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

            # Build users vector
            users = [builder.CreateString(str(u.name)) for u in node.users if hasattr(u, 'name')]
            PlaceholderWrapperStartUsersVector(builder, len(users))
            for user in reversed(users):
                builder.PrependUOffsetTRelative(user)
            users_vec = builder.EndVector()

            # Build the node based on its operation type
            node_offset = None
            node_type = Node.NONE

            if node.op == "placeholder":
                node_type = Node.placeholder
                node_offset = build_placeholder_node(builder, node, users_vec)
            elif node.op == "call_function":
                node_type = Node.call_function
                node_offset = build_call_function_node(builder, node)
            elif node.op == "call_method":
                node_type = Node.call_method
                node_offset = build_call_method_node(builder, node)
            elif node.op == "call_module":
                node_type = Node.call_module
                node_offset = build_call_module_node(builder, node)
            elif node.op == "get_attr":
                node_type = Node.get_attr
                node_offset = build_get_attr_node(builder, node)
            elif node.op == "output":
                node_type = Node.output
                node_offset = build_output_node(builder, node)

            if node_offset is not None:
                # Build NodeWrapper
                NodeWrapperStart(builder)
                NodeWrapperAddNodeType(builder, node_type)
                NodeWrapperAddNode(builder, node_offset)
                node_wrapper_offset = NodeWrapperEnd(builder)
                node_offsets.append(node_wrapper_offset)

        if not node_offsets:
            raise ValueError("No valid nodes found for serialization")

        # 2. Build node vector
        GraphStartNodesVector(builder, len(node_offsets))
        for offset in reversed(node_offsets):  # Use reversed for proper flatbuffer construction
            builder.PrependUOffsetTRelative(offset)
        nodes_vec = builder.EndVector()

        # 3. Build example inputs if provided
        example_inputs_offset = serialize_example_inputs(builder, example_inputs)

        # 4. Build the final Graph structure
        GraphStart(builder)
        GraphAddNodes(builder, nodes_vec)
        if example_inputs_offset is not None:
            GraphAddExampleInputs(builder, example_inputs_offset)
        graph_offset = GraphEnd(builder)

        # 5. Finish and validate
        if graph_offset <= 0:
            raise ValueError(f"Invalid graph offset: {graph_offset}")

        builder.Finish(graph_offset)

        # 6. Get output and validate
        output = builder.Output()
        if len(output) <= 0:
            raise ValueError("Generated buffer is empty")

        return bytes(output)

    except Exception as e:
        raise RuntimeError(f"Failed to serialize graph: {e}") from e


def build_placeholder_node(builder: flatbuffers.Builder, node: torch.fx.Node, users_vec: int) -> int:
    """Build a PlaceholderWrapper node."""
    name = builder.CreateString(str(node.name))
    target = builder.CreateString(str(node.target))

    # For placeholder nodes, we need to determine the value type
    # This is typically a tensor or symint, but we'll use a placeholder for now
    value_type = Placeholder.NONE
    value_offset = 0

    # Build the PlaceholderWrapper
    PlaceholderWrapperStart(builder)
    PlaceholderWrapperAddName(builder, name)
    PlaceholderWrapperAddValueType(builder, value_type)
    PlaceholderWrapperAddValue(builder, value_offset)
    PlaceholderWrapperAddTarget(builder, target)
    PlaceholderWrapperAddUsers(builder, users_vec)
    return PlaceholderWrapperEnd(builder)


def build_call_function_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a CallFunction node."""
    name = builder.CreateString(str(node.target))
    CallFunctionStart(builder)
    CallFunctionAddName(builder, name)
    return CallFunctionEnd(builder)


def build_call_method_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a CallMethod node."""
    name = builder.CreateString(str(node.target))
    CallMethodStart(builder)
    CallMethodAddName(builder, name)
    return CallMethodEnd(builder)


def build_call_module_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a CallModule node."""
    name = builder.CreateString(str(node.target))
    CallModuleStart(builder)
    CallModuleAddName(builder, name)
    return CallModuleEnd(builder)


def build_get_attr_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a GetAttr node."""
    name = builder.CreateString(str(node.target))
    GetAttrStart(builder)
    GetAttrAddName(builder, name)
    return GetAttrEnd(builder)


def build_output_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build an Output node."""
    name = builder.CreateString(str(node.name))
    OutputStart(builder)
    OutputAddName(builder, name)
    return OutputEnd(builder)


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
            "_all_attrs": {k: v for k, v in node.__dict__.items() if not k.startswith("_")},
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
                "_all_attrs": {k: v for k, v in inp.__dict__.items() if not k.startswith("_")},
            }
        elif hasattr(inp, "__class__") and inp.__class__.__name__ == "SymInt":
            return {
                "type": "symint",
                "value": str(inp),
                "_all_attrs": {k: v for k, v in inp.__dict__.items() if not k.startswith("_")},
            }
        else:
            return {
                "type": "unknown",
                "value": str(inp),
                "_all_attrs": {k: v for k, v in inp.__dict__.items() if not k.startswith("_")},
            }

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
            shape_dims = []
            for dim in arg.shape:
                if hasattr(dim, '__class__') and dim.__class__.__name__ == "SymInt":
                    # Handle symbolic dimensions
                    symint_value = builder.CreateString(str(dim))
                    SymIntStart(builder)
                    SymIntAddValue(builder, symint_value)
                    symint_offset = SymIntEnd(builder)
                    shape_dims.append(symint_offset)
                else:
                    # Handle concrete dimensions
                    valint_value = int(dim)
                    ValIntStart(builder)
                    ValIntAddValue(builder, valint_value)
                    valint_offset = ValIntEnd(builder)
                    shape_dims.append(valint_offset)

            # Build shape vector
            ShapeStartDimsVector(builder, len(shape_dims))
            for dim_offset in reversed(shape_dims):
                builder.PrependUOffsetTRelative(dim_offset)
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

            # Build stride vector
            stride = list(arg.stride()) if hasattr(arg, 'stride') else []
            TensorStartStrideVector(builder, len(stride))
            for s in reversed(stride):
                builder.PrependUint32(s)
            stride_vec = builder.EndVector()

            # Build tensor
            TensorStart(builder)
            TensorAddShape(builder, shape_offset)
            TensorAddDtype(builder, dtype_enum)
            TensorAddDevice(builder, device_str)
            TensorAddStride(builder, stride_vec)
            TensorAddRequiresGrad(builder, getattr(arg, 'requires_grad', False))
            tensor_offset = TensorEnd(builder)

            # Build ExampleInputWrapper as a tensor
            ExampleInputWrapperStart(builder)
            ExampleInputWrapperAddValue(builder, tensor_offset)
            sample_input_offsets.append(ExampleInputWrapperEnd(builder))

        elif hasattr(arg, '__class__') and arg.__class__.__name__ == "SymInt":
            # Build SymInt with string value
            symint_value = builder.CreateString(str(arg))

            SymIntStart(builder)
            SymIntAddValue(builder, symint_value)
            symint_offset = SymIntEnd(builder)

            # Build ExampleInputWrapper as a symint
            ExampleInputWrapperStart(builder)
            ExampleInputWrapperAddValue(builder, symint_offset)
            sample_input_offsets.append(ExampleInputWrapperEnd(builder))

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
    # Note: kwargs are not implemented in this version as they're not part of the basic schema
    example_inputs_offset = ExampleInputsEnd(builder)

    return example_inputs_offset
