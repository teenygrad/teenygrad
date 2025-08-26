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


from typing import Any

import flatbuffers  # type: ignore

import torch
from torch.fx.node import Argument

from .FXGraph.CallFunction import (
    CallFunctionAddArgs,
    CallFunctionAddKwargs,
    CallFunctionAddName,
    CallFunctionAddTarget,
    CallFunctionAddUsers,
    CallFunctionEnd,
    CallFunctionStart,
    CallFunctionStartArgsVector,
    CallFunctionStartKwargsVector,
    CallFunctionStartUsersVector,
)
from .FXGraph.CallMethod import (
    CallMethodAddArgs,
    CallMethodAddKwargs,
    CallMethodAddName,
    CallMethodAddTarget,
    CallMethodAddUsers,
    CallMethodEnd,
    CallMethodStart,
    CallMethodStartArgsVector,
    CallMethodStartKwargsVector,
    CallMethodStartUsersVector,
)
from .FXGraph.CallModule import CallModuleAddName, CallModuleEnd, CallModuleStart

# Import the generated flatbuffer classes and functions
from .FXGraph.DType import DType
from .FXGraph.ExampleInput import ExampleInput
from .FXGraph.ExampleInputs import (
    ExampleInputsAddInputs,
    ExampleInputsEnd,
    ExampleInputsStart,
    ExampleInputsStartInputsVector,
)
from .FXGraph.ExampleInputWrapper import (
    ExampleInputWrapperAddValue,
    ExampleInputWrapperAddValueType,
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
from .FXGraph.KeyValue import (
    KeyValueAddKey,
    KeyValueAddValue,
    KeyValueEnd,
    KeyValueStart,
)
from .FXGraph.Node import Node
from .FXGraph.NodeWrapper import (
    NodeWrapperAddNode,
    NodeWrapperAddNodeType,
    NodeWrapperEnd,
    NodeWrapperStart,
)
from .FXGraph.Output import OutputAddName, OutputEnd, OutputStart
from .FXGraph.PlaceholderWrapper import (
    PlaceholderWrapperAddName,
    PlaceholderWrapperAddTarget,
    PlaceholderWrapperAddUsers,
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
from .FXGraph.Value import Value
from .FXGraph.ValueWrapper import (
    ValueWrapperAddValue,
    ValueWrapperAddValueType,
    ValueWrapperEnd,
    ValueWrapperStart,
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

    # Build the PlaceholderWrapper
    PlaceholderWrapperStart(builder)
    PlaceholderWrapperAddName(builder, name)
    PlaceholderWrapperAddTarget(builder, target)
    PlaceholderWrapperAddUsers(builder, users_vec)
    return PlaceholderWrapperEnd(builder)


def _build_value_wrapper_for_arg(builder: flatbuffers.Builder, arg: Any) -> int:
    """Build a ValueWrapper for a function argument."""
    if isinstance(arg, (int, float)):
        # Create ValInt for numeric values
        ValIntStart(builder)
        ValIntAddValue(builder, int(arg))
        val_int_offset = ValIntEnd(builder)

        # Create ValueWrapper
        ValueWrapperStart(builder)
        ValueWrapperAddValueType(builder, Value.valint)
        ValueWrapperAddValue(builder, val_int_offset)
        return ValueWrapperEnd(builder)
    else:
        # For other types, create a placeholder ValueWrapper
        ValueWrapperStart(builder)
        ValueWrapperAddValueType(builder, Value.NONE)
        ValueWrapperAddValue(builder, 0)  # No value for now
        return ValueWrapperEnd(builder)


def _build_args_vector(builder: flatbuffers.Builder, args: tuple[Argument, ...], vector_start_func) -> int:
    """Build a vector of ValueWrapper objects for function arguments."""
    if not args:
        return 0

    args_offsets = []
    for arg in args:
        args_offsets.append(_build_value_wrapper_for_arg(builder, arg))

    # Build args vector
    vector_start_func(builder, len(args_offsets))
    for arg_offset in reversed(args_offsets):
        builder.PrependUOffsetTRelative(arg_offset)
    return builder.EndVector()


def _build_kwargs_vector(builder: flatbuffers.Builder, kwargs: dict[str, Any], vector_start_func) -> int:
    """Build a vector of KeyValue objects for function keyword arguments."""
    if not kwargs:
        return 0

    kwargs_offsets = []
    for key, value in kwargs.items():
        key_str = builder.CreateString(str(key))

        # Create ValueWrapper for the value
        value_wrapper_offset = _build_value_wrapper_for_arg(builder, value)

        # Create KeyValue
        KeyValueStart(builder)
        KeyValueAddKey(builder, key_str)
        KeyValueAddValue(builder, value_wrapper_offset)
        kwargs_offsets.append(KeyValueEnd(builder))

    # Build kwargs vector
    vector_start_func(builder, len(kwargs_offsets))
    for kwarg_offset in reversed(kwargs_offsets):
        builder.PrependUOffsetTRelative(kwarg_offset)
    return builder.EndVector()


def _build_users_vector(builder: flatbuffers.Builder, users: dict[torch.fx.node.Node, None], vector_start_func) -> int:
    """Build a vector of user names for a node."""
    if not users:
        return 0

    user_strings = [builder.CreateString(str(u.name)) for u in users if hasattr(u, 'name')]
    if not user_strings:
        return 0

    vector_start_func(builder, len(user_strings))
    for user in reversed(user_strings):
        builder.PrependUOffsetTRelative(user)
    return builder.EndVector()


def build_call_function_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a CallFunction node."""
    name = builder.CreateString(str(node.name))
    target = builder.CreateString(str(node.target))

    # Build vectors using helper functions
    args_vec = _build_args_vector(builder, node.args, CallFunctionStartArgsVector)
    kwargs_vec = _build_kwargs_vector(builder, node.kwargs, CallFunctionStartKwargsVector)
    users_vec = _build_users_vector(builder, node.users, CallFunctionStartUsersVector)

    # Build the complete CallFunction
    CallFunctionStart(builder)
    CallFunctionAddName(builder, name)
    CallFunctionAddTarget(builder, target)
    CallFunctionAddArgs(builder, args_vec)
    CallFunctionAddKwargs(builder, kwargs_vec)
    CallFunctionAddUsers(builder, users_vec)
    return CallFunctionEnd(builder)


def build_call_method_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a CallMethod node."""
    name = builder.CreateString(str(node.name))
    target = builder.CreateString(str(node.target))

    # Build vectors using helper functions
    args_vec = _build_args_vector(builder, node.args, CallMethodStartArgsVector)
    kwargs_vec = _build_kwargs_vector(builder, node.kwargs, CallMethodStartKwargsVector)
    users_vec = _build_users_vector(builder, node.users, CallMethodStartUsersVector)

    # Build the complete CallMethod
    CallMethodStart(builder)
    CallMethodAddName(builder, name)
    CallMethodAddTarget(builder, target)
    CallMethodAddArgs(builder, args_vec)
    CallMethodAddKwargs(builder, kwargs_vec)
    CallMethodAddUsers(builder, users_vec)
    return CallMethodEnd(builder)


def build_call_module_node(builder: flatbuffers.Builder, node: torch.fx.Node) -> int:
    """Build a CallModule node."""
    name = builder.CreateString(str(node.name))
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
            "users": [str(u) for u in getattr(node, "users", [])]
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
                "device": str(inp.device) if hasattr(inp, "device") else None
            }
        elif hasattr(inp, "__class__") and inp.__class__.__name__ == "SymInt":
            return {
                "type": "symint",
                "value": str(inp),
            }
        else:
            return {
                "type": "unknown",
                "value": str(inp)
            }

    print("--------------------------------")
    print({
        "graph": graph_dict,
        "example_inputs": [serialize_example_input(inp) for inp in example_inputs] if example_inputs else [],
    })
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
            # Build shape - schema only supports SymInt, so convert all dimensions to SymInt
            shape_dims = []
            for dim in arg.shape:
                # Convert all dimensions to SymInt as required by the schema
                # This includes both symbolic and concrete dimensions
                symint_value = builder.CreateString(str(dim))
                SymIntStart(builder)
                SymIntAddValue(builder, symint_value)
                symint_offset = SymIntEnd(builder)
                shape_dims.append(symint_offset)

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
            # Convert negative strides to positive uint32 values
            # PyTorch can have negative strides, but FlatBuffers uint expects positive values
            stride_uint32 = []
            for s in stride:
                if s < 0:
                    # Convert negative stride to positive by taking absolute value
                    # This is a common approach in serialization
                    stride_uint32.append(abs(s))
                else:
                    stride_uint32.append(s)

            TensorStartStrideVector(builder, len(stride_uint32))
            for s in reversed(stride_uint32):
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
            ExampleInputWrapperAddValueType(builder, ExampleInput.tensor)  # ExampleInput.tensor = 2
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
            ExampleInputWrapperAddValueType(builder, ExampleInput.symint)  # ExampleInput.symint = 1
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
