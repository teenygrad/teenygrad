/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

use protobuf::{CodedInputStream, Enum, Message};
use teeny_core::graph::{DtypeRepr, Graph, Op, Shape};

use crate::{
    errors::{Error, Result},
    onnx::onnx_proto3::{ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto},
};

include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));

pub struct Onnx {}

impl Onnx {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Graph> {
        let mut file = File::open(path)?;
        Self::from_reader(&mut file)
    }

    pub fn from_reader(reader: &mut impl Read) -> Result<Graph> {
        let mut reader = CodedInputStream::new(reader);
        let model = ModelProto::parse_from(&mut reader)?;
        let onnx_graph = model
            .graph
            .as_ref()
            .ok_or_else(|| Error::InvalidModel("Model is missing graph field".to_string()))?;

        let mut graph = Graph::new();
        let mut value_to_node: BTreeMap<String, usize> = BTreeMap::new();

        let mut value_info_map: BTreeMap<String, (DtypeRepr, Shape)> = BTreeMap::new();
        for info in &onnx_graph.input {
            if let Some(meta) = value_info_meta(info)? {
                value_info_map.insert(info.name.clone(), meta);
            }
        }
        for info in &onnx_graph.value_info {
            if let Some(meta) = value_info_meta(info)? {
                value_info_map.insert(info.name.clone(), meta);
            }
        }
        for info in &onnx_graph.output {
            if let Some(meta) = value_info_meta(info)? {
                value_info_map.insert(info.name.clone(), meta);
            }
        }

        let initializer_names: BTreeSet<&str> = onnx_graph
            .initializer
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        let initializer_map: BTreeMap<&str, &TensorProto> = onnx_graph
            .initializer
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        for init in &onnx_graph.initializer {
            let dtype = tensor_dtype_to_dtype(init.data_type)?;
            let shape = dims_to_shape(&init.dims);
            let id = graph.add_node(Op::Input, vec![], dtype, shape);
            value_to_node.insert(init.name.clone(), id);
        }

        for input in &onnx_graph.input {
            if initializer_names.contains(input.name.as_str()) {
                continue;
            }
            let (dtype, shape) = value_info_map
                .get(&input.name)
                .cloned()
                .unwrap_or((DtypeRepr::F32, vec![]));
            let id = graph.add_node(Op::Input, vec![], dtype, shape);
            value_to_node.insert(input.name.clone(), id);
        }

        for node in &onnx_graph.node {
            let mut inputs = Vec::with_capacity(node.input.len());
            for input_name in &node.input {
                if input_name.is_empty() {
                    continue;
                }
                let input_id = value_to_node.get(input_name).copied().ok_or_else(|| {
                    Error::InvalidModel(format!(
                        "Node '{}' references unknown input '{}'",
                        node.name, input_name
                    ))
                })?;
                inputs.push(input_id);
            }

            if inputs.is_empty() {
                return Err(Error::InvalidModel(format!(
                    "Node '{}' has no resolvable inputs",
                    node.name
                ))
                .into());
            }

            let first_output = node
                .output
                .iter()
                .find(|name| !name.is_empty())
                .ok_or_else(|| {
                    Error::InvalidModel(format!("Node '{}' has no output names", node.name))
                })?;

            let (dtype, shape) = value_info_map
                .get(first_output)
                .cloned()
                .unwrap_or_else(|| {
                    let input_node = &graph.nodes[inputs[0]];
                    (input_node.dtype, input_node.shape.clone())
                });

            let op = map_node_op(node, &initializer_map)?;
            let node_id = graph.add_node(op, inputs, dtype, shape);

            if !node.name.is_empty() {
                graph.names.insert(node_id, node.name.clone());
            }

            for output_name in &node.output {
                if !output_name.is_empty() {
                    value_to_node.insert(output_name.clone(), node_id);
                }
            }
        }

        Ok(graph)
    }
}

fn value_info_meta(info: &ValueInfoProto) -> Result<Option<(DtypeRepr, Shape)>> {
    let ty = info
        .type_
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("ValueInfoProto has no type".to_string()))?;
    if !ty.has_tensor_type() {
        return Ok(None);
    }
    let tensor = ty.tensor_type();
    let dtype = tensor_dtype_to_dtype(tensor.elem_type)?;
    let shape = tensor.shape.as_ref().map_or_else(Vec::new, |shape| {
        shape
            .dim
            .iter()
            .map(|d| {
                if d.has_dim_value() && d.dim_value() >= 0 {
                    Some(d.dim_value() as usize)
                } else {
                    None
                }
            })
            .collect()
    });

    Ok(Some((dtype, shape)))
}

fn dims_to_shape(dims: &[i64]) -> Shape {
    dims.iter()
        .map(|d| if *d >= 0 { Some(*d as usize) } else { None })
        .collect()
}

fn tensor_dtype_to_dtype(dtype: i32) -> Result<DtypeRepr> {
    Ok(match tensor_proto::DataType::from_i32(dtype) {
        Some(tensor_proto::DataType::BOOL) => DtypeRepr::Bool,
        Some(tensor_proto::DataType::INT8) => DtypeRepr::I8,
        Some(tensor_proto::DataType::INT16) => DtypeRepr::I16,
        Some(tensor_proto::DataType::INT32) => DtypeRepr::I32,
        Some(tensor_proto::DataType::INT64) => DtypeRepr::I64,
        Some(tensor_proto::DataType::UINT8) => DtypeRepr::U8,
        Some(tensor_proto::DataType::UINT16) => DtypeRepr::U16,
        Some(tensor_proto::DataType::UINT32) => DtypeRepr::U32,
        Some(tensor_proto::DataType::UINT64) => DtypeRepr::U64,
        Some(tensor_proto::DataType::FLOAT16) => DtypeRepr::F16,
        Some(tensor_proto::DataType::BFLOAT16) => DtypeRepr::BF16,
        Some(tensor_proto::DataType::DOUBLE) => DtypeRepr::F64,
        Some(tensor_proto::DataType::FLOAT) | None => DtypeRepr::F32,
        Some(_) => {
            return Err(Error::InvalidModel(format!("Unsupported ONNX dtype: {}", dtype)).into());
        }
    })
}

fn get_attr_int(node: &NodeProto, name: &str, default: i64) -> i64 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

fn get_attr_ints<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

fn map_node_op(node: &NodeProto, initializers: &BTreeMap<&str, &TensorProto>) -> Result<Op> {
    let op = match node.op_type.as_str() {
        "Relu" => Op::Relu,
        "Sigmoid" => Op::Sigmoid,
        "Tanh" => Op::Tanh,
        "Add" => Op::Add,
        "Flatten" => Op::Flatten,
        "Softmax" => {
            let dim = get_attr_int(node, "axis", 1);
            Op::Softmax {
                dim: dim.max(0) as usize,
            }
        }
        "Conv" => {
            let weight_name = node.input.get(1).ok_or_else(|| {
                Error::InvalidModel(format!("Conv node '{}' missing weight input", node.name))
            })?;
            let weight = initializers.get(weight_name.as_str()).ok_or_else(|| {
                Error::InvalidModel(format!(
                    "Conv node '{}' requires initializer for weight '{}'",
                    node.name, weight_name
                ))
            })?;

            if weight.dims.len() != 4 {
                return Err(Error::InvalidModel(format!(
                    "Conv node '{}' expected 4D weights, found {:?}",
                    node.name, weight.dims
                ))
                .into());
            }

            let has_bias = node.input.get(2).is_some_and(|s| !s.is_empty());
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            let groups = get_attr_int(node, "group", 1).max(1) as usize;

            Op::Conv2d {
                in_channels: weight.dims[1] as usize * groups,
                out_channels: weight.dims[0] as usize,
                kernel_h: weight.dims[2] as usize,
                kernel_w: weight.dims[3] as usize,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                padding_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                padding_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                groups,
                has_bias,
            }
        }
        "MaxPool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            Op::MaxPool2d {
                kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                pad_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                pad_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
            }
        }
        "AveragePool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            Op::AvgPool2d {
                kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
            }
        }
        other => {
            return Err(Error::InvalidModel(format!(
                "Unsupported ONNX op '{}' in node '{}'",
                other, node.name
            ))
            .into());
        }
    };
    Ok(op)
}
