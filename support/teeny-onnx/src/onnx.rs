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

            // Some ops (Constant, SequenceEmpty, OptionalHasElement) have zero
            // tensor inputs by design — allow them through.
            let zero_input_allowed = matches!(
                node.op_type.as_str(),
                "Constant" | "SequenceEmpty" | "OptionalHasElement"
            );
            if inputs.is_empty() && !zero_input_allowed {
                return Err(Error::InvalidModel(format!(
                    "Node '{}' (op='{}') has no resolvable inputs",
                    node.name, node.op_type
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
                    if !inputs.is_empty() {
                        let input_node = &graph.nodes[inputs[0]];
                        (input_node.dtype, input_node.shape.clone())
                    } else {
                        // Zero-input op with no value_info — fall back to F32 scalar.
                        (DtypeRepr::F32, vec![])
                    }
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
        Some(tensor_proto::DataType::BOOL)    => DtypeRepr::Bool,
        Some(tensor_proto::DataType::INT8)    => DtypeRepr::I8,
        Some(tensor_proto::DataType::INT16)   => DtypeRepr::I16,
        Some(tensor_proto::DataType::INT32)   => DtypeRepr::I32,
        Some(tensor_proto::DataType::INT64)   => DtypeRepr::I64,
        Some(tensor_proto::DataType::UINT8)   => DtypeRepr::U8,
        Some(tensor_proto::DataType::UINT16)  => DtypeRepr::U16,
        Some(tensor_proto::DataType::UINT32)  => DtypeRepr::U32,
        Some(tensor_proto::DataType::UINT64)  => DtypeRepr::U64,
        Some(tensor_proto::DataType::FLOAT16) => DtypeRepr::F16,
        Some(tensor_proto::DataType::BFLOAT16)=> DtypeRepr::BF16,
        Some(tensor_proto::DataType::DOUBLE)  => DtypeRepr::F64,
        Some(tensor_proto::DataType::FLOAT) | None => DtypeRepr::F32,
        // STRING tensors — represent as U8 (byte storage) for graph purposes.
        Some(tensor_proto::DataType::STRING) => DtypeRepr::U8,
        // Sub-byte and float8 types — map to nearest supported type.
        Some(tensor_proto::DataType::FLOAT8E4M3FN)
        | Some(tensor_proto::DataType::FLOAT8E4M3FNUZ)
        | Some(tensor_proto::DataType::FLOAT8E5M2)
        | Some(tensor_proto::DataType::FLOAT8E5M2FNUZ) => DtypeRepr::F16,
        Some(_) => {
            // Unknown or newer sub-byte types (INT4, UINT4, INT2, UINT2, FLOAT4…).
            // Map to I8/U8 as a safe placeholder for graph-building purposes.
            if dtype % 2 == 0 {
                DtypeRepr::U8
            } else {
                DtypeRepr::I8
            }
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

fn get_attr_float(node: &NodeProto, name: &str, default: f32) -> f32 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.f)
        .unwrap_or(default)
}

fn get_attr_ints<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

fn get_attr_string<'a>(node: &'a NodeProto, name: &str, default: &'a str) -> &'a str {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| std::str::from_utf8(&a.s).ok())
        .unwrap_or(default)
}

fn map_node_op(node: &NodeProto, initializers: &BTreeMap<&str, &TensorProto>) -> Result<Op> {
    let op = match node.op_type.as_str() {
        // ---------------------------------------------------------------
        // Existing ops (previously supported)
        // ---------------------------------------------------------------
        "Relu" => Op::Relu,
        "Sigmoid" => Op::Sigmoid,
        "Tanh" => Op::Tanh,
        "Add" => Op::Add,
        "Flatten" => {
            Op::Flatten
        }
        "Softmax" => {
            let axis = get_attr_int(node, "axis", 1);
            Op::Softmax { dim: axis.max(0) as usize }
        }

        "Conv" => {
            let weight_name = node.input.get(1).map(|s| s.as_str()).unwrap_or("");
            // Weight dims: [out_ch, in_ch/groups, k...] if from an initializer,
            // otherwise inferred from the kernel_shape attribute.
            let weight_dims: Vec<i64> = if let Some(w) = initializers.get(weight_name) {
                w.dims.clone()
            } else {
                let ks = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
                let mut d = vec![0i64, 0i64]; // out/in channels unknown
                d.extend_from_slice(ks);
                d
            };

            let has_bias = node.input.get(2).is_some_and(|s| !s.is_empty());
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            let groups = get_attr_int(node, "group", 1).max(1) as usize;
            let weight = &weight_dims[..];
            let ndim = weight.len();

            if ndim == 3 {
                Op::Conv1d {
                    in_channels: weight.get(1).copied().unwrap_or(0) as usize * groups,
                    out_channels: weight.first().copied().unwrap_or(0) as usize,
                    kernel_l: weight.get(2).copied().unwrap_or(1) as usize,
                    stride: strides.first().copied().unwrap_or(1).max(1) as usize,
                    padding: pads.first().copied().unwrap_or(0).max(0) as usize,
                    has_bias,
                }
            } else if ndim == 4 {
                Op::Conv2d {
                    in_channels: weight.get(1).copied().unwrap_or(0) as usize * groups,
                    out_channels: weight.first().copied().unwrap_or(0) as usize,
                    kernel_h: weight.get(2).copied().unwrap_or(1) as usize,
                    kernel_w: weight.get(3).copied().unwrap_or(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    padding_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                    padding_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                    groups,
                    has_bias,
                }
            } else if ndim == 5 {
                Op::Conv3d {
                    in_channels: weight.get(1).copied().unwrap_or(0) as usize * groups,
                    out_channels: weight.first().copied().unwrap_or(0) as usize,
                    kernel_d: weight.get(2).copied().unwrap_or(1) as usize,
                    kernel_h: weight.get(3).copied().unwrap_or(1) as usize,
                    kernel_w: weight.get(4).copied().unwrap_or(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(1).max(1) as usize,
                    padding_d: pads.first().copied().unwrap_or(0).max(0) as usize,
                    padding_h: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                    padding_w: pads.get(2).copied().unwrap_or(0).max(0) as usize,
                    has_bias,
                }
            } else {
                return Err(Error::InvalidModel(format!(
                    "Conv node '{}': unsupported weight rank {}",
                    node.name, ndim
                ))
                .into());
            }
        }

        "ConvTranspose" => {
            let weight_name = node.input.get(1).map(|s| s.as_str()).unwrap_or("");
            let (out_channels, kernel_h, kernel_w) =
                if let Some(w) = initializers.get(weight_name) {
                    let nd = w.dims.len();
                    if nd >= 4 {
                        (w.dims[1] as usize, w.dims[2] as usize, w.dims[3] as usize)
                    } else {
                        (0, 1, 1)
                    }
                } else {
                    (0, 1, 1)
                };
            let has_bias = node.input.get(2).is_some_and(|s| !s.is_empty());
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            let out_pads = get_attr_ints(node, "output_padding").unwrap_or(&[0, 0]);
            let groups = get_attr_int(node, "group", 1).max(1) as usize;
            let in_channels = if let Some(w) = initializers.get(weight_name) {
                w.dims.first().copied().unwrap_or(0) as usize * groups
            } else {
                0
            };
            Op::ConvTranspose {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                padding_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                padding_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                output_padding_h: out_pads.first().copied().unwrap_or(0).max(0) as usize,
                output_padding_w: out_pads.get(1).copied().unwrap_or(0).max(0) as usize,
                groups,
                has_bias,
            }
        }

        "MaxPool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(kernel);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            match kernel.len() {
                1 => Op::MaxPool1d {
                    kernel_l: kernel[0].max(1) as usize,
                    stride: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                },
                3 => Op::MaxPool3d {
                    kernel_d: kernel[0].max(1) as usize,
                    kernel_h: kernel[1].max(1) as usize,
                    kernel_w: kernel[2].max(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(kernel[1]).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(kernel[2]).max(1) as usize,
                },
                _ => Op::MaxPool2d {
                    kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                    kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    pad_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                    pad_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                },
            }
        }

        "AveragePool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(kernel);
            match kernel.len() {
                1 => Op::AvgPool1d {
                    kernel_l: kernel[0].max(1) as usize,
                    stride: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                },
                3 => Op::AvgPool3d {
                    kernel_d: kernel[0].max(1) as usize,
                    kernel_h: kernel[1].max(1) as usize,
                    kernel_w: kernel[2].max(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(kernel[1]).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(kernel[2]).max(1) as usize,
                },
                _ => Op::AvgPool2d {
                    kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                    kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                },
            }
        }

        "GlobalAveragePool" => Op::GlobalAvgPool,
        "GlobalMaxPool"     => Op::GlobalMaxPool,

        "LpPool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(kernel);
            let p = get_attr_int(node, "p", 2) as f64;
            match kernel.len() {
                1 => Op::LpPool1d {
                    kernel_l: kernel[0].max(1) as usize,
                    stride: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    p,
                },
                3 => Op::LpPool3d {
                    kernel_d: kernel[0].max(1) as usize,
                    kernel_h: kernel[1].max(1) as usize,
                    kernel_w: kernel[2].max(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(kernel[1]).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(kernel[2]).max(1) as usize,
                    p,
                },
                _ => Op::LpPool2d {
                    kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                    kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    p,
                },
            }
        }

        "MaxUnpool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(kernel);
            Op::MaxUnpool {
                kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
            }
        }

        // ---------------------------------------------------------------
        // Normalisation
        // ---------------------------------------------------------------
        "BatchNormalization" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            let momentum = get_attr_float(node, "momentum", 0.9) as f64;
            // Use BatchNorm2d as the representative variant (most common in practice).
            Op::BatchNorm2d {
                num_features: 0,
                eps,
                momentum,
                affine: true,
                track_running_stats: true,
            }
        }

        "LayerNormalization" | "LayerNorm" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::LayerNorm { normalized_shape: vec![], eps, affine: true }
        }

        "RMSNormalization" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::RmsNorm { normalized_shape: vec![], eps, affine: true }
        }

        "GroupNormalization" => {
            let num_groups = get_attr_int(node, "num_groups", 1).max(1) as usize;
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::GroupNorm { num_groups, num_channels: 0, eps, affine: true }
        }

        "InstanceNormalization" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::InstanceNorm2d {
                num_features: 0,
                eps,
                momentum: 0.1,
                affine: true,
                track_running_stats: false,
            }
        }

        "LRN" => {
            let alpha = get_attr_float(node, "alpha", 0.0001) as f64;
            let beta  = get_attr_float(node, "beta",  0.75)   as f64;
            let bias  = get_attr_float(node, "bias",  1.0)    as f64;
            let size  = get_attr_int(node, "size", 1).max(1) as usize;
            Op::LRN { alpha, beta, bias, size }
        }

        "MeanVarianceNormalization" => {
            let axes = get_attr_ints(node, "axes").unwrap_or(&[0, 2, 3]).to_vec();
            Op::MeanVarianceNormalization { axes }
        }

        "LpNormalization" => {
            let axis = get_attr_int(node, "axis", -1);
            let p    = get_attr_int(node, "p", 2);
            Op::LpNormalization { axis, p }
        }

        // ---------------------------------------------------------------
        // Activations
        // ---------------------------------------------------------------
        "Elu"  => Op::Elu  { alpha: get_attr_float(node, "alpha", 1.0) as f64 },
        "Selu" => Op::Selu,
        "Celu" => Op::Celu { alpha: get_attr_float(node, "alpha", 1.0) as f64 },
        "Gelu" => Op::Gelu,
        "Mish" => Op::Mish,
        "HardSigmoid" => Op::Hardsigmoid,
        "HardSwish"   => Op::Hardswish,
        "Swish"       => Op::Swish,
        "LeakyRelu"   => Op::LeakyRelu {
            negative_slope: get_attr_float(node, "alpha", 0.01) as f64,
        },
        "Softplus"    => Op::Softplus { beta: 1.0, threshold: 20.0 },
        "Softsign"    => Op::Softsign,
        "LogSoftmax"  => Op::LogSoftmax { axis: get_attr_int(node, "axis", -1) },
        "Hardmax"     => Op::Hardmax    { axis: get_attr_int(node, "axis", -1) },
        "PRelu"       => Op::PRelu,
        "ThresholdedRelu" => Op::ThresholdedRelu {
            alpha: get_attr_float(node, "alpha", 1.0) as f64,
        },
        "Shrink" => Op::Shrink {
            lambd: get_attr_float(node, "lambd", 0.5) as f64,
            bias:  get_attr_float(node, "bias",  0.0) as f64,
        },
        "Clip" => Op::Clip,

        // ---------------------------------------------------------------
        // Element-wise unary
        // ---------------------------------------------------------------
        "Abs"        => Op::Abs,
        "Neg"        => Op::Neg,
        "Ceil"       => Op::Ceil,
        "Floor"      => Op::Floor,
        "Round"      => Op::Round,
        "Sqrt"       => Op::Sqrt,
        "Reciprocal" => Op::Reciprocal,
        "Exp"        => Op::Exp,
        "Log"        => Op::Log,
        "Erf"        => Op::Erf,
        "Sign"       => Op::Sign,
        "IsNaN"      => Op::IsNaN,
        "IsInf"      => Op::IsInf {
            detect_negative: get_attr_int(node, "detect_negative", 1) != 0,
            detect_positive: get_attr_int(node, "detect_positive", 1) != 0,
        },
        "Not"        => Op::Not,
        "BitwiseNot" => Op::BitwiseNot,
        "Sin"        => Op::Sin,
        "Cos"        => Op::Cos,
        "Tan"        => Op::Tan,
        "Asin"       => Op::Asin,
        "Acos"       => Op::Acos,
        "Atan"       => Op::Atan,
        "Sinh"       => Op::Sinh,
        "Cosh"       => Op::Cosh,
        "Asinh"      => Op::Asinh,
        "Acosh"      => Op::Acosh,
        "Atanh"      => Op::Atanh,

        // ---------------------------------------------------------------
        // Element-wise binary / variadic
        // ---------------------------------------------------------------
        "Mul"  => Op::Mul,
        "Sub"  => Op::Sub,
        "Div"  => Op::Div,
        "Pow"  => Op::Pow,
        "Mod"  => Op::Mod { fmod: get_attr_int(node, "fmod", 0) != 0 },
        "Min"  => Op::ElemMin,
        "Max"  => Op::ElemMax,
        "Mean" => Op::ElemMean,
        "Sum"  => Op::ElemSum,
        "Equal"          => Op::Equal,
        "Greater"        => Op::Greater,
        "GreaterOrEqual" => Op::GreaterOrEqual,
        "Less"           => Op::Less,
        "LessOrEqual"    => Op::LessOrEqual,
        "And"  => Op::And,
        "Or"   => Op::Or,
        "Xor"  => Op::Xor,
        "BitwiseAnd" => Op::BitwiseAnd,
        "BitwiseOr"  => Op::BitwiseOr,
        "BitwiseXor" => Op::BitwiseXor,
        "BitShift" => Op::BitShift {
            direction: get_attr_string(node, "direction", "LEFT").to_string(),
        },

        // ---------------------------------------------------------------
        // Tensor structural
        // ---------------------------------------------------------------
        "Reshape"  => Op::Reshape,
        "Transpose" => Op::Transpose {
            perm: get_attr_ints(node, "perm")
                .map(|v| v.iter().map(|&i| i.max(0) as usize).collect())
                .unwrap_or_default(),
        },
        "Squeeze" => Op::Squeeze {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
        },
        "Unsqueeze" => Op::Unsqueeze {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
        },
        "Concat" => Op::Concat { axis: get_attr_int(node, "axis", 0) },
        "Split"  => Op::Split  {
            axis: get_attr_int(node, "axis", 0),
            num_outputs: node.output.len().max(1),
        },
        "Slice"         => Op::Slice,
        "Gather"        => Op::Gather        { axis: get_attr_int(node, "axis", 0) },
        "GatherElements"=> Op::GatherElements{ axis: get_attr_int(node, "axis", 0) },
        "GatherND"      => Op::GatherND      { batch_dims: get_attr_int(node, "batch_dims", 0) },
        "ScatterElements" => Op::ScatterElements {
            axis: get_attr_int(node, "axis", 0),
        },
        "ScatterND"      => Op::ScatterND,
        "Scatter"        => Op::Scatter { axis: get_attr_int(node, "axis", 0) },
        "TensorScatter"  => Op::TensorScatter,
        "Tile"           => Op::Tile,
        "Expand"         => Op::Expand,
        "Shape"          => Op::ShapeOf {
            start: get_attr_int(node, "start", 0),
            end:   get_attr_int(node, "end",   i64::MAX),
        },
        "Size"           => Op::SizeOf,
        "Identity"       => Op::Identity,
        "Cast"           => {
            let to = get_attr_int(node, "to", 1);
            let dtype = tensor_dtype_to_dtype(to as i32)?;
            Op::Cast { to: dtype }
        }
        "CastLike"       => Op::CastLike,
        "Where"          => Op::Where,
        "Compress"       => Op::Compress { axis: get_attr_int(node, "axis", 0) },
        "Range"          => Op::Range,
        "Constant"       => {
            // Shape/dtype extracted from the tensor attribute.
            if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                let tensor = &attr.t;
                let dtype  = tensor_dtype_to_dtype(tensor.data_type)?;
                let shape  = dims_to_shape(&tensor.dims);
                Op::Constant { dtype, shape }
            } else {
                Op::Constant { dtype: DtypeRepr::F32, shape: vec![] }
            }
        }
        "ConstantOfShape" => {
            let dtype = if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                tensor_dtype_to_dtype(attr.t.data_type)?
            } else {
                DtypeRepr::F32
            };
            Op::ConstantOfShape { dtype }
        }
        "Trilu" => Op::Trilu { upper: get_attr_int(node, "upper", 1) != 0 },
        "BitCast" => {
            let to = get_attr_int(node, "to", 1);
            let dtype = tensor_dtype_to_dtype(to as i32)?;
            Op::BitCast { to: dtype }
        }
        "Pad" => Op::Pad {
            mode: get_attr_string(node, "mode", "constant").to_string(),
        },
        "ReverseSequence" => Op::ReverseSequence {
            batch_axis: get_attr_int(node, "batch_axis", 1),
            time_axis:  get_attr_int(node, "time_axis",  0),
        },
        "NonZero" => Op::NonZero,

        // ---------------------------------------------------------------
        // Matrix
        // ---------------------------------------------------------------
        "Gemm" => Op::Gemm {
            alpha:   get_attr_float(node, "alpha",   1.0) as f64,
            beta:    get_attr_float(node, "beta",    1.0) as f64,
            trans_a: get_attr_int(node, "transA", 0) != 0,
            trans_b: get_attr_int(node, "transB", 0) != 0,
        },
        "MatMul"        => Op::MatMul,
        "MatMulInteger" => Op::MatMulInteger,
        "Einsum"        => Op::Einsum {
            equation: get_attr_string(node, "equation", "").to_string(),
        },
        "Det"           => Op::Det,
        "QLinearMatMul" => Op::QLinearMatMul,
        "ConvInteger"   => Op::ConvInteger {
            groups: get_attr_int(node, "group", 1).max(1) as usize,
        },
        "DeformConv"    => Op::DeformConv {
            group:        get_attr_int(node, "group",        1).max(1) as usize,
            offset_group: get_attr_int(node, "offset_group", 1).max(1) as usize,
        },
        "QLinearConv"   => Op::QLinearConv {
            groups: get_attr_int(node, "group", 1).max(1) as usize,
        },
        "Col2Im" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            Op::Col2Im {
                kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
            }
        }

        // ---------------------------------------------------------------
        // Reductions
        // ---------------------------------------------------------------
        "ReduceSum" => Op::ReduceSum {
            keepdims:              get_attr_int(node, "keepdims",              1) != 0,
            noop_with_empty_axes:  get_attr_int(node, "noop_with_empty_axes",  0) != 0,
        },
        "ReduceMean" => Op::ReduceMean {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceMax" => Op::ReduceMax {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceMin" => Op::ReduceMin {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceProd" => Op::ReduceProd {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceL1" => Op::ReduceL1 {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceL2" => Op::ReduceL2 {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceLogSum" => Op::ReduceLogSum {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceLogSumExp" => Op::ReduceLogSumExp {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceSumSquare" => Op::ReduceSumSquare {
            keepdims:             get_attr_int(node, "keepdims",             1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "CumSum"  => Op::CumSum {
            exclusive: get_attr_int(node, "exclusive", 0) != 0,
            reverse:   get_attr_int(node, "reverse",   0) != 0,
        },
        "CumProd" => Op::CumProd {
            exclusive: get_attr_int(node, "exclusive", 0) != 0,
            reverse:   get_attr_int(node, "reverse",   0) != 0,
        },
        "ArgMax" => Op::ArgMax {
            axis:              get_attr_int(node, "axis", 0),
            keepdims:          get_attr_int(node, "keepdims",          1) != 0,
            select_last_index: get_attr_int(node, "select_last_index", 0) != 0,
        },
        "ArgMin" => Op::ArgMin {
            axis:              get_attr_int(node, "axis", 0),
            keepdims:          get_attr_int(node, "keepdims",          1) != 0,
            select_last_index: get_attr_int(node, "select_last_index", 0) != 0,
        },

        // ---------------------------------------------------------------
        // Resize / spatial
        // ---------------------------------------------------------------
        "Upsample" | "Resize" => Op::Resize {
            mode: get_attr_string(node, "mode", "nearest").to_string(),
            coordinate_transformation_mode: get_attr_string(
                node, "coordinate_transformation_mode", "asymmetric",
            ).to_string(),
            antialias: get_attr_int(node, "antialias", 0) != 0,
        },

        "GridSample" => Op::GridSample {
            mode:         get_attr_string(node, "mode",         "bilinear").to_string(),
            padding_mode: get_attr_string(node, "padding_mode", "zeros").to_string(),
            align_corners: get_attr_int(node, "align_corners", 0) != 0,
        },

        "SpaceToDepth" => Op::SpaceToDepth {
            blocksize: get_attr_int(node, "blocksize", 2).max(1) as usize,
        },
        "DepthToSpace" => Op::DepthToSpace {
            blocksize: get_attr_int(node, "blocksize", 2).max(1) as usize,
            mode:      get_attr_string(node, "mode", "DCR").to_string(),
        },

        "RoiAlign" => Op::RoiAlign {
            output_h:      get_attr_int(node, "output_height",  1).max(1) as usize,
            output_w:      get_attr_int(node, "output_width",   1).max(1) as usize,
            sampling_ratio:get_attr_int(node, "sampling_ratio", 0),
            spatial_scale: get_attr_float(node, "spatial_scale", 1.0) as f64,
        },

        "AffineGrid" => Op::AffineGrid {
            align_corners: get_attr_int(node, "align_corners", 0) != 0,
        },

        "CenterCropPad" => Op::CenterCropPad {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
        },

        "NonMaxSuppression" => Op::NonMaxSuppression {
            center_point_box: get_attr_int(node, "center_point_box", 0) != 0,
        },

        // ---------------------------------------------------------------
        // Recurrent
        // ---------------------------------------------------------------
        "LSTM" => {
            let hidden_size  = get_attr_int(node, "hidden_size", 1).max(1) as usize;
            let direction    = get_attr_string(node, "direction", "forward").to_string();
            let bidirectional = direction == "bidirectional";
            Op::Lstm { hidden_size, direction, bidirectional }
        }
        "GRU" => {
            let hidden_size  = get_attr_int(node, "hidden_size", 1).max(1) as usize;
            let direction    = get_attr_string(node, "direction", "forward").to_string();
            let bidirectional = direction == "bidirectional";
            Op::Gru { hidden_size, direction, bidirectional }
        }
        "RNN" => {
            let hidden_size  = get_attr_int(node, "hidden_size", 1).max(1) as usize;
            let direction    = get_attr_string(node, "direction", "forward").to_string();
            let bidirectional = direction == "bidirectional";
            Op::Rnn { hidden_size, direction, bidirectional }
        }

        // ---------------------------------------------------------------
        // Attention
        // ---------------------------------------------------------------
        "Attention" => Op::MultiHeadAttention {
            q_num_heads:  get_attr_int(node, "q_num_heads",  1).max(1) as usize,
            kv_num_heads: get_attr_int(node, "kv_num_heads", 1).max(1) as usize,
        },
        "RotaryEmbedding" => Op::RotaryEmbedding,

        // ---------------------------------------------------------------
        // Misc
        // ---------------------------------------------------------------
        "TopK" => Op::TopK {
            axis:    get_attr_int(node, "axis",   -1),
            largest: get_attr_int(node, "largest", 1) != 0,
            sorted:  get_attr_int(node, "sorted",  1) != 0,
        },
        "Unique" => Op::Unique {
            sorted: get_attr_int(node, "sorted", 1) != 0,
        },
        "Dropout" => Op::Dropout {
            training_mode: get_attr_int(node, "training_mode", 0) != 0,
        },
        "EyeLike" => {
            let to = get_attr_int(node, "dtype", -1);
            let dtype = if to >= 0 { Some(tensor_dtype_to_dtype(to as i32)?) } else { None };
            Op::EyeLike { dtype, k: get_attr_int(node, "k", 0) }
        }
        "OneHot" => Op::OneHot { axis: get_attr_int(node, "axis", -1) },
        "Bernoulli" => {
            let to = get_attr_int(node, "dtype", -1);
            let dtype = if to >= 0 { Some(tensor_dtype_to_dtype(to as i32)?) } else { None };
            Op::Bernoulli { dtype }
        }
        "RandomUniformLike" => {
            let to = get_attr_int(node, "dtype", -1);
            let dtype = if to >= 0 { Some(tensor_dtype_to_dtype(to as i32)?) } else { None };
            Op::RandomUniformLike {
                dtype,
                high: get_attr_float(node, "high", 1.0) as f64,
                low:  get_attr_float(node, "low",  0.0) as f64,
            }
        }

        // ---------------------------------------------------------------
        // Quantisation
        // ---------------------------------------------------------------
        "QuantizeLinear" => Op::QuantizeLinear {
            axis:     get_attr_int(node, "axis", 1),
            saturate: get_attr_int(node, "saturate", 1) != 0,
        },
        "DequantizeLinear" => Op::DequantizeLinear {
            axis: get_attr_int(node, "axis", 1),
        },
        "DynamicQuantizeLinear" => Op::DynamicQuantizeLinear,

        // ---------------------------------------------------------------
        // Signal
        // ---------------------------------------------------------------
        "DFT"  => Op::Dft {
            inverse:  get_attr_int(node, "inverse",  0) != 0,
            onesided: get_attr_int(node, "onesided", 0) != 0,
        },
        "STFT"             => Op::Stft,
        "MelWeightMatrix"  => Op::MelWeightMatrix,
        "HannWindow"       => Op::HannWindow    { periodic: get_attr_int(node, "periodic", 1) != 0 },
        "BlackmanWindow"   => Op::BlackmanWindow{ periodic: get_attr_int(node, "periodic", 1) != 0 },
        "HammingWindow"    => Op::HammingWindow { periodic: get_attr_int(node, "periodic", 1) != 0 },

        // ---------------------------------------------------------------
        // Loss
        // ---------------------------------------------------------------
        "NegativeLogLikelihoodLoss" => Op::NegativeLogLikelihoodLoss {
            reduction: get_attr_string(node, "reduction", "mean").to_string(),
        },
        "SoftmaxCrossEntropyLoss" => Op::SoftmaxCrossEntropyLoss {
            reduction: get_attr_string(node, "reduction", "mean").to_string(),
        },

        // ---------------------------------------------------------------
        // Sequences / optionals
        // ---------------------------------------------------------------
        "SequenceAt"       => Op::SequenceAt,
        "SequenceConstruct"=> Op::SequenceConstruct,
        "SequenceEmpty"    => Op::SequenceEmpty,
        "SequenceErase"    => Op::SequenceErase,
        "SequenceInsert"   => Op::SequenceInsert,
        "SequenceLength"   => Op::SequenceLength,
        "SequenceMap"      => Op::SequenceMap,
        "SplitToSequence"  => Op::SplitToSequence {
            axis:     get_attr_int(node, "axis",     0),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
        },
        "ConcatFromSequence" => Op::ConcatFromSequence {
            axis:     get_attr_int(node, "axis",     0),
            new_axis: get_attr_int(node, "new_axis", 0) != 0,
        },
        "OptionalGetElement"  => Op::OptionalGetElement,
        "OptionalHasElement"  => Op::OptionalHasElement,

        // ---------------------------------------------------------------
        // Control flow
        // ---------------------------------------------------------------
        "Loop" => Op::Loop,
        "Scan" => Op::Scan { num_scan_inputs: get_attr_int(node, "num_scan_inputs", 0) },
        "If"   => Op::If,

        // ---------------------------------------------------------------
        // Optimiser ops
        // ---------------------------------------------------------------
        "Adagrad"  => Op::Adagrad,
        "Adam"     => Op::Adam,
        "Momentum" => Op::Momentum,
        "Gradient" => Op::Gradient,

        // ---------------------------------------------------------------
        // String / NLP
        // ---------------------------------------------------------------
        "StringNormalizer" => Op::StringNormalizer,
        "RegexFullMatch"   => Op::RegexFullMatch {
            pattern: get_attr_string(node, "pattern", "").to_string(),
        },
        "StringConcat" => Op::StringConcat,
        "StringSplit"  => Op::StringSplit,
        "TfIdfVectorizer" => Op::TfIdfVectorizer,
        "LabelEncoder"    => Op::LabelEncoder,

        // ---------------------------------------------------------------
        // Other ML / deprecated
        // ---------------------------------------------------------------
        "ArrayFeatureExtractor" => Op::ArrayFeatureExtractor,
        "Binarizer" => Op::Binarizer {
            threshold: get_attr_float(node, "threshold", 0.0) as f64,
        },
        "TreeEnsemble"     => Op::TreeEnsemble,
        "ImageDecoder"     => Op::ImageDecoder,

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
