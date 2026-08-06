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

//! Output convention: [vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors)
//! layered on plain `.safetensors`, so quantized checkpoints stay loadable by existing HF/vLLM
//! tooling for INT8/FP8. For a quantized weight tensor named `foo.weight`:
//!
//! - `foo.weight` itself becomes the quantized values (`I8`, packed `U8` for INT4, or
//!   `F8_E4M3`/`F8_E5M2`).
//! - `foo.weight_scale` holds one `F32` scale per group (flattened to 1-D, length = number of
//!   groups -- see [`crate::quant::compute_groups`] for how elements map to groups).
//! - `foo.weight_zero_point` holds one `I32` zero-point per group, only for asymmetric schemes
//!   (symmetric schemes have an implicit zero-point of `0` and omit this tensor).
//! - A `quantization_config` JSON blob in the file's `__metadata__` header describes the scheme
//!   (`config_groups`), which tensors were left unquantized (`ignore`), and -- since this
//!   crate's INT4 packing (see [`crate::quant::pack4`]) doesn't match compressed-tensors' own
//!   int32-based `pack-quantized` layout -- a `teenygrad_packed_int4` extension recording each
//!   packed tensor's true logical shape.
//!
//! Tensors are only quantized if they're rank >= 2 (see [`should_quantize`]) -- 1-D tensors
//! (biases, norm weights) are passed through unchanged and listed in `ignore`, matching common
//! PTQ tooling's default of leaving those alone.

use std::collections::HashMap;

use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::quant::pack4::pack_i4;
use crate::quant::{Fp8Variant, Granularity, Scheme, dequantize_affine, dequantize_fp8};
use crate::quant::{quantize_affine, quantize_fp8};
use crate::read::is_quantizable_float;
use crate::write::OutputTensor;

/// The compressed-tensors `weights` block for one scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsConfig {
    /// Bits per quantized element.
    pub num_bits: u8,
    /// `"int"` or `"float"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Whether `zero_point` is implicitly `0` (no `_zero_point` tensor is written).
    pub symmetric: bool,
    /// `"tensor"`, `"channel"`, or `"group"` (see [`Granularity::strategy_name`]).
    pub strategy: String,
    /// The axis grouping runs along, present for `"channel"`/`"group"` strategies. Required to
    /// reconstruct the exact element->group mapping (see [`crate::quant::compute_groups`]) --
    /// `strategy` and `group_size` alone are ambiguous for tensors with rank > 2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axis: Option<usize>,
    /// Elements per group, only present when `strategy == "group"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_size: Option<usize>,
    /// `"e4m3"` or `"e5m2"`, present only for `type == "float"`. The two FP8 encodings are
    /// *not* bit-compatible with each other, so this is required to correctly decode -- without
    /// it, [`crate::validate`] would have to guess the variant used to write the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp8_variant: Option<String>,
}

/// One compressed-tensors config group: a scheme plus the module types it applies to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigGroup {
    /// The quantization scheme for this group.
    pub weights: WeightsConfig,
    /// Module-type targets this group applies to. `teeny-quant` operates on raw tensor names
    /// rather than a module graph, so this is always `["*"]` today -- a placeholder for when a
    /// real module-type mapping (e.g. via the ONNX graph) becomes available.
    pub targets: Vec<String>,
}

/// Logical shape of a nibble-packed INT4 tensor, since the packed `U8` tensor's own shape
/// (`[ceil(n / 2)]`) doesn't reflect it. See [`crate::quant::pack4`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedTensorInfo {
    /// The tensor's shape before packing.
    pub logical_shape: Vec<usize>,
    /// `logical_shape.iter().product()`, for convenience.
    pub elements: usize,
}

/// The full `quantization_config` metadata blob embedded in the output `.safetensors` header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Always `"compressed-tensors"`.
    pub quant_method: String,
    /// `"int-quantized"`, `"float-quantized"`, or `"pack-quantized"`.
    pub format: String,
    /// Named config groups (today, always a single `"group_0"`).
    pub config_groups: HashMap<String, ConfigGroup>,
    /// Tensor names left unquantized (see [`should_quantize`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
    /// `teeny-quant` extension: logical shape for each nibble-packed INT4 tensor.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub teenygrad_packed_int4: HashMap<String, PackedTensorInfo>,
}

/// The metadata key `quantization_config` is stored under in the `.safetensors` header.
pub const QUANTIZATION_CONFIG_KEY: &str = "quantization_config";

/// `foo.weight` -> `foo.weight_scale`.
pub fn scale_tensor_name(weight_name: &str) -> String {
    format!("{weight_name}_scale")
}

/// `foo.weight` -> `foo.weight_zero_point`.
pub fn zero_point_tensor_name(weight_name: &str) -> String {
    format!("{weight_name}_zero_point")
}

/// Whether a tensor should be quantized: rank >= 2 (so biases/norm weights are left alone) and a
/// dtype [`crate::read::read_f32`] can upcast from.
pub fn should_quantize(shape: &[usize], dtype: Dtype) -> bool {
    shape.len() >= 2 && is_quantizable_float(dtype)
}

fn f32_vec_to_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_vec_to_le_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Quantizes one tensor per `scheme`/`granularity`, returning the output tensors to insert
/// (quantized weight, `_scale`, and -- for asymmetric schemes -- `_zero_point`), keyed by their
/// final names. `name` is the *original* (unsuffixed) tensor name.
pub fn quantize_tensor(
    name: &str,
    data: &[f32],
    shape: &[usize],
    scheme: Scheme,
    granularity: Granularity,
) -> Result<HashMap<String, OutputTensor>> {
    let mut out = HashMap::with_capacity(3);

    match scheme {
        Scheme::Int8 { symmetric } => {
            let q = quantize_affine(name, data, shape, granularity, symmetric, 8)?;
            let qbytes: Vec<u8> = q.qvalues.iter().map(|&v| v as u8).collect();
            out.insert(
                name.to_string(),
                OutputTensor {
                    dtype: Dtype::I8,
                    shape: shape.to_vec(),
                    data: qbytes,
                },
            );
            let scales: Vec<f32> = q.params.iter().map(|p| p.scale).collect();
            out.insert(
                scale_tensor_name(name),
                OutputTensor {
                    dtype: Dtype::F32,
                    shape: vec![scales.len()],
                    data: f32_vec_to_le_bytes(&scales),
                },
            );
            if !symmetric {
                let zps: Vec<i32> = q.params.iter().map(|p| p.zero_point).collect();
                out.insert(
                    zero_point_tensor_name(name),
                    OutputTensor {
                        dtype: Dtype::I32,
                        shape: vec![zps.len()],
                        data: i32_vec_to_le_bytes(&zps),
                    },
                );
            }
        }
        Scheme::Int4 { symmetric } => {
            let q = quantize_affine(name, data, shape, granularity, symmetric, 4)?;
            let packed = pack_i4(&q.qvalues);
            out.insert(
                name.to_string(),
                OutputTensor {
                    dtype: Dtype::U8,
                    shape: vec![packed.len()],
                    data: packed,
                },
            );
            let scales: Vec<f32> = q.params.iter().map(|p| p.scale).collect();
            out.insert(
                scale_tensor_name(name),
                OutputTensor {
                    dtype: Dtype::F32,
                    shape: vec![scales.len()],
                    data: f32_vec_to_le_bytes(&scales),
                },
            );
            if !symmetric {
                let zps: Vec<i32> = q.params.iter().map(|p| p.zero_point).collect();
                out.insert(
                    zero_point_tensor_name(name),
                    OutputTensor {
                        dtype: Dtype::I32,
                        shape: vec![zps.len()],
                        data: i32_vec_to_le_bytes(&zps),
                    },
                );
            }
        }
        Scheme::Fp8 { variant } => {
            let q = quantize_fp8(data, shape, granularity, variant);
            let dtype = match variant {
                Fp8Variant::E4M3 => Dtype::F8_E4M3,
                Fp8Variant::E5M2 => Dtype::F8_E5M2,
            };
            out.insert(
                name.to_string(),
                OutputTensor {
                    dtype,
                    shape: shape.to_vec(),
                    data: q.qvalues.clone(),
                },
            );
            out.insert(
                scale_tensor_name(name),
                OutputTensor {
                    dtype: Dtype::F32,
                    shape: vec![q.scales.len()],
                    data: f32_vec_to_le_bytes(&q.scales),
                },
            );
        }
    }

    Ok(out)
}

/// Dequantizes a tensor previously written by [`quantize_tensor`], given its already-decoded
/// scale (and, for asymmetric schemes, zero-point) values. Used by [`crate::validate`].
pub fn dequantize_tensor(
    scheme: Scheme,
    qview: &TensorView<'_>,
    shape: &[usize],
    scales: &[f32],
    zero_points: Option<&[i32]>,
    groups: &[u32],
) -> Vec<f32> {
    match scheme {
        Scheme::Int8 { .. } | Scheme::Int4 { .. } => {
            let qvalues: Vec<i8> = match scheme {
                Scheme::Int4 { .. } => {
                    let n: usize = shape.iter().product();
                    crate::quant::pack4::unpack_i4(qview.data(), n)
                }
                _ => qview.data().iter().map(|&b| b as i8).collect(),
            };
            let params: Vec<crate::quant::AffineParams> = scales
                .iter()
                .enumerate()
                .map(|(i, &scale)| crate::quant::AffineParams {
                    scale,
                    zero_point: zero_points.map(|z| z[i]).unwrap_or(0),
                })
                .collect();
            dequantize_affine(&crate::quant::QuantizedAffine {
                qvalues,
                params,
                groups: groups.to_vec(),
            })
        }
        Scheme::Fp8 { variant } => dequantize_fp8(&crate::quant::QuantizedFp8 {
            qvalues: qview.data().to_vec(),
            scales: scales.to_vec(),
            groups: groups.to_vec(),
            variant,
        }),
    }
}

/// Assembles the full [`QuantizationConfig`] for a checkpoint quantized uniformly with
/// `scheme`/`granularity`.
pub fn build_config(
    scheme: Scheme,
    granularity: Granularity,
    ignored: Vec<String>,
    packed_int4: HashMap<String, PackedTensorInfo>,
) -> QuantizationConfig {
    let format = match scheme {
        Scheme::Int8 { .. } => "int-quantized",
        Scheme::Int4 { .. } => "pack-quantized",
        Scheme::Fp8 { .. } => "float-quantized",
    };

    let group_size = match granularity {
        Granularity::Group { group_size, .. } => Some(group_size),
        _ => None,
    };
    let axis = granularity.axis_and_group_size().map(|(axis, _)| axis);
    let fp8_variant = match scheme {
        Scheme::Fp8 {
            variant: Fp8Variant::E4M3,
        } => Some("e4m3".to_string()),
        Scheme::Fp8 {
            variant: Fp8Variant::E5M2,
        } => Some("e5m2".to_string()),
        _ => None,
    };

    let weights = WeightsConfig {
        num_bits: scheme.num_bits(),
        type_: scheme.type_name().to_string(),
        symmetric: scheme.is_symmetric(),
        strategy: granularity.strategy_name().to_string(),
        axis,
        group_size,
        fp8_variant,
    };

    let mut config_groups = HashMap::new();
    config_groups.insert(
        "group_0".to_string(),
        ConfigGroup {
            weights,
            targets: vec!["*".to_string()],
        },
    );

    QuantizationConfig {
        quant_method: "compressed-tensors".to_string(),
        format: format.to_string(),
        config_groups,
        ignore: ignored,
        teenygrad_packed_int4: packed_int4,
    }
}

/// Serializes `config` into the `.safetensors` string-metadata map under
/// [`QUANTIZATION_CONFIG_KEY`].
pub fn config_to_metadata(config: &QuantizationConfig) -> Result<HashMap<String, String>> {
    let mut metadata = HashMap::new();
    metadata.insert(
        QUANTIZATION_CONFIG_KEY.to_string(),
        serde_json::to_string(config)?,
    );
    Ok(metadata)
}

/// Parses a `quantization_config` blob previously produced by [`config_to_metadata`].
pub fn config_from_metadata(
    metadata: &HashMap<String, String>,
) -> Result<Option<QuantizationConfig>> {
    match metadata.get(QUANTIZATION_CONFIG_KEY) {
        Some(json) => Ok(Some(serde_json::from_str(json)?)),
        None => Ok(None),
    }
}
