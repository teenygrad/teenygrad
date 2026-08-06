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

//! Tensor-level (not full-model) quantization error metrics: for each quantized tensor,
//! dequantize and compare against the original `f32` weights. No forward pass is needed -- this
//! is a pure diff between the original and quantized-then-reconstructed safetensors files.

use std::path::Path;

use crate::error::{Error, Result};
use crate::format;
use crate::quant::Scheme;
use crate::quant::compute_groups;
use crate::quant::fp8::Fp8Variant;
use crate::quant::granularity::Granularity;
use crate::read::read_f32;

/// Per-tensor quantization error metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorReport {
    /// The (original, unsuffixed) tensor name.
    pub name: String,
    /// Largest absolute error between original and dequantized-reconstructed elements.
    pub max_abs_error: f32,
    /// Mean absolute error across all elements.
    pub mean_abs_error: f32,
    /// Signal-to-quantization-noise ratio in dB (higher is better; `+inf` for an exact match).
    pub sqnr_db: f32,
}

/// Compares `original` against `reconstructed` element-wise, computing [`TensorReport`] metrics.
/// Both slices must be the same length (the caller is responsible for shape bookkeeping).
pub fn compare_tensors(
    name: &str,
    original: &[f32],
    reconstructed: &[f32],
) -> Result<TensorReport> {
    if original.len() != reconstructed.len() {
        return Err(Error::ShapeMismatch {
            name: name.to_string(),
            a: vec![original.len()],
            b: vec![reconstructed.len()],
        });
    }
    if original.is_empty() {
        return Ok(TensorReport {
            name: name.to_string(),
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            sqnr_db: f32::INFINITY,
        });
    }

    let mut max_abs_error = 0f32;
    let mut sum_abs_error = 0f64;
    let mut signal_energy = 0f64;
    let mut noise_energy = 0f64;
    for (&o, &r) in original.iter().zip(reconstructed.iter()) {
        let err = (o - r).abs();
        max_abs_error = max_abs_error.max(err);
        sum_abs_error += err as f64;
        signal_energy += (o as f64) * (o as f64);
        noise_energy += (err as f64) * (err as f64);
    }

    let mean_abs_error = (sum_abs_error / original.len() as f64) as f32;
    let sqnr_db = if noise_energy == 0.0 {
        f32::INFINITY
    } else {
        (10.0 * (signal_energy / noise_energy).log10()) as f32
    };

    Ok(TensorReport {
        name: name.to_string(),
        max_abs_error,
        mean_abs_error,
        sqnr_db,
    })
}

fn granularity_from_config(weights: &format::WeightsConfig) -> Result<Granularity> {
    match (weights.strategy.as_str(), weights.axis, weights.group_size) {
        ("tensor", _, _) => Ok(Granularity::PerTensor),
        ("channel", Some(axis), _) => Ok(Granularity::PerChannel { axis }),
        ("group", Some(axis), Some(group_size)) => Ok(Granularity::Group { axis, group_size }),
        _ => Err(Error::TensorNotFound(format!(
            "quantization_config has strategy '{}' but is missing axis/group_size",
            weights.strategy
        ))),
    }
}

fn scheme_from_config(weights: &format::WeightsConfig) -> Option<Scheme> {
    match (weights.type_.as_str(), weights.num_bits) {
        ("int", 8) => Some(Scheme::Int8 {
            symmetric: weights.symmetric,
        }),
        ("int", 4) => Some(Scheme::Int4 {
            symmetric: weights.symmetric,
        }),
        ("float", 8) => {
            // The two FP8 encodings are not bit-compatible, so decoding with the wrong one
            // silently produces garbage rather than a compile/runtime error -- the variant used
            // must come from the file's own quantization_config, not be guessed.
            let variant = match weights.fp8_variant.as_deref() {
                Some("e4m3") => Fp8Variant::E4M3,
                Some("e5m2") => Fp8Variant::E5M2,
                _ => return None,
            };
            Some(Scheme::Fp8 { variant })
        }
        _ => None,
    }
}

/// Compares every quantized tensor in `quantized_path` (as recorded by its embedded
/// `quantization_config`) against the corresponding tensor in `original_path`, returning one
/// [`TensorReport`] per quantized tensor, in the checkpoint's tensor order.
pub fn validate_checkpoint(
    original_path: &Path,
    quantized_path: &Path,
) -> Result<Vec<TensorReport>> {
    let original_mapped = teeny_data::safetensors::SafeTensors::from_pretrained(original_path)?;
    let original_tensors = original_mapped.tensors()?;

    let quantized_mapped = teeny_data::safetensors::SafeTensors::from_pretrained(quantized_path)?;
    let quantized_tensors = quantized_mapped.tensors()?;

    let metadata = crate::read::read_metadata(quantized_path)?;
    let config = format::config_from_metadata(&metadata)?.ok_or_else(|| {
        Error::TensorNotFound(format!(
            "'{}' has no quantization_config metadata -- was it produced by `teeny-quant quantize`?",
            quantized_path.display()
        ))
    })?;

    let weights = &config
        .config_groups
        .get("group_0")
        .ok_or_else(|| {
            Error::TensorNotFound("quantization_config.config_groups.group_0".to_string())
        })?
        .weights;
    let scheme = scheme_from_config(weights).ok_or_else(|| {
        Error::TensorNotFound(format!(
            "unrecognized scheme in quantization_config: type={} num_bits={}",
            weights.type_, weights.num_bits
        ))
    })?;

    let mut reports = Vec::new();
    for name in quantized_tensors.names() {
        if config.ignore.iter().any(|i| i == name) {
            continue;
        }
        if name.ends_with("_scale") || name.ends_with("_zero_point") {
            continue;
        }

        let original_view = original_tensors
            .tensor(name)
            .map_err(|_| Error::TensorNotFound(name.to_string()))?;
        let original_data = read_f32(&original_view, name)?;
        let shape = original_view.shape().to_vec();

        let packed = config.teenygrad_packed_int4.get(name);
        let logical_shape = packed.map(|p| p.logical_shape.clone()).unwrap_or(shape);
        let granularity = granularity_from_config(weights)?;
        let (groups, _) = compute_groups(&logical_shape, granularity);

        let quantized_view = quantized_tensors
            .tensor(name)
            .map_err(|_| Error::TensorNotFound(name.to_string()))?;
        let scale_name = format::scale_tensor_name(name);
        let scale_view = quantized_tensors
            .tensor(&scale_name)
            .map_err(|_| Error::TensorNotFound(scale_name.clone()))?;
        let scales = read_f32(&scale_view, &scale_name)?;

        let zero_points = if weights.symmetric {
            None
        } else {
            let zp_name = format::zero_point_tensor_name(name);
            let zp_view = quantized_tensors
                .tensor(&zp_name)
                .map_err(|_| Error::TensorNotFound(zp_name.clone()))?;
            Some(
                zp_view
                    .data()
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().expect("chunks_exact(4)")))
                    .collect::<Vec<_>>(),
            )
        };

        let reconstructed = format::dequantize_tensor(
            scheme,
            &quantized_view,
            &logical_shape,
            &scales,
            zero_points.as_deref(),
            &groups,
        );

        reports.push(compare_tensors(name, &original_data, &reconstructed)?);
    }

    Ok(reports)
}
