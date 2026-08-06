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

//! End-to-end: build a tiny in-memory `.safetensors` "checkpoint", quantize it (INT8/INT4/FP8),
//! and check the output round-trips through [`teeny_quant::validate`] with small error and the
//! expected `quantization_config` metadata -- all without touching the filesystem.

use std::collections::HashMap;

use safetensors::Dtype;
use safetensors::tensor::TensorView;
use teeny_quant::format::{self, PackedTensorInfo};
use teeny_quant::quant::{Fp8Variant, Granularity, Scheme};
use teeny_quant::read::read_f32;
use teeny_quant::write::{OutputTensor, serialize_safetensors};

/// Builds a tiny fake checkpoint: a 2-D `weight` (quantizable) and a 1-D `bias` (should be
/// passed through unquantized).
fn fake_checkpoint() -> Vec<u8> {
    let weight_shape = vec![4usize, 8];
    let weight_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.37).collect();
    let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bias_shape = vec![4usize];
    let bias_data = [0.1f32, -0.2, 0.3, -0.4];
    let bias_bytes: Vec<u8> = bias_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let weight_view = TensorView::new(Dtype::F32, weight_shape, &weight_bytes).unwrap();
    let bias_view = TensorView::new(Dtype::F32, bias_shape, &bias_bytes).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), weight_view);
    tensors.insert("bias".to_string(), bias_view);

    safetensors::serialize(tensors, None).unwrap()
}

fn quantize_bytes(input_bytes: &[u8], scheme: Scheme, granularity: Granularity) -> Vec<u8> {
    let tensors = safetensors::SafeTensors::deserialize(input_bytes).unwrap();

    let mut outputs: HashMap<String, OutputTensor> = HashMap::new();
    let mut ignored = Vec::new();
    let mut packed_int4 = HashMap::new();

    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape = view.shape().to_vec();

        if !format::should_quantize(&shape, view.dtype()) {
            ignored.push(name.to_string());
            outputs.insert(
                name.to_string(),
                OutputTensor {
                    dtype: view.dtype(),
                    shape,
                    data: view.data().to_vec(),
                },
            );
            continue;
        }

        let data = read_f32(&view, name).unwrap();
        let quantized = format::quantize_tensor(name, &data, &shape, scheme, granularity).unwrap();
        if matches!(scheme, Scheme::Int4 { .. }) {
            packed_int4.insert(
                name.to_string(),
                PackedTensorInfo {
                    logical_shape: shape.clone(),
                    elements: shape.iter().product(),
                },
            );
        }
        outputs.extend(quantized);
    }

    let config = format::build_config(scheme, granularity, ignored, packed_int4);
    let metadata = format::config_to_metadata(&config).unwrap();
    serialize_safetensors(&outputs, metadata).unwrap()
}

fn assert_round_trips(scheme: Scheme, granularity: Granularity, max_tol_fraction: f32) {
    let original_bytes = fake_checkpoint();
    let quantized_bytes = quantize_bytes(&original_bytes, scheme, granularity);

    let original = safetensors::SafeTensors::deserialize(&original_bytes).unwrap();
    let quantized = safetensors::SafeTensors::deserialize(&quantized_bytes).unwrap();

    // bias (1-D) must be passed through byte-for-byte unquantized.
    let orig_bias = original.tensor("bias").unwrap();
    let quant_bias = quantized.tensor("bias").unwrap();
    assert_eq!(orig_bias.dtype(), quant_bias.dtype());
    assert_eq!(orig_bias.data(), quant_bias.data());

    // weight (2-D) must be quantized and reconstruct within tolerance.
    let orig_weight_view = original.tensor("weight").unwrap();
    let orig_weight = read_f32(&orig_weight_view, "weight").unwrap();

    let quant_weight_view = quantized.tensor("weight").unwrap();
    let scale_view = quantized.tensor("weight_scale").unwrap();
    let scales = read_f32(&scale_view, "weight_scale").unwrap();
    let zero_points = if scheme.is_symmetric() {
        None
    } else {
        let zp_view = quantized.tensor("weight_zero_point").unwrap();
        Some(
            zp_view
                .data()
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
        )
    };

    let shape = vec![4usize, 8];
    let (groups, _) = teeny_quant::quant::compute_groups(&shape, granularity);
    let reconstructed = format::dequantize_tensor(
        scheme,
        &quant_weight_view,
        &shape,
        &scales,
        zero_points.as_deref(),
        &groups,
    );

    for (o, r) in orig_weight.iter().zip(reconstructed.iter()) {
        let tol = o.abs().max(1.0) * max_tol_fraction;
        assert!((o - r).abs() <= tol, "orig={o} reconstructed={r} tol={tol}");
    }
}

#[test]
fn int8_symmetric_per_channel_round_trips() {
    assert_round_trips(
        Scheme::Int8 { symmetric: true },
        Granularity::PerChannel { axis: 0 },
        0.05,
    );
}

#[test]
fn int8_asymmetric_per_tensor_round_trips() {
    assert_round_trips(
        Scheme::Int8 { symmetric: false },
        Granularity::PerTensor,
        0.05,
    );
}

#[test]
fn int4_group_round_trips_with_looser_tolerance() {
    assert_round_trips(
        Scheme::Int4 { symmetric: true },
        Granularity::Group {
            axis: 1,
            group_size: 4,
        },
        0.5,
    );
}

#[test]
fn fp8_e4m3_per_channel_round_trips() {
    assert_round_trips(
        Scheme::Fp8 {
            variant: Fp8Variant::E4M3,
        },
        Granularity::PerChannel { axis: 0 },
        0.3,
    );
}

#[test]
fn fp8_e5m2_per_tensor_round_trips() {
    assert_round_trips(
        Scheme::Fp8 {
            variant: Fp8Variant::E5M2,
        },
        Granularity::PerTensor,
        0.3,
    );
}

/// Regression test for a bug where `teeny_quant::validate::validate_checkpoint` (unlike
/// `assert_round_trips` above, which is handed the scheme directly) reconstructs the scheme
/// purely from the file's own `quantization_config` metadata -- and originally always assumed
/// `Fp8Variant::E4M3` there regardless of which variant was actually written, silently decoding
/// `E5M2` bytes as `E4M3` (~0dB SQNR, i.e. noise). This exercises that exact metadata ->
/// scheme -> dequantize path end to end, unlike `assert_round_trips`.
#[test]
fn validate_checkpoint_decodes_fp8_e5m2_correctly() {
    let dir = std::env::temp_dir().join(format!("teeny-quant-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let original_path = dir.join("original.safetensors");
    let quantized_path = dir.join("quantized_e5m2.safetensors");

    std::fs::write(&original_path, fake_checkpoint()).unwrap();
    let quantized_bytes = quantize_bytes(
        &std::fs::read(&original_path).unwrap(),
        Scheme::Fp8 {
            variant: Fp8Variant::E5M2,
        },
        Granularity::PerTensor,
    );
    std::fs::write(&quantized_path, quantized_bytes).unwrap();

    let reports =
        teeny_quant::validate::validate_checkpoint(&original_path, &quantized_path).unwrap();
    let weight_report = reports.iter().find(|r| r.name == "weight").unwrap();
    // A correctly-decoded E5M2 tensor should land well above 10dB SQNR for this data; the
    // pre-fix bug (decoding as E4M3) produced ~0dB.
    assert!(
        weight_report.sqnr_db > 10.0,
        "SQNR too low ({} dB) -- FP8 variant likely decoded incorrectly",
        weight_report.sqnr_db
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn quantization_config_metadata_round_trips() {
    let original_bytes = fake_checkpoint();
    let quantized_bytes = quantize_bytes(
        &original_bytes,
        Scheme::Int8 { symmetric: true },
        Granularity::PerChannel { axis: 0 },
    );

    let (_, metadata) = safetensors::SafeTensors::read_metadata(&quantized_bytes).unwrap();
    let map = metadata.metadata().clone().unwrap_or_default();
    let config = format::config_from_metadata(&map).unwrap().unwrap();

    assert_eq!(config.quant_method, "compressed-tensors");
    assert_eq!(config.format, "int-quantized");
    assert_eq!(config.ignore, vec!["bias".to_string()]);
    let weights = &config.config_groups.get("group_0").unwrap().weights;
    assert_eq!(weights.num_bits, 8);
    assert_eq!(weights.type_, "int");
    assert!(weights.symmetric);
    assert_eq!(weights.strategy, "channel");
    assert_eq!(weights.axis, Some(0));
}
