/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use std::path::PathBuf;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const M: usize = 16;
#[cfg(feature = "hardware")]
const N: usize = 128;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-5;
const BLOCK_N: i32 = 256;
#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// Source snapshot tests (no CUDA required)
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_inference_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForwardInference::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "layer_norm_inference_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_layer_norm_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!("layer_norm_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_layer_norm_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormBackward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!("layer_norm_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// ASM snapshot tests (compile to ASM, no GPU required)
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_inference_asm() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForwardInference::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let asm = teeny_test::read_compiled_asm(ptx_path);
    assert_debug_snapshot!(
        format!("layer_norm_inference_asm_{}", teeny_runtime::BACKEND_NAME),
        asm
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests (requires GPU + fixtures from generate.py)
// ---------------------------------------------------------------------------

/// `layer_norm_forward` had NO numerics test -- only a source snapshot, while
/// the one hardware test here exercises the separate inference kernel
/// (teenygrad-3rk6.2 stage 2).
///
/// That gap matters for this conversion more than any so far: the kernel now
/// runs two declared reduction passes where the second reads the first's
/// result, and writes three outputs. A source snapshot cannot tell whether the
/// mean reaches the variance pass, nor whether the saved statistics are right.
///
/// `y` is checked against the PyTorch fixture; `mean` and `rstd` against a
/// plain per-row loop, since no fixture saves them.
#[test]
#[cfg(feature = "hardware")]
fn test_layer_norm_forward_saves_statistics_and_matches_reference() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/x.bin");
    let weight_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/weight.bin");
    let bias_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/bias.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/expected_forward.bin");

    let mut y_host = vec![0.0f32; M * N];
    let mut mean_host = vec![0.0f32; M];
    let mut rstd_host = vec![0.0f32; M];

    let mut x_buf = device.buffer::<f32>(M * N)?;
    let mut w_buf = device.buffer::<f32>(N)?;
    let mut b_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(M * N)?;
    let mean_buf = device.buffer::<f32>(M)?;
    let rstd_buf = device.buffer::<f32>(M)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&weight_host)?;
    b_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::layernorm::LayerNormForward<f32>,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [M as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            b_buf.as_device_ptr(),
            mean_buf.as_device_ptr(),
            rstd_buf.as_device_ptr(),
            M as i32,
            N as i32,
            EPS,
        ),
    )?;

    y_buf.to_host(&mut y_host)?;
    mean_buf.to_host(&mut mean_host)?;
    rstd_buf.to_host(&mut rstd_host)?;

    for i in 0..M * N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "layer_norm_forward y mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    // The saved statistics, which only this kernel writes. If the mean failed
    // to reach the variance pass, `rstd` is what would show it.
    for row in 0..M {
        let r = &x_host[row * N..(row + 1) * N];
        let mean = r.iter().sum::<f32>() / N as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / N as f32;
        let rstd = 1.0f32 / (var + EPS).sqrt();
        assert!(
            (mean_host[row] - mean).abs() < 1e-4,
            "layer_norm_forward mean mismatch at row {row}: gpu={}, expected={mean}",
            mean_host[row]
        );
        assert!(
            (rstd_host[row] - rstd).abs() < 1e-3 * rstd.abs().max(1.0),
            "layer_norm_forward rstd mismatch at row {row}: gpu={}, expected={rstd}",
            rstd_host[row]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_layer_norm_inference() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/x.bin");
    let weight_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/weight.bin");
    let bias_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/bias.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "layernorm/expected_forward.bin");
    let mut y_host = vec![0.0f32; M * N];

    let mut x_buf = device.buffer::<f32>(M * N)?;
    let mut w_buf = device.buffer::<f32>(N)?;
    let mut b_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(M * N)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&weight_host)?;
    b_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForwardInference::<f32>::new(BLOCK_N);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::layernorm::LayerNormForwardInference<f32>,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [M as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            b_buf.as_device_ptr(),
            M as i32,
            N as i32,
            EPS,
        ),
    )?;

    y_buf.to_host(&mut y_host)?;
    for i in 0..M * N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "layer_norm_inference mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}
