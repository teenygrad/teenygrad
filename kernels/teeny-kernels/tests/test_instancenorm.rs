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
const N: usize = 4; // batch size
#[cfg(feature = "hardware")]
const C: usize = 8; // channels
#[cfg(feature = "hardware")]
const L: usize = 16; // spatial length
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-5;
const BLOCK_L: i32 = 256;
#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// Source snapshot tests (no CUDA required)
// ---------------------------------------------------------------------------

#[test]
fn test_instance_norm_inference_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel =
        teeny_kernels::nn::norm::instancenorm::InstanceNormForwardInference::<f32>::new(BLOCK_L);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "instance_norm_inference_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_instance_norm_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::instancenorm::InstanceNormForward::<f32>::new(BLOCK_L);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "instance_norm_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_instance_norm_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::instancenorm::InstanceNormBackward::<f32>::new(BLOCK_L);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "instance_norm_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// MLIR snapshot tests (compile to MLIR, no GPU required)
// ---------------------------------------------------------------------------

#[test]
fn test_instance_norm_inference_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel =
        teeny_kernels::nn::norm::instancenorm::InstanceNormForwardInference::<f32>::new(BLOCK_L);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "instance_norm_inference_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests (requires GPU + fixtures from generate.py)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "hardware")]
fn test_instance_norm_inference_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "instancenorm/x.bin");
    let weight_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "instancenorm/weight.bin");
    let bias_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "instancenorm/bias.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "instancenorm/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; N * C * L];

    let mut x_buf = device.buffer::<f32>(N * C * L)?;
    let mut w_buf = device.buffer::<f32>(C)?;
    let mut b_buf = device.buffer::<f32>(C)?;
    let y_buf = device.buffer::<f32>(N * C * L)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&weight_host)?;
    b_buf.to_device(&bias_host)?;

    let kernel =
        teeny_kernels::nn::norm::instancenorm::InstanceNormForwardInference::<f32>::new(BLOCK_L);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::instancenorm::InstanceNormForwardInference<f32>,
    >(&ptx_path)?;

    // Grid: [N * C] — one CTA per (sample, channel)
    let cfg = teeny_runtime::launch_config_custom(
        [(N * C) as u32, 1, 1],
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
            N as i32,
            C as i32,
            L as i32,
            EPS,
        ),
    )?;

    y_buf.to_host(&mut y_host)?;
    for i in 0..N * C * L {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "instance_norm_inference mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}
