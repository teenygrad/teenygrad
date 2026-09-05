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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_sigmoid_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::activation::sigmoid::SigmoidForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("sigmoid_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("sigmoid_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_silu_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::activation::sigmoid::SiluForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("silu_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("silu_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_logsigmoid_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::activation::sigmoid::LogsigmoidForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("logsigmoid_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("logsigmoid_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA tests ────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_sigmoid_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "sigmoid/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "sigmoid/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::sigmoid::SigmoidForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::sigmoid::SigmoidForward<f32>,
    >(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);
    device.launch(
        &program,
        &cfg,
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "sigmoid_forward mismatch at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_sigmoid_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "sigmoid/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "sigmoid/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "sigmoid/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    // Compute y = sigmoid(x) on host for backward input
    let y_host: Vec<f32> = x_host.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::activation::sigmoid::SigmoidBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::sigmoid::SigmoidBackward<f32>,
    >(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "sigmoid_backward mismatch at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_silu_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "silu/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "silu/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::sigmoid::SiluForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::sigmoid::SiluForward<f32>,
    >(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);
    device.launch(
        &program,
        &cfg,
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "silu_forward mismatch at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_silu_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "silu/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "silu/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "silu/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::sigmoid::SiluBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::sigmoid::SiluBackward<f32>,
    >(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "silu_backward mismatch at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_logsigmoid_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "logsigmoid/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "logsigmoid/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::sigmoid::LogsigmoidForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::sigmoid::LogsigmoidForward<f32>,
    >(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);
    device.launch(
        &program,
        &cfg,
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "logsigmoid_forward mismatch at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_logsigmoid_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "logsigmoid/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "logsigmoid/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "logsigmoid/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::sigmoid::LogsigmoidBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::sigmoid::LogsigmoidBackward<f32>,
    >(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "logsigmoid_backward mismatch at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
