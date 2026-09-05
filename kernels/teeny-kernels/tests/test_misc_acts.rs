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
fn test_leaky_relu_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::activation::misc::LeakyReluForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("leaky_relu_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("leaky_relu_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_softsign_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::activation::misc::SoftsignForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("softsign_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("softsign_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_softplus_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::activation::misc::SoftplusForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("softplus_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("softplus_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: LeakyReLU ───────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_leaky_relu_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "leaky_relu/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "leaky_relu/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::LeakyReluForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::LeakyReluForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N as i32,
            0.01_f32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "leaky_relu_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_leaky_relu_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "leaky_relu/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "leaky_relu/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "leaky_relu/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::LeakyReluBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::LeakyReluBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
            0.01_f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "leaky_relu_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Threshold ───────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_threshold_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "threshold/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "threshold/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::ThresholdForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::ThresholdForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N as i32,
            0.5_f32,
            0.0_f32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "threshold_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_threshold_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "threshold/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "threshold/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "threshold/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::ThresholdBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::ThresholdBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
            0.5_f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "threshold_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Softsign ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_softsign_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softsign/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "softsign/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::SoftsignForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::SoftsignForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "softsign_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_softsign_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softsign/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softsign/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "softsign/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::SoftsignBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::SoftsignBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
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
            "softsign_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Softshrink ──────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_softshrink_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softshrink/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "softshrink/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::SoftshrinkForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::SoftshrinkForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N as i32,
            0.5_f32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "softshrink_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_softshrink_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softshrink/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softshrink/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "softshrink/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::SoftshrinkBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::SoftshrinkBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
            0.5_f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "softshrink_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Softplus ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_softplus_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softplus/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "softplus/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::SoftplusForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::SoftplusForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N as i32,
            1.0_f32,
            20.0_f32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "softplus_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_softplus_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softplus/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "softplus/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "softplus/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::misc::SoftplusBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::misc::SoftplusBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
            1.0_f32,
            20.0_f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "softplus_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
