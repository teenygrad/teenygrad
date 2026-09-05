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
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const N: usize = 1024;
const BLOCK_SIZE: i32 = 1024;
#[cfg(feature = "hardware")]
const PTX_THREADS: u32 = 128;

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_l1_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::elementwise::L1LossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("l1_loss_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("l1_loss_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_l1_loss_backward_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::elementwise::L1LossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("l1_loss_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("l1_loss_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_mse_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::elementwise::MseLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("mse_loss_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("mse_loss_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_huber_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::elementwise::HuberLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("huber_loss_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("huber_loss_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_smooth_l1_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::elementwise::SmoothL1LossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "smooth_l1_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "smooth_l1_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_l1_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::L1LossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<teeny_kernels::nn::loss::elementwise::L1LossForward>(
        &ptx_path,
    )?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    let args = (
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "l1_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_l1_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/dy.bin");
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::L1LossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::L1LossBackward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "l1_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_mse_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/mse_x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/mse_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/mse_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::MseLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::MseLossForward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    let args = (
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "mse_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_mse_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/mse_dy.bin");
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/mse_x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/mse_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/mse_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::MseLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::MseLossBackward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "mse_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_huber_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/huber_x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/huber_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/huber_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::HuberLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::HuberLossForward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    // delta = 1.0 matches the PyTorch fixture
    let args = (
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "huber_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_huber_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/huber_dy.bin");
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/huber_x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/huber_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/huber_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::HuberLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::HuberLossBackward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "huber_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_smooth_l1_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/sl1_x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/sl1_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/sl1_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::SmoothL1LossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::SmoothL1LossForward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    // beta = 1.0 matches the PyTorch fixture
    let args = (
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "smooth_l1_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_smooth_l1_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/sl1_dy.bin");
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/sl1_x.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_elementwise/sl1_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_elementwise/sl1_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::SmoothL1LossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::elementwise::SmoothL1LossBackward,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "smooth_l1_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
