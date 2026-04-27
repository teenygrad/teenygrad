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

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use std::path::PathBuf;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};

const N: usize = 1024;
const BLOCK_SIZE: i32 = 1024;
const PTX_THREADS: u32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_l1_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::elementwise::L1LossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("l1_loss_forward_source", kernel.source());
    assert_debug_snapshot!("l1_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_l1_loss_backward_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::elementwise::L1LossBackward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("l1_loss_backward_source", kernel.source());
    assert_debug_snapshot!("l1_loss_backward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_mse_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::elementwise::MseLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("mse_loss_forward_source", kernel.source());
    assert_debug_snapshot!("mse_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_huber_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::elementwise::HuberLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("huber_loss_forward_source", kernel.source());
    assert_debug_snapshot!("huber_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_smooth_l1_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::elementwise::SmoothL1LossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("smooth_l1_loss_forward_source", kernel.source());
    assert_debug_snapshot!("smooth_l1_loss_forward_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_l1_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("loss_elementwise/x.bin");
    let y_host = load_fixture("loss_elementwise/y.bin");
    let expected = load_fixture("loss_elementwise/expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::L1LossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::L1LossForward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "l1_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_l1_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host = load_fixture("loss_elementwise/dy.bin");
    let x_host  = load_fixture("loss_elementwise/x.bin");
    let y_host  = load_fixture("loss_elementwise/y.bin");
    let expected = load_fixture("loss_elementwise/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf  = device.buffer::<f32>(N)?;
    let mut y_buf  = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::L1LossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::L1LossBackward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "l1_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_mse_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("loss_elementwise/mse_x.bin");
    let y_host = load_fixture("loss_elementwise/mse_y.bin");
    let expected = load_fixture("loss_elementwise/mse_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::MseLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::MseLossForward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "mse_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_mse_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host = load_fixture("loss_elementwise/mse_dy.bin");
    let x_host  = load_fixture("loss_elementwise/mse_x.bin");
    let y_host  = load_fixture("loss_elementwise/mse_y.bin");
    let expected = load_fixture("loss_elementwise/mse_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf  = device.buffer::<f32>(N)?;
    let mut y_buf  = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::MseLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::MseLossBackward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "mse_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_huber_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("loss_elementwise/huber_x.bin");
    let y_host = load_fixture("loss_elementwise/huber_y.bin");
    let expected = load_fixture("loss_elementwise/huber_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::HuberLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::HuberLossForward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    // delta = 1.0 matches the PyTorch fixture
    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "huber_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_huber_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host = load_fixture("loss_elementwise/huber_dy.bin");
    let x_host  = load_fixture("loss_elementwise/huber_x.bin");
    let y_host  = load_fixture("loss_elementwise/huber_y.bin");
    let expected = load_fixture("loss_elementwise/huber_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf  = device.buffer::<f32>(N)?;
    let mut y_buf  = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::HuberLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::HuberLossBackward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "huber_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_smooth_l1_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("loss_elementwise/sl1_x.bin");
    let y_host = load_fixture("loss_elementwise/sl1_y.bin");
    let expected = load_fixture("loss_elementwise/sl1_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::SmoothL1LossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::SmoothL1LossForward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    // beta = 1.0 matches the PyTorch fixture
    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "smooth_l1_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_smooth_l1_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host = load_fixture("loss_elementwise/sl1_dy.bin");
    let x_host  = load_fixture("loss_elementwise/sl1_x.bin");
    let y_host  = load_fixture("loss_elementwise/sl1_y.bin");
    let expected = load_fixture("loss_elementwise/sl1_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf  = device.buffer::<f32>(N)?;
    let mut y_buf  = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::elementwise::SmoothL1LossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::elementwise::SmoothL1LossBackward>(&ptx)?;

    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
        1.0_f32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "smooth_l1_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}
