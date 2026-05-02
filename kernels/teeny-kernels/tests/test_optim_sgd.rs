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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};
#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_sgd_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::sgd::SgdStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("sgd_step_source", kernel.source());
    assert_debug_snapshot!("sgd_step_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_sgd_momentum_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::sgd::SgdMomentumStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("sgd_momentum_step_source", kernel.source());
    assert_debug_snapshot!("sgd_momentum_step_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_sgd_nesterov_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::sgd::SgdNesterovStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("sgd_nesterov_step_source", kernel.source());
    assert_debug_snapshot!("sgd_nesterov_step_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: SGD (no momentum) ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_sgd_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in = load_fixture("optim_sgd/sgd_params_in.bin");
    let grad      = load_fixture("optim_sgd/sgd_grad.bin");
    let params_ex = load_fixture("optim_sgd/sgd_params_out.bin");

    let mut params_buf = env.device.buffer::<f32>(N)?;
    let mut grad_buf   = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;

    let kernel = teeny_kernels::nn::optim::sgd::SgdStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::sgd::SgdStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        N as i32,
        0.01_f32,   // lr
        1e-4_f32,   // weight_decay
    ))?;

    let mut params_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "sgd_step params at {i}: got={} expected={}", params_out[i], params_ex[i]);
    }
    Ok(())
}

// ── CUDA: SGD with momentum ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_sgd_momentum_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in  = load_fixture("optim_sgd/sgd_mom_params_in.bin");
    let grad       = load_fixture("optim_sgd/sgd_mom_grad.bin");
    let buf_in     = load_fixture("optim_sgd/sgd_mom_buf_in.bin");
    let params_ex  = load_fixture("optim_sgd/sgd_mom_params_out.bin");
    let buf_ex     = load_fixture("optim_sgd/sgd_mom_buf_out.bin");

    let mut params_buf = env.device.buffer::<f32>(N)?;
    let mut grad_buf   = env.device.buffer::<f32>(N)?;
    let mut buf_buf    = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    buf_buf.to_device(&buf_in)?;

    let kernel = teeny_kernels::nn::optim::sgd::SgdMomentumStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::sgd::SgdMomentumStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        buf_buf.as_device_ptr() as *mut f32,
        N as i32,
        0.01_f32,   // lr
        0.9_f32,    // momentum
        0.0_f32,    // dampening
        1e-4_f32,   // weight_decay
    ))?;

    let mut params_out = vec![0.0f32; N];
    let mut buf_out    = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    buf_buf.to_host(&mut buf_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "sgd_momentum params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((buf_out[i] - buf_ex[i]).abs() < 1e-4,
            "sgd_momentum buf at {i}: got={} expected={}", buf_out[i], buf_ex[i]);
    }
    Ok(())
}

// ── CUDA: SGD Nesterov ────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_sgd_nesterov_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in  = load_fixture("optim_sgd/sgd_nes_params_in.bin");
    let grad       = load_fixture("optim_sgd/sgd_nes_grad.bin");
    let buf_in     = load_fixture("optim_sgd/sgd_nes_buf_in.bin");
    let params_ex  = load_fixture("optim_sgd/sgd_nes_params_out.bin");
    let buf_ex     = load_fixture("optim_sgd/sgd_nes_buf_out.bin");

    let mut params_buf = env.device.buffer::<f32>(N)?;
    let mut grad_buf   = env.device.buffer::<f32>(N)?;
    let mut buf_buf    = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    buf_buf.to_device(&buf_in)?;

    let kernel = teeny_kernels::nn::optim::sgd::SgdNesterovStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::sgd::SgdNesterovStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        buf_buf.as_device_ptr() as *mut f32,
        N as i32,
        0.01_f32,   // lr
        0.9_f32,    // momentum
        0.0_f32,    // dampening
        1e-4_f32,   // weight_decay
    ))?;

    let mut params_out = vec![0.0f32; N];
    let mut buf_out    = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    buf_buf.to_host(&mut buf_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "sgd_nesterov params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((buf_out[i] - buf_ex[i]).abs() < 1e-4,
            "sgd_nesterov buf at {i}: got={} expected={}", buf_out[i], buf_ex[i]);
    }
    Ok(())
}
