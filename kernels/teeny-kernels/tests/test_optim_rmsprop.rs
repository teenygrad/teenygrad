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

// RMSprop hyperparameters (must match generate.py)
const LR: f32    = 0.01;
const ALPHA: f32 = 0.99;
const EPS: f32   = 1e-8;
const WD: f32    = 1e-4;
const MU: f32    = 0.9;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_rmsprop_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("rmsprop_step_source", kernel.source());
    assert_debug_snapshot!("rmsprop_step_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_rmsprop_momentum_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropMomentumStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("rmsprop_momentum_step_source", kernel.source());
    assert_debug_snapshot!("rmsprop_momentum_step_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: RMSprop (no momentum) ───────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_rmsprop_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in  = load_fixture("optim_rmsprop/rms_params_in.bin");
    let grad       = load_fixture("optim_rmsprop/rms_grad.bin");
    let sq_avg_in  = load_fixture("optim_rmsprop/rms_sq_avg_in.bin");
    let params_ex  = load_fixture("optim_rmsprop/rms_params_out.bin");
    let sq_avg_ex  = load_fixture("optim_rmsprop/rms_sq_avg_out.bin");

    let mut params_buf = env.device.buffer::<f32>(N)?;
    let mut grad_buf   = env.device.buffer::<f32>(N)?;
    let mut sq_avg_buf = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sq_avg_buf.to_device(&sq_avg_in)?;

    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::rmsprop::RmspropStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        sq_avg_buf.as_device_ptr() as *mut f32,
        N as i32,
        LR, ALPHA, EPS, WD,
    ))?;

    let mut params_out = vec![0.0f32; N];
    let mut sq_avg_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sq_avg_buf.to_host(&mut sq_avg_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "rmsprop params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((sq_avg_out[i] - sq_avg_ex[i]).abs() < 1e-5,
            "rmsprop sq_avg at {i}: got={} expected={}", sq_avg_out[i], sq_avg_ex[i]);
    }
    Ok(())
}

// ── CUDA: RMSprop with momentum ───────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_rmsprop_momentum_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in  = load_fixture("optim_rmsprop/rmsm_params_in.bin");
    let grad       = load_fixture("optim_rmsprop/rmsm_grad.bin");
    let sq_avg_in  = load_fixture("optim_rmsprop/rmsm_sq_avg_in.bin");
    let buf_in     = load_fixture("optim_rmsprop/rmsm_buf_in.bin");
    let params_ex  = load_fixture("optim_rmsprop/rmsm_params_out.bin");
    let sq_avg_ex  = load_fixture("optim_rmsprop/rmsm_sq_avg_out.bin");
    let buf_ex     = load_fixture("optim_rmsprop/rmsm_buf_out.bin");

    let mut params_buf = env.device.buffer::<f32>(N)?;
    let mut grad_buf   = env.device.buffer::<f32>(N)?;
    let mut sq_avg_buf = env.device.buffer::<f32>(N)?;
    let mut buf_buf    = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sq_avg_buf.to_device(&sq_avg_in)?;
    buf_buf.to_device(&buf_in)?;

    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropMomentumStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::rmsprop::RmspropMomentumStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        sq_avg_buf.as_device_ptr() as *mut f32,
        buf_buf.as_device_ptr() as *mut f32,
        N as i32,
        LR, ALPHA, EPS, WD, MU,
    ))?;

    let mut params_out = vec![0.0f32; N];
    let mut sq_avg_out = vec![0.0f32; N];
    let mut buf_out    = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sq_avg_buf.to_host(&mut sq_avg_out)?;
    buf_buf.to_host(&mut buf_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "rmsprop_mom params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((sq_avg_out[i] - sq_avg_ex[i]).abs() < 1e-5,
            "rmsprop_mom sq_avg at {i}: got={} expected={}", sq_avg_out[i], sq_avg_ex[i]);
        assert!((buf_out[i] - buf_ex[i]).abs() < 1e-4,
            "rmsprop_mom buf at {i}: got={} expected={}", buf_out[i], buf_ex[i]);
    }
    Ok(())
}
