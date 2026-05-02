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

// RAdam hyperparameters (must match generate.py)
const LR: f32    = 0.001;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32   = 1e-8;
const WD: f32    = 1e-4;

fn radam_adaptive_scalars(step: i32) -> (f32, f32) {
    let rho_inf: f32 = 2.0 / (1.0 - BETA2) - 1.0;
    let bc1 = 1.0 - BETA1.powi(step);
    let bc2 = 1.0 - BETA2.powi(step);
    let bc2_sqrt = bc2.sqrt();
    let rho_t = rho_inf - 2.0 * step as f32 * BETA2.powi(step) / bc2;
    let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
    let step_size = LR * rect / bc1;
    (step_size, bc2_sqrt)
}

fn radam_sgd_scalars(step: i32) -> f32 {
    let bc1 = 1.0 - BETA1.powi(step);
    LR / bc1
}

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_radam_adaptive_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::radam::RadamAdaptiveStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("radam_adaptive_step_source", kernel.source());
    assert_debug_snapshot!("radam_adaptive_step_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_radam_sgd_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::radam::RadamSgdStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("radam_sgd_step_source", kernel.source());
    assert_debug_snapshot!("radam_sgd_step_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: RAdam adaptive (step=100, rho_t > 5) ────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_radam_adaptive_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let (step_size, bc2_sqrt) = radam_adaptive_scalars(100);

    let params_in      = load_fixture("optim_radam/radam_adap_params_in.bin");
    let grad           = load_fixture("optim_radam/radam_adap_grad.bin");
    let exp_avg_in     = load_fixture("optim_radam/radam_adap_exp_avg_in.bin");
    let exp_avg_sq_in  = load_fixture("optim_radam/radam_adap_exp_avg_sq_in.bin");
    let params_ex      = load_fixture("optim_radam/radam_adap_params_out.bin");
    let exp_avg_ex     = load_fixture("optim_radam/radam_adap_exp_avg_out.bin");
    let exp_avg_sq_ex  = load_fixture("optim_radam/radam_adap_exp_avg_sq_out.bin");

    let mut params_buf     = env.device.buffer::<f32>(N)?;
    let mut grad_buf       = env.device.buffer::<f32>(N)?;
    let mut exp_avg_buf    = env.device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::radam::RadamAdaptiveStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::radam::RadamAdaptiveStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        exp_avg_buf.as_device_ptr() as *mut f32,
        exp_avg_sq_buf.as_device_ptr() as *mut f32,
        N as i32,
        step_size, bc2_sqrt, BETA1, BETA2, EPS, WD,
    ))?;

    let mut params_out     = vec![0.0f32; N];
    let mut exp_avg_out    = vec![0.0f32; N];
    let mut exp_avg_sq_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    exp_avg_buf.to_host(&mut exp_avg_out)?;
    exp_avg_sq_buf.to_host(&mut exp_avg_sq_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "radam_adaptive params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "radam_adaptive exp_avg at {i}: got={} expected={}", exp_avg_out[i], exp_avg_ex[i]);
        assert!((exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "radam_adaptive exp_avg_sq at {i}: got={} expected={}", exp_avg_sq_out[i], exp_avg_sq_ex[i]);
    }
    Ok(())
}

// ── CUDA: RAdam SGD fallback (step=1, rho_t <= 5) ─────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_radam_sgd_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let step_size = radam_sgd_scalars(1);

    let params_in      = load_fixture("optim_radam/radam_sgd_params_in.bin");
    let grad           = load_fixture("optim_radam/radam_sgd_grad.bin");
    let exp_avg_in     = load_fixture("optim_radam/radam_sgd_exp_avg_in.bin");
    let exp_avg_sq_in  = load_fixture("optim_radam/radam_sgd_exp_avg_sq_in.bin");
    let params_ex      = load_fixture("optim_radam/radam_sgd_params_out.bin");
    let exp_avg_ex     = load_fixture("optim_radam/radam_sgd_exp_avg_out.bin");
    let exp_avg_sq_ex  = load_fixture("optim_radam/radam_sgd_exp_avg_sq_out.bin");

    let mut params_buf     = env.device.buffer::<f32>(N)?;
    let mut grad_buf       = env.device.buffer::<f32>(N)?;
    let mut exp_avg_buf    = env.device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::radam::RadamSgdStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::radam::RadamSgdStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(N, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        exp_avg_buf.as_device_ptr() as *mut f32,
        exp_avg_sq_buf.as_device_ptr() as *mut f32,
        N as i32,
        step_size, BETA1, BETA2, WD,
    ))?;

    let mut params_out     = vec![0.0f32; N];
    let mut exp_avg_out    = vec![0.0f32; N];
    let mut exp_avg_sq_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    exp_avg_buf.to_host(&mut exp_avg_out)?;
    exp_avg_sq_buf.to_host(&mut exp_avg_sq_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "radam_sgd params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "radam_sgd exp_avg at {i}: got={} expected={}", exp_avg_out[i], exp_avg_ex[i]);
        assert!((exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "radam_sgd exp_avg_sq at {i}: got={} expected={}", exp_avg_sq_out[i], exp_avg_sq_ex[i]);
    }
    Ok(())
}
