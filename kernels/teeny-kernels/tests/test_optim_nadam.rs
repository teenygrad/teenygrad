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

// NAdam hyperparameters (must match generate.py)
const LR: f32             = 0.002;
const BETA1: f32          = 0.9;
const BETA2: f32          = 0.999;
const EPS: f32            = 1e-8;
const WD: f32             = 1e-4;
const STEP: usize         = 5;
const MOMENTUM_DECAY: f32 = 0.004;

fn nadam_scalars() -> (f32, f32, f32) {
    let mu_t  = BETA1 * (1.0 - 0.5 * 0.96_f32.powf(STEP as f32 * MOMENTUM_DECAY));
    let mu_t1 = BETA1 * (1.0 - 0.5 * 0.96_f32.powf((STEP + 1) as f32 * MOMENTUM_DECAY));
    let mu_product: f32 = (1..=STEP)
        .map(|i| BETA1 * (1.0 - 0.5 * 0.96_f32.powf(i as f32 * MOMENTUM_DECAY)))
        .product();
    let mu_product_next = mu_product * mu_t1;
    let bias_c2 = 1.0_f32 - BETA2.powi(STEP as i32);
    let bias_c2_sqrt = bias_c2.sqrt();
    let coeff_g = (1.0 - mu_t)  / (1.0 - mu_product);
    let coeff_m = mu_t1         / (1.0 - mu_product_next);
    (bias_c2_sqrt, coeff_g, coeff_m)
}

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── MLIR snapshot ─────────────────────────────────────────────────────────────

#[test]
fn test_nadam_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::nadam::NadamStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("nadam_step_source", kernel.source());
    assert_debug_snapshot!("nadam_step_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: NAdam ───────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_nadam_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let (bc2_sqrt, coeff_g, coeff_m) = nadam_scalars();

    let params_in      = load_fixture("optim_nadam/nadam_params_in.bin");
    let grad           = load_fixture("optim_nadam/nadam_grad.bin");
    let exp_avg_in     = load_fixture("optim_nadam/nadam_exp_avg_in.bin");
    let exp_avg_sq_in  = load_fixture("optim_nadam/nadam_exp_avg_sq_in.bin");
    let params_ex      = load_fixture("optim_nadam/nadam_params_out.bin");
    let exp_avg_ex     = load_fixture("optim_nadam/nadam_exp_avg_out.bin");
    let exp_avg_sq_ex  = load_fixture("optim_nadam/nadam_exp_avg_sq_out.bin");

    let mut params_buf     = env.device.buffer::<f32>(N)?;
    let mut grad_buf       = env.device.buffer::<f32>(N)?;
    let mut exp_avg_buf    = env.device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::nadam::NadamStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::nadam::NadamStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        exp_avg_buf.as_device_ptr() as *mut f32,
        exp_avg_sq_buf.as_device_ptr() as *mut f32,
        N as i32,
        LR, BETA1, BETA2, EPS, WD, bc2_sqrt, coeff_g, coeff_m,
    ))?;

    let mut params_out     = vec![0.0f32; N];
    let mut exp_avg_out    = vec![0.0f32; N];
    let mut exp_avg_sq_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    exp_avg_buf.to_host(&mut exp_avg_out)?;
    exp_avg_sq_buf.to_host(&mut exp_avg_sq_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "nadam params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "nadam exp_avg at {i}: got={} expected={}", exp_avg_out[i], exp_avg_ex[i]);
        assert!((exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "nadam exp_avg_sq at {i}: got={} expected={}", exp_avg_sq_out[i], exp_avg_sq_ex[i]);
    }
    Ok(())
}
