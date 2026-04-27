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
fn test_adagrad_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::adagrad::AdagradStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("adagrad_step_source", kernel.source());
    assert_debug_snapshot!("adagrad_step_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_adadelta_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::adagrad::AdadeltaStep::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("adadelta_step_source", kernel.source());
    assert_debug_snapshot!("adadelta_step_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: Adagrad ─────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_adagrad_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in = load_fixture("optim_adagrad/adagrad_params_in.bin");
    let grad      = load_fixture("optim_adagrad/adagrad_grad.bin");
    let sum_in    = load_fixture("optim_adagrad/adagrad_sum_in.bin");
    let params_ex = load_fixture("optim_adagrad/adagrad_params_out.bin");
    let sum_ex    = load_fixture("optim_adagrad/adagrad_sum_out.bin");

    let mut params_buf = env.device.buffer::<f32>(N)?;
    let mut grad_buf   = env.device.buffer::<f32>(N)?;
    let mut sum_buf    = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sum_buf.to_device(&sum_in)?;

    let kernel = teeny_kernels::nn::optim::adagrad::AdagradStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::adagrad::AdagradStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        sum_buf.as_device_ptr() as *mut f32,
        N as i32,
        0.01_f32,    // lr
        1e-10_f32,   // eps
        1e-4_f32,    // weight_decay
    ))?;

    let mut params_out = vec![0.0f32; N];
    let mut sum_out    = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sum_buf.to_host(&mut sum_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "adagrad params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((sum_out[i] - sum_ex[i]).abs() < 1e-5,
            "adagrad sum at {i}: got={} expected={}", sum_out[i], sum_ex[i]);
    }
    Ok(())
}

// ── CUDA: Adadelta ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_adadelta_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    let params_in    = load_fixture("optim_adagrad/adadelta_params_in.bin");
    let grad         = load_fixture("optim_adagrad/adadelta_grad.bin");
    let sq_avg_in    = load_fixture("optim_adagrad/adadelta_sq_avg_in.bin");
    let acc_delta_in = load_fixture("optim_adagrad/adadelta_acc_delta_in.bin");
    let params_ex    = load_fixture("optim_adagrad/adadelta_params_out.bin");
    let sq_avg_ex    = load_fixture("optim_adagrad/adadelta_sq_avg_out.bin");
    let acc_delta_ex = load_fixture("optim_adagrad/adadelta_acc_delta_out.bin");

    let mut params_buf    = env.device.buffer::<f32>(N)?;
    let mut grad_buf      = env.device.buffer::<f32>(N)?;
    let mut sq_avg_buf    = env.device.buffer::<f32>(N)?;
    let mut acc_delta_buf = env.device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sq_avg_buf.to_device(&sq_avg_in)?;
    acc_delta_buf.to_device(&acc_delta_in)?;

    let kernel = teeny_kernels::nn::optim::adagrad::AdadeltaStep::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::adagrad::AdadeltaStep>(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        sq_avg_buf.as_device_ptr() as *mut f32,
        acc_delta_buf.as_device_ptr() as *mut f32,
        N as i32,
        1.0_f32,    // lr
        0.9_f32,    // rho
        1e-6_f32,   // eps
        1e-4_f32,   // weight_decay
    ))?;

    let mut params_out    = vec![0.0f32; N];
    let mut sq_avg_out    = vec![0.0f32; N];
    let mut acc_delta_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sq_avg_buf.to_host(&mut sq_avg_out)?;
    acc_delta_buf.to_host(&mut acc_delta_out)?;
    for i in 0..N {
        assert!((params_out[i] - params_ex[i]).abs() < 1e-4,
            "adadelta params at {i}: got={} expected={}", params_out[i], params_ex[i]);
        assert!((sq_avg_out[i] - sq_avg_ex[i]).abs() < 1e-5,
            "adadelta sq_avg at {i}: got={} expected={}", sq_avg_out[i], sq_avg_ex[i]);
        assert!((acc_delta_out[i] - acc_delta_ex[i]).abs() < 1e-5,
            "adadelta acc_delta at {i}: got={} expected={}", acc_delta_out[i], acc_delta_ex[i]);
    }
    Ok(())
}
