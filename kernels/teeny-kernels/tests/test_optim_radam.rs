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

// RAdam hyperparameters (must match generate.py)
#[cfg(feature = "hardware")]
const LR: f32 = 0.001;
#[cfg(feature = "hardware")]
const BETA1: f32 = 0.9;
#[cfg(feature = "hardware")]
const BETA2: f32 = 0.999;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-8;
#[cfg(feature = "hardware")]
const WD: f32 = 1e-4;

#[cfg(feature = "hardware")]
fn radam_adaptive_scalars(step: i32) -> (f32, f32) {
    let rho_inf: f32 = 2.0 / (1.0 - BETA2) - 1.0;
    let bc1 = 1.0 - BETA1.powi(step);
    let bc2 = 1.0 - BETA2.powi(step);
    let bc2_sqrt = bc2.sqrt();
    let rho_t = rho_inf - 2.0 * step as f32 * BETA2.powi(step) / bc2;
    let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
        .sqrt();
    let step_size = LR * rect / bc1;
    (step_size, bc2_sqrt)
}

#[cfg(feature = "hardware")]
fn radam_sgd_scalars(step: i32) -> f32 {
    let bc1 = 1.0 - BETA1.powi(step);
    LR / bc1
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_radam_adaptive_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::radam::RadamAdaptiveStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("radam_adaptive_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("radam_adaptive_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_radam_sgd_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::radam::RadamSgdStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("radam_sgd_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("radam_sgd_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: RAdam adaptive (step=100, rho_t > 5) ────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_radam_adaptive_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let (step_size, bc2_sqrt) = radam_adaptive_scalars(100);

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_params_in.bin",
    );
    let grad = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_grad.bin",
    );
    let exp_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_exp_avg_in.bin",
    );
    let exp_avg_sq_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_exp_avg_sq_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_params_out.bin",
    );
    let exp_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_exp_avg_out.bin",
    );
    let exp_avg_sq_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_adap_exp_avg_sq_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::radam::RadamAdaptiveStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<teeny_kernels::nn::optim::radam::RadamAdaptiveStep>(
        &ptx_path,
    )?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            exp_avg_buf.as_device_ptr(),
            exp_avg_sq_buf.as_device_ptr(),
            N as i32,
            step_size,
            bc2_sqrt,
            BETA1,
            BETA2,
            EPS,
            WD,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut exp_avg_out = vec![0.0f32; N];
    let mut exp_avg_sq_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    exp_avg_buf.to_host(&mut exp_avg_out)?;
    exp_avg_sq_buf.to_host(&mut exp_avg_sq_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "radam_adaptive params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "radam_adaptive exp_avg at {i}: got={} expected={}",
            exp_avg_out[i],
            exp_avg_ex[i]
        );
        assert!(
            (exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "radam_adaptive exp_avg_sq at {i}: got={} expected={}",
            exp_avg_sq_out[i],
            exp_avg_sq_ex[i]
        );
    }
    Ok(())
}

// ── CUDA: RAdam SGD fallback (step=1, rho_t <= 5) ─────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_radam_sgd_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let step_size = radam_sgd_scalars(1);

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_sgd_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_radam/radam_sgd_grad.bin");
    let exp_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_sgd_exp_avg_in.bin",
    );
    let exp_avg_sq_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_sgd_exp_avg_sq_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_sgd_params_out.bin",
    );
    let exp_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_sgd_exp_avg_out.bin",
    );
    let exp_avg_sq_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_radam/radam_sgd_exp_avg_sq_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::radam::RadamSgdStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::radam::RadamSgdStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            exp_avg_buf.as_device_ptr(),
            exp_avg_sq_buf.as_device_ptr(),
            N as i32,
            step_size,
            BETA1,
            BETA2,
            WD,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut exp_avg_out = vec![0.0f32; N];
    let mut exp_avg_sq_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    exp_avg_buf.to_host(&mut exp_avg_out)?;
    exp_avg_sq_buf.to_host(&mut exp_avg_sq_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "radam_sgd params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "radam_sgd exp_avg at {i}: got={} expected={}",
            exp_avg_out[i],
            exp_avg_ex[i]
        );
        assert!(
            (exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "radam_sgd exp_avg_sq at {i}: got={} expected={}",
            exp_avg_sq_out[i],
            exp_avg_sq_ex[i]
        );
    }
    Ok(())
}
