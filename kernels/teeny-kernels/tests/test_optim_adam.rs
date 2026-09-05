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

// Adam hyperparameters (must match generate.py)
#[cfg(feature = "hardware")]
const BETA1: f32 = 0.9;
#[cfg(feature = "hardware")]
const BETA2: f32 = 0.999;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-8;
#[cfg(feature = "hardware")]
const WD: f32 = 1e-4;
#[cfg(feature = "hardware")]
const LR: f32 = 0.001;
#[cfg(feature = "hardware")]
const STEP: i32 = 5;

#[cfg(feature = "hardware")]
fn adam_scalars() -> (f32, f32) {
    let bias_c1 = 1.0 - BETA1.powi(STEP);
    let bias_c2 = 1.0 - BETA2.powi(STEP);
    let step_size = LR / bias_c1;
    let bc2_sqrt = bias_c2.sqrt();
    (step_size, bc2_sqrt)
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_adam_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::adam::AdamStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("adam_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("adam_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_adamw_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::adam::AdamwStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("adamw_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("adamw_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: Adam ────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_adam_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let (step_size, bc2_sqrt) = adam_scalars();

    let params_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adam/adam_params_in.bin");
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adam/adam_grad.bin");
    let exp_avg_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adam/adam_exp_avg_in.bin");
    let exp_avg_sq_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adam_exp_avg_sq_in.bin",
    );
    let params_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adam/adam_params_out.bin");
    let exp_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adam_exp_avg_out.bin",
    );
    let exp_avg_sq_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adam_exp_avg_sq_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::adam::AdamStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::adam::AdamStep>(&ptx_path)?;
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
            "adam params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "adam exp_avg at {i}: got={} expected={}",
            exp_avg_out[i],
            exp_avg_ex[i]
        );
        assert!(
            (exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "adam exp_avg_sq at {i}: got={} expected={}",
            exp_avg_sq_out[i],
            exp_avg_sq_ex[i]
        );
    }
    Ok(())
}

// ── CUDA: AdamW ───────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_adamw_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let (step_size, bc2_sqrt) = adam_scalars();

    let params_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adam/adamw_params_in.bin");
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adam/adamw_grad.bin");
    let exp_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adamw_exp_avg_in.bin",
    );
    let exp_avg_sq_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adamw_exp_avg_sq_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adamw_params_out.bin",
    );
    let exp_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adamw_exp_avg_out.bin",
    );
    let exp_avg_sq_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adam/adamw_exp_avg_sq_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_sq_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_avg_sq_buf.to_device(&exp_avg_sq_in)?;

    let kernel = teeny_kernels::nn::optim::adam::AdamwStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::adam::AdamwStep>(&ptx_path)?;
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
            LR,
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
            "adamw params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "adamw exp_avg at {i}: got={} expected={}",
            exp_avg_out[i],
            exp_avg_ex[i]
        );
        assert!(
            (exp_avg_sq_out[i] - exp_avg_sq_ex[i]).abs() < 1e-6,
            "adamw exp_avg_sq at {i}: got={} expected={}",
            exp_avg_sq_out[i],
            exp_avg_sq_ex[i]
        );
    }
    Ok(())
}
