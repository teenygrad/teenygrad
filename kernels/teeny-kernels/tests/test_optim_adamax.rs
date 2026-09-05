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

// Adamax hyperparameters (must match generate.py)
#[cfg(feature = "hardware")]
const LR: f32 = 0.002;
#[cfg(feature = "hardware")]
const BETA1: f32 = 0.9;
#[cfg(feature = "hardware")]
const BETA2: f32 = 0.999;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-8;
#[cfg(feature = "hardware")]
const WD: f32 = 1e-4;
#[cfg(feature = "hardware")]
const STEP: i32 = 5;

// ── MLIR snapshot ─────────────────────────────────────────────────────────────

#[test]
fn test_adamax_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::adamax::AdamaxStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("adamax_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("adamax_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: Adamax ──────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_adamax_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let bias_c1 = 1.0_f32 - BETA1.powi(STEP);
    let clr = LR / bias_c1;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adamax/adamax_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adamax/adamax_grad.bin");
    let exp_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adamax/adamax_exp_avg_in.bin",
    );
    let exp_inf_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adamax/adamax_exp_inf_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adamax/adamax_params_out.bin",
    );
    let exp_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adamax/adamax_exp_avg_out.bin",
    );
    let exp_inf_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adamax/adamax_exp_inf_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut exp_avg_buf = device.buffer::<f32>(N)?;
    let mut exp_inf_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    exp_avg_buf.to_device(&exp_avg_in)?;
    exp_inf_buf.to_device(&exp_inf_in)?;

    let kernel = teeny_kernels::nn::optim::adamax::AdamaxStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::adamax::AdamaxStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            exp_avg_buf.as_device_ptr(),
            exp_inf_buf.as_device_ptr(),
            N as i32,
            clr,
            BETA1,
            BETA2,
            EPS,
            WD,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut exp_avg_out = vec![0.0f32; N];
    let mut exp_inf_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    exp_avg_buf.to_host(&mut exp_avg_out)?;
    exp_inf_buf.to_host(&mut exp_inf_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "adamax params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (exp_avg_out[i] - exp_avg_ex[i]).abs() < 1e-5,
            "adamax exp_avg at {i}: got={} expected={}",
            exp_avg_out[i],
            exp_avg_ex[i]
        );
        assert!(
            (exp_inf_out[i] - exp_inf_ex[i]).abs() < 1e-5,
            "adamax exp_inf at {i}: got={} expected={}",
            exp_inf_out[i],
            exp_inf_ex[i]
        );
    }
    Ok(())
}
