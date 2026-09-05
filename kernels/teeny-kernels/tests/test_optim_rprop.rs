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

// Rprop hyperparameters (must match generate.py)
#[cfg(feature = "hardware")]
const ETA_PLUS: f32 = 1.2;
#[cfg(feature = "hardware")]
const ETA_MINUS: f32 = 0.5;
#[cfg(feature = "hardware")]
const STEP_MIN: f32 = 1e-6;
#[cfg(feature = "hardware")]
const STEP_MAX: f32 = 50.0;

// ── MLIR snapshot ─────────────────────────────────────────────────────────────

#[test]
fn test_rprop_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::rprop::RpropStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("rprop_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("rprop_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: Rprop ───────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_rprop_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rprop/rprop_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_rprop/rprop_grad.bin");
    let prev_grad_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rprop/rprop_prev_grad_in.bin",
    );
    let step_size_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rprop/rprop_step_size_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rprop/rprop_params_out.bin",
    );
    let step_size_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rprop/rprop_step_size_out.bin",
    );
    let prev_grad_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rprop/rprop_prev_grad_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut prev_grad_buf = device.buffer::<f32>(N)?;
    let mut step_size_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    prev_grad_buf.to_device(&prev_grad_in)?;
    step_size_buf.to_device(&step_size_in)?;

    let kernel = teeny_kernels::nn::optim::rprop::RpropStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::rprop::RpropStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            prev_grad_buf.as_device_ptr(),
            step_size_buf.as_device_ptr(),
            N as i32,
            ETA_PLUS,
            ETA_MINUS,
            STEP_MIN,
            STEP_MAX,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut step_size_out = vec![0.0f32; N];
    let mut prev_grad_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    step_size_buf.to_host(&mut step_size_out)?;
    prev_grad_buf.to_host(&mut prev_grad_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "rprop params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (step_size_out[i] - step_size_ex[i]).abs() < 1e-5,
            "rprop step_size at {i}: got={} expected={}",
            step_size_out[i],
            step_size_ex[i]
        );
        assert!(
            (prev_grad_out[i] - prev_grad_ex[i]).abs() < 1e-5,
            "rprop prev_grad at {i}: got={} expected={}",
            prev_grad_out[i],
            prev_grad_ex[i]
        );
    }
    Ok(())
}
