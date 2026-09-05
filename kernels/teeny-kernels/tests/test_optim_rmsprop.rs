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

// RMSprop hyperparameters (must match generate.py)
#[cfg(feature = "hardware")]
const LR: f32 = 0.01;
#[cfg(feature = "hardware")]
const ALPHA: f32 = 0.99;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-8;
#[cfg(feature = "hardware")]
const WD: f32 = 1e-4;
#[cfg(feature = "hardware")]
const MU: f32 = 0.9;

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_rmsprop_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("rmsprop_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("rmsprop_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_rmsprop_momentum_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropMomentumStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "rmsprop_momentum_step_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("rmsprop_momentum_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: RMSprop (no momentum) ───────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_rmsprop_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rms_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_rmsprop/rms_grad.bin");
    let sq_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rms_sq_avg_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rms_params_out.bin",
    );
    let sq_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rms_sq_avg_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut sq_avg_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sq_avg_buf.to_device(&sq_avg_in)?;

    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::rmsprop::RmspropStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            sq_avg_buf.as_device_ptr(),
            N as i32,
            LR,
            ALPHA,
            EPS,
            WD,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut sq_avg_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sq_avg_buf.to_host(&mut sq_avg_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "rmsprop params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (sq_avg_out[i] - sq_avg_ex[i]).abs() < 1e-5,
            "rmsprop sq_avg at {i}: got={} expected={}",
            sq_avg_out[i],
            sq_avg_ex[i]
        );
    }
    Ok(())
}

// ── CUDA: RMSprop with momentum ───────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_rmsprop_momentum_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rmsm_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_rmsprop/rmsm_grad.bin");
    let sq_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rmsm_sq_avg_in.bin",
    );
    let buf_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_rmsprop/rmsm_buf_in.bin");
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rmsm_params_out.bin",
    );
    let sq_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_rmsprop/rmsm_sq_avg_out.bin",
    );
    let buf_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_rmsprop/rmsm_buf_out.bin");

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut sq_avg_buf = device.buffer::<f32>(N)?;
    let mut buf_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sq_avg_buf.to_device(&sq_avg_in)?;
    buf_buf.to_device(&buf_in)?;

    let kernel = teeny_kernels::nn::optim::rmsprop::RmspropMomentumStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::optim::rmsprop::RmspropMomentumStep,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            sq_avg_buf.as_device_ptr(),
            buf_buf.as_device_ptr(),
            N as i32,
            LR,
            ALPHA,
            EPS,
            WD,
            MU,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut sq_avg_out = vec![0.0f32; N];
    let mut buf_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sq_avg_buf.to_host(&mut sq_avg_out)?;
    buf_buf.to_host(&mut buf_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "rmsprop_mom params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (sq_avg_out[i] - sq_avg_ex[i]).abs() < 1e-5,
            "rmsprop_mom sq_avg at {i}: got={} expected={}",
            sq_avg_out[i],
            sq_avg_ex[i]
        );
        assert!(
            (buf_out[i] - buf_ex[i]).abs() < 1e-4,
            "rmsprop_mom buf at {i}: got={} expected={}",
            buf_out[i],
            buf_ex[i]
        );
    }
    Ok(())
}
