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

// ASGD hyperparameters (must match generate.py: step=20, t0=10, d_ax=max(1,20-10)=10)
#[cfg(feature = "hardware")]
const LR: f32 = 0.01;
#[cfg(feature = "hardware")]
const WD: f32 = 1e-4;
#[cfg(feature = "hardware")]
const D_AX: f32 = 10.0;

// ── MLIR snapshot ─────────────────────────────────────────────────────────────

#[test]
fn test_asgd_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::asgd::AsgdStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("asgd_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("asgd_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: ASGD ────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_asgd_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_asgd/asgd_params_in.bin");
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_asgd/asgd_grad.bin");
    let ax_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_asgd/asgd_ax_in.bin");
    let params_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_asgd/asgd_params_out.bin");
    let ax_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_asgd/asgd_ax_out.bin");

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut ax_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    ax_buf.to_device(&ax_in)?;

    let kernel = teeny_kernels::nn::optim::asgd::AsgdStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::asgd::AsgdStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            ax_buf.as_device_ptr(),
            N as i32,
            LR,
            WD,
            D_AX,
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut ax_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    ax_buf.to_host(&mut ax_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "asgd params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (ax_out[i] - ax_ex[i]).abs() < 1e-4,
            "asgd ax at {i}: got={} expected={}",
            ax_out[i],
            ax_ex[i]
        );
    }
    Ok(())
}
