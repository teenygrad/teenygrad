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

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_adagrad_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::adagrad::AdagradStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("adagrad_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("adagrad_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_adadelta_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::adagrad::AdadeltaStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("adadelta_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("adadelta_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: Adagrad ─────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_adagrad_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adagrad_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_adagrad/adagrad_grad.bin");
    let sum_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adagrad_sum_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adagrad_params_out.bin",
    );
    let sum_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adagrad_sum_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut sum_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sum_buf.to_device(&sum_in)?;

    let kernel = teeny_kernels::nn::optim::adagrad::AdagradStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::adagrad::AdagradStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            sum_buf.as_device_ptr(),
            N as i32,
            0.01_f32,  // lr
            1e-10_f32, // eps
            1e-4_f32,  // weight_decay
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut sum_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sum_buf.to_host(&mut sum_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "adagrad params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (sum_out[i] - sum_ex[i]).abs() < 1e-5,
            "adagrad sum at {i}: got={} expected={}",
            sum_out[i],
            sum_ex[i]
        );
    }
    Ok(())
}

// ── CUDA: Adadelta ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_adadelta_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_params_in.bin",
    );
    let grad = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_grad.bin",
    );
    let sq_avg_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_sq_avg_in.bin",
    );
    let acc_delta_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_acc_delta_in.bin",
    );
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_params_out.bin",
    );
    let sq_avg_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_sq_avg_out.bin",
    );
    let acc_delta_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_adagrad/adadelta_acc_delta_out.bin",
    );

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut sq_avg_buf = device.buffer::<f32>(N)?;
    let mut acc_delta_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    sq_avg_buf.to_device(&sq_avg_in)?;
    acc_delta_buf.to_device(&acc_delta_in)?;

    let kernel = teeny_kernels::nn::optim::adagrad::AdadeltaStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::adagrad::AdadeltaStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            sq_avg_buf.as_device_ptr(),
            acc_delta_buf.as_device_ptr(),
            N as i32,
            1.0_f32,  // lr
            0.9_f32,  // rho
            1e-6_f32, // eps
            1e-4_f32, // weight_decay
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut sq_avg_out = vec![0.0f32; N];
    let mut acc_delta_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    sq_avg_buf.to_host(&mut sq_avg_out)?;
    acc_delta_buf.to_host(&mut acc_delta_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "adadelta params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (sq_avg_out[i] - sq_avg_ex[i]).abs() < 1e-5,
            "adadelta sq_avg at {i}: got={} expected={}",
            sq_avg_out[i],
            sq_avg_ex[i]
        );
        assert!(
            (acc_delta_out[i] - acc_delta_ex[i]).abs() < 1e-5,
            "adadelta acc_delta at {i}: got={} expected={}",
            acc_delta_out[i],
            acc_delta_ex[i]
        );
    }
    Ok(())
}
