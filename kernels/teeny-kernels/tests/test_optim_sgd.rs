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
fn test_sgd_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::sgd::SgdStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("sgd_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("sgd_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_sgd_momentum_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::sgd::SgdMomentumStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("sgd_momentum_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("sgd_momentum_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_sgd_nesterov_step_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::optim::sgd::SgdNesterovStep::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("sgd_nesterov_step_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("sgd_nesterov_step_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA: SGD (no momentum) ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_sgd_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_params_in.bin");
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_grad.bin");
    let params_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_params_out.bin");

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;

    let kernel = teeny_kernels::nn::optim::sgd::SgdStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<teeny_kernels::nn::optim::sgd::SgdStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            N as i32,
            0.01_f32, // lr
            1e-4_f32, // weight_decay
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "sgd_step params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
    }
    Ok(())
}

// ── CUDA: SGD with momentum ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_sgd_momentum_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_sgd/sgd_mom_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_mom_grad.bin");
    let buf_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_mom_buf_in.bin");
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_sgd/sgd_mom_params_out.bin",
    );
    let buf_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_mom_buf_out.bin");

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut buf_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    buf_buf.to_device(&buf_in)?;

    let kernel = teeny_kernels::nn::optim::sgd::SgdMomentumStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::sgd::SgdMomentumStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            buf_buf.as_device_ptr(),
            N as i32,
            0.01_f32, // lr
            0.9_f32,  // momentum
            0.0_f32,  // dampening
            1e-4_f32, // weight_decay
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut buf_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    buf_buf.to_host(&mut buf_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "sgd_momentum params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (buf_out[i] - buf_ex[i]).abs() < 1e-4,
            "sgd_momentum buf at {i}: got={} expected={}",
            buf_out[i],
            buf_ex[i]
        );
    }
    Ok(())
}

// ── CUDA: SGD Nesterov ────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_sgd_nesterov_step_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let params_in = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_sgd/sgd_nes_params_in.bin",
    );
    let grad = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_nes_grad.bin");
    let buf_in = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_nes_buf_in.bin");
    let params_ex = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "optim_sgd/sgd_nes_params_out.bin",
    );
    let buf_ex = load_fixture(env!("CARGO_MANIFEST_DIR"), "optim_sgd/sgd_nes_buf_out.bin");

    let mut params_buf = device.buffer::<f32>(N)?;
    let mut grad_buf = device.buffer::<f32>(N)?;
    let mut buf_buf = device.buffer::<f32>(N)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;
    buf_buf.to_device(&buf_in)?;

    let kernel = teeny_kernels::nn::optim::sgd::SgdNesterovStep::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::optim::sgd::SgdNesterovStep>(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            params_buf.as_device_ptr(),
            grad_buf.as_device_ptr(),
            buf_buf.as_device_ptr(),
            N as i32,
            0.01_f32, // lr
            0.9_f32,  // momentum
            0.0_f32,  // dampening
            1e-4_f32, // weight_decay
        ),
    )?;

    let mut params_out = vec![0.0f32; N];
    let mut buf_out = vec![0.0f32; N];
    params_buf.to_host(&mut params_out)?;
    buf_buf.to_host(&mut buf_out)?;
    for i in 0..N {
        assert!(
            (params_out[i] - params_ex[i]).abs() < 1e-4,
            "sgd_nesterov params at {i}: got={} expected={}",
            params_out[i],
            params_ex[i]
        );
        assert!(
            (buf_out[i] - buf_ex[i]).abs() < 1e-4,
            "sgd_nesterov buf at {i}: got={} expected={}",
            buf_out[i],
            buf_ex[i]
        );
    }
    Ok(())
}
