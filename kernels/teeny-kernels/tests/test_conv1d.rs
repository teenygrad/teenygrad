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

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use std::path::PathBuf;
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

// ── No-padding constants ─────────────────────────────────────────────────────
#[cfg(feature = "hardware")]
const B: usize = 1;
#[cfg(feature = "hardware")]
const C_IN: usize = 2;
#[cfg(feature = "hardware")]
const C_OUT: usize = 4;
#[cfg(feature = "hardware")]
const L: usize = 16;
const KL: i32 = 3;
const STRIDE: i32 = 1;
const PAD: i32 = 0;
#[cfg(feature = "hardware")]
const OL: usize = (L + 2 * PAD as usize - KL as usize) / STRIDE as usize + 1; // 14
const BLOCK_OL: i32 = 8;

// ── Same-padding constants (PAD=1, same spatial size) ────────────────────────
#[cfg(feature = "hardware")]
const PAD_P: i32 = 1;
#[cfg(feature = "hardware")]
const OL_P: usize = (L + 2 * PAD_P as usize - KL as usize) / STRIDE as usize + 1; // 16

#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_forward_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dForward::<f32>::new(KL, STRIDE, PAD, BLOCK_OL);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("conv1d_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("conv1d_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_conv1d_backward_dx_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDx::<f32>::new(KL, STRIDE, PAD, BLOCK_OL);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("conv1d_backward_dx_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("conv1d_backward_dx_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_conv1d_backward_dw_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDw::<f32>::new(KL, STRIDE, PAD, BLOCK_OL);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("conv1d_backward_dw_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("conv1d_backward_dw_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests — no padding (PAD=0)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "hardware")]
fn test_conv1d_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/x.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/w.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/expected_forward.bin");
    let mut y_host = vec![0.0f32; B * C_OUT * OL];

    let mut x_buf = device.buffer::<f32>(B * C_IN * L)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KL as usize)?;
    let y_buf = device.buffer::<f32>(B * C_OUT * OL)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dForward::<f32>::new(KL, STRIDE, PAD, BLOCK_OL);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[conv1d_forward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<teeny_kernels::nn::conv::conv1d::Conv1dForward<f32>>(
        &ptx_path,
    )?;

    let num_ol_tiles = OL.div_ceil(BLOCK_OL as usize);
    let grid_size = B * C_OUT * num_ol_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_size as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        x_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        L as i32,
        OL as i32,
    );

    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..(B * C_OUT * OL) {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "conv1d_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv1d_backward_dx_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/dy.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/w.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/expected_dx.bin");
    let mut dx_host = vec![0.0f32; B * C_IN * L];

    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OL)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KL as usize)?;
    let dx_buf = device.buffer::<f32>(B * C_IN * L)?;

    dy_buf.to_device(&dy_host)?;
    w_buf.to_device(&w_host)?;

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDx::<f32>::new(KL, STRIDE, PAD, BLOCK_OL);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDx<f32>,
    >(&ptx_path)?;

    let num_ol_tiles = OL.div_ceil(BLOCK_OL as usize);
    let grid_size = B * C_OUT * num_ol_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_size as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        L as i32,
        OL as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C_IN * L) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "conv1d_backward_dx mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv1d_backward_dw_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d/expected_dw.bin");
    let mut dw_host = vec![0.0f32; C_OUT * C_IN * KL as usize];

    let mut x_buf = device.buffer::<f32>(B * C_IN * L)?;
    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OL)?;
    let dw_buf = device.buffer::<f32>(C_OUT * C_IN * KL as usize)?;

    x_buf.to_device(&x_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDw::<f32>::new(KL, STRIDE, PAD, BLOCK_OL);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDw<f32>,
    >(&ptx_path)?;

    let num_ol_tiles = OL.div_ceil(BLOCK_OL as usize);
    let grid_size = B * C_OUT * num_ol_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_size as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        dw_buf.as_device_ptr(),
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        L as i32,
        OL as i32,
    );

    device.launch(&program, &cfg, args)?;
    dw_buf.to_host(&mut dw_host)?;

    for i in 0..(C_OUT * C_IN * KL as usize) {
        assert!(
            (dw_host[i] - expected[i]).abs() < 1e-3,
            "conv1d_backward_dw mismatch at {i}: gpu={}, expected={}",
            dw_host[i],
            expected[i]
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests — same padding (PAD=1, OL=L=16)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "hardware")]
fn test_conv1d_padded_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/x.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/w.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "conv1d_padded/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; B * C_OUT * OL_P];

    let mut x_buf = device.buffer::<f32>(B * C_IN * L)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KL as usize)?;
    let y_buf = device.buffer::<f32>(B * C_OUT * OL_P)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dForward::<f32>::new(KL, STRIDE, PAD_P, BLOCK_OL);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[conv1d_padded_forward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<teeny_kernels::nn::conv::conv1d::Conv1dForward<f32>>(
        &ptx_path,
    )?;

    let num_ol_tiles = OL_P.div_ceil(BLOCK_OL as usize);
    let grid_size = B * C_OUT * num_ol_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_size as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        x_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        L as i32,
        OL_P as i32,
    );

    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..(B * C_OUT * OL_P) {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "conv1d_padded_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv1d_padded_backward_dx_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/dy.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/w.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/expected_dx.bin");
    let mut dx_host = vec![0.0f32; B * C_IN * L];

    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OL_P)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KL as usize)?;
    let dx_buf = device.buffer::<f32>(B * C_IN * L)?;

    dy_buf.to_device(&dy_host)?;
    w_buf.to_device(&w_host)?;

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDx::<f32>::new(KL, STRIDE, PAD_P, BLOCK_OL);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDx<f32>,
    >(&ptx_path)?;

    let num_ol_tiles = OL_P.div_ceil(BLOCK_OL as usize);
    let grid_size = B * C_OUT * num_ol_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_size as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        L as i32,
        OL_P as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C_IN * L) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "conv1d_padded_backward_dx mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv1d_padded_backward_dw_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv1d_padded/expected_dw.bin");
    let mut dw_host = vec![0.0f32; C_OUT * C_IN * KL as usize];

    let mut x_buf = device.buffer::<f32>(B * C_IN * L)?;
    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OL_P)?;
    let dw_buf = device.buffer::<f32>(C_OUT * C_IN * KL as usize)?;

    x_buf.to_device(&x_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel =
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDw::<f32>::new(KL, STRIDE, PAD_P, BLOCK_OL);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv1d::Conv1dBackwardDw<f32>,
    >(&ptx_path)?;

    let num_ol_tiles = OL_P.div_ceil(BLOCK_OL as usize);
    let grid_size = B * C_OUT * num_ol_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_size as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        x_buf.as_device_ptr(),
        dw_buf.as_device_ptr(),
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        L as i32,
        OL_P as i32,
    );

    device.launch(&program, &cfg, args)?;
    dw_buf.to_host(&mut dw_host)?;

    for i in 0..(C_OUT * C_IN * KL as usize) {
        assert!(
            (dw_host[i] - expected[i]).abs() < 1e-3,
            "conv1d_padded_backward_dw mismatch at {i}: gpu={}, expected={}",
            dw_host[i],
            expected[i]
        );
    }

    Ok(())
}
