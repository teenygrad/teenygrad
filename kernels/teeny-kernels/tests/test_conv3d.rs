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
const DV: usize = 4;
#[cfg(feature = "hardware")]
const H: usize = 4;
#[cfg(feature = "hardware")]
const W: usize = 8;
const KD: i32 = 2;
const KH: i32 = 2;
const KW: i32 = 3;
const STRIDE_D: i32 = 1;
const STRIDE_H: i32 = 1;
const STRIDE_W: i32 = 1;
const PAD_D: i32 = 0;
const PAD_H: i32 = 0;
const PAD_W: i32 = 0;
#[cfg(feature = "hardware")]
const OD: usize = (DV + 2 * PAD_D as usize - KD as usize) / STRIDE_D as usize + 1; // 3
#[cfg(feature = "hardware")]
const OH: usize = (H + 2 * PAD_H as usize - KH as usize) / STRIDE_H as usize + 1; // 3
#[cfg(feature = "hardware")]
const OW: usize = (W + 2 * PAD_W as usize - KW as usize) / STRIDE_W as usize + 1; // 6
const BLOCK_OW: i32 = 8;

// ── Padded constants (PAD_D=PAD_H=PAD_W=1) ───────────────────────────────────
#[cfg(feature = "hardware")]
const PAD_D_P: i32 = 1;
#[cfg(feature = "hardware")]
const PAD_H_P: i32 = 1;
#[cfg(feature = "hardware")]
const PAD_W_P: i32 = 1;
#[cfg(feature = "hardware")]
const OD_P: usize = (DV + 2 * PAD_D_P as usize - KD as usize) / STRIDE_D as usize + 1; // 5
#[cfg(feature = "hardware")]
const OH_P: usize = (H + 2 * PAD_H_P as usize - KH as usize) / STRIDE_H as usize + 1; // 5
#[cfg(feature = "hardware")]
const OW_P: usize = (W + 2 * PAD_W_P as usize - KW as usize) / STRIDE_W as usize + 1; // 8

#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_forward_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dForward::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, BLOCK_OW,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("conv3d_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("conv3d_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_conv3d_backward_dx_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dBackwardDx::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, BLOCK_OW,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("conv3d_backward_dx_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("conv3d_backward_dx_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_conv3d_backward_dw_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dBackwardDw::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, BLOCK_OW,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("conv3d_backward_dw_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("conv3d_backward_dw_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests — no padding (PAD=0)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "hardware")]
fn test_conv3d_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/x.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/w.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/expected_forward.bin");
    let mut y_host = vec![0.0f32; B * C_OUT * OD * OH * OW];

    let mut x_buf = device.buffer::<f32>(B * C_IN * DV * H * W)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KD as usize * KH as usize * KW as usize)?;
    let y_buf = device.buffer::<f32>(B * C_OUT * OD * OH * OW)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dForward::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[conv3d_forward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<teeny_kernels::nn::conv::conv3d::Conv3dForward<f32>>(
        &ptx_path,
    )?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OD * OH * num_ow_tiles;
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
        DV as i32,
        H as i32,
        W as i32,
        OD as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..(B * C_OUT * OD * OH * OW) {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "conv3d_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv3d_backward_dx_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/dy.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/w.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/expected_dx.bin");
    let mut dx_host = vec![0.0f32; B * C_IN * DV * H * W];

    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OD * OH * OW)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KD as usize * KH as usize * KW as usize)?;
    let dx_buf = device.buffer::<f32>(B * C_IN * DV * H * W)?;

    dy_buf.to_device(&dy_host)?;
    w_buf.to_device(&w_host)?;

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dBackwardDx::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv3d::Conv3dBackwardDx<f32>,
    >(&ptx_path)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OD * OH * num_ow_tiles;
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
        DV as i32,
        H as i32,
        W as i32,
        OD as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C_IN * DV * H * W) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "conv3d_backward_dx mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv3d_backward_dw_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d/expected_dw.bin");
    let mut dw_host = vec![0.0f32; C_OUT * C_IN * KD as usize * KH as usize * KW as usize];

    let mut x_buf = device.buffer::<f32>(B * C_IN * DV * H * W)?;
    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OD * OH * OW)?;
    let dw_buf = device.buffer::<f32>(C_OUT * C_IN * KD as usize * KH as usize * KW as usize)?;

    x_buf.to_device(&x_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dBackwardDw::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D, PAD_H, PAD_W, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv3d::Conv3dBackwardDw<f32>,
    >(&ptx_path)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OD * OH * num_ow_tiles;
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
        DV as i32,
        H as i32,
        W as i32,
        OD as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    dw_buf.to_host(&mut dw_host)?;

    for i in 0..(C_OUT * C_IN * KD as usize * KH as usize * KW as usize) {
        assert!(
            (dw_host[i] - expected[i]).abs() < 1e-3,
            "conv3d_backward_dw mismatch at {i}: gpu={}, expected={}",
            dw_host[i],
            expected[i]
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests — padded (PAD_D=PAD_H=PAD_W=1)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "hardware")]
fn test_conv3d_padded_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/x.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/w.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "conv3d_padded/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; B * C_OUT * OD_P * OH_P * OW_P];

    let mut x_buf = device.buffer::<f32>(B * C_IN * DV * H * W)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KD as usize * KH as usize * KW as usize)?;
    let y_buf = device.buffer::<f32>(B * C_OUT * OD_P * OH_P * OW_P)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dForward::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D_P, PAD_H_P, PAD_W_P, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[conv3d_padded_forward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<teeny_kernels::nn::conv::conv3d::Conv3dForward<f32>>(
        &ptx_path,
    )?;

    let num_ow_tiles = OW_P.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OD_P * OH_P * num_ow_tiles;
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
        DV as i32,
        H as i32,
        W as i32,
        OD_P as i32,
        OH_P as i32,
        OW_P as i32,
    );

    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..(B * C_OUT * OD_P * OH_P * OW_P) {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "conv3d_padded_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv3d_padded_backward_dx_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/dy.bin");
    let w_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/w.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/expected_dx.bin");
    let mut dx_host = vec![0.0f32; B * C_IN * DV * H * W];

    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OD_P * OH_P * OW_P)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KD as usize * KH as usize * KW as usize)?;
    let dx_buf = device.buffer::<f32>(B * C_IN * DV * H * W)?;

    dy_buf.to_device(&dy_host)?;
    w_buf.to_device(&w_host)?;

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dBackwardDx::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D_P, PAD_H_P, PAD_W_P, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv3d::Conv3dBackwardDx<f32>,
    >(&ptx_path)?;

    let num_ow_tiles = OW_P.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OD_P * OH_P * num_ow_tiles;
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
        DV as i32,
        H as i32,
        W as i32,
        OD_P as i32,
        OH_P as i32,
        OW_P as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C_IN * DV * H * W) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "conv3d_padded_backward_dx mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_conv3d_padded_backward_dw_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "conv3d_padded/expected_dw.bin");
    let mut dw_host = vec![0.0f32; C_OUT * C_IN * KD as usize * KH as usize * KW as usize];

    let mut x_buf = device.buffer::<f32>(B * C_IN * DV * H * W)?;
    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OD_P * OH_P * OW_P)?;
    let dw_buf = device.buffer::<f32>(C_OUT * C_IN * KD as usize * KH as usize * KW as usize)?;

    x_buf.to_device(&x_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::nn::conv::conv3d::Conv3dBackwardDw::<f32>::new(
        KD, KH, KW, STRIDE_D, STRIDE_H, STRIDE_W, PAD_D_P, PAD_H_P, PAD_W_P, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::conv::conv3d::Conv3dBackwardDw<f32>,
    >(&ptx_path)?;

    let num_ow_tiles = OW_P.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OD_P * OH_P * num_ow_tiles;
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
        DV as i32,
        H as i32,
        W as i32,
        OD_P as i32,
        OH_P as i32,
        OW_P as i32,
    );

    device.launch(&program, &cfg, args)?;
    dw_buf.to_host(&mut dw_host)?;

    for i in 0..(C_OUT * C_IN * KD as usize * KH as usize * KW as usize) {
        assert!(
            (dw_host[i] - expected[i]).abs() < 1e-3,
            "conv3d_padded_backward_dw mismatch at {i}: gpu={}, expected={}",
            dw_host[i],
            expected[i]
        );
    }

    Ok(())
}
