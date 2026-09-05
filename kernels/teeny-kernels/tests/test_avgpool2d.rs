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

#[cfg(feature = "hardware")]
const B: usize = 2;
#[cfg(feature = "hardware")]
const C: usize = 4;
#[cfg(feature = "hardware")]
const H: usize = 8;
#[cfg(feature = "hardware")]
const W: usize = 8;
const KH: i32 = 2;
const KW: i32 = 2;
const STRIDE_H: i32 = 2;
const STRIDE_W: i32 = 2;
#[cfg(feature = "hardware")]
const OH: usize = (H - KH as usize) / STRIDE_H as usize + 1; // 4
#[cfg(feature = "hardware")]
const OW: usize = (W - KW as usize) / STRIDE_W as usize + 1; // 4
const BLOCK_OW: i32 = 4;

/// Must match `.reqntid` in the generated PTX.
#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_avgpool2d_forward_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::pool::avgpool2d::Avgpool2dForward::<f32>::new(
        KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("avgpool2d_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("avgpool2d_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_avgpool2d_backward_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::pool::avgpool2d::Avgpool2dBackward::<f32>::new(
        KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("avgpool2d_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("avgpool2d_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "hardware")]
fn test_avgpool2d_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let input_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "avgpool2d/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "avgpool2d/expected_forward.bin");
    let mut output_host = vec![0.0f32; B * C * OH * OW];

    let mut input_buf = device.buffer::<f32>(B * C * H * W)?;
    let output_buf = device.buffer::<f32>(B * C * OH * OW)?;

    input_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::nn::pool::avgpool2d::Avgpool2dForward::<f32>::new(
        KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[avgpool2d_forward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::pool::avgpool2d::Avgpool2dForward<f32>,
    >(&ptx_path)?;

    let num_ow_tiles = (OW as u32).div_ceil(BLOCK_OW as u32);
    let grid_x = (B * C * OH) as u32 * num_ow_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_x, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        input_buf.as_device_ptr(),
        output_buf.as_device_ptr(),
        B as i32,
        C as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    output_buf.to_host(&mut output_host)?;

    for i in 0..(B * C * OH * OW) {
        assert!(
            (output_host[i] - expected[i]).abs() < 1e-4,
            "avgpool2d_forward mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_avgpool2d_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "avgpool2d/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "avgpool2d/expected_backward.bin",
    );
    let zeros = vec![0.0f32; B * C * H * W];
    let mut dx_host = vec![0.0f32; B * C * H * W];

    let mut dy_buf = device.buffer::<f32>(B * C * OH * OW)?;
    // dx must be zero-initialised before launching (atomics accumulate).
    let mut dx_zero_buf = device.buffer::<f32>(B * C * H * W)?;

    dy_buf.to_device(&dy_host)?;
    dx_zero_buf.to_device(&zeros)?;

    let kernel = teeny_kernels::nn::pool::avgpool2d::Avgpool2dBackward::<f32>::new(
        KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[avgpool2d_backward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::pool::avgpool2d::Avgpool2dBackward<f32>,
    >(&ptx_path)?;

    let num_ow_tiles = (OW as u32).div_ceil(BLOCK_OW as u32);
    let grid_x = (B * C * OH) as u32 * num_ow_tiles;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_x, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    let args = (
        dy_buf.as_device_ptr(),
        dx_zero_buf.as_device_ptr(),
        B as i32,
        C as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_zero_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C * H * W) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "avgpool2d_backward mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}
