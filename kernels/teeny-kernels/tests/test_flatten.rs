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
const B: usize = 64;
#[cfg(feature = "hardware")]
const N: usize = 96;
const BLOCK_B: i32 = 32;
const BLOCK_N: i32 = 32;

// Over-allocated buffer: every other row is selected (stride_ib = 2*N).
#[cfg(feature = "hardware")]
const PAD_ROWS: usize = 2 * B;

/// Must match `.reqntid` in the generated PTX.
#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_flatten_forward_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::mlp::flatten::FlattenForward::<f32>::new(BLOCK_B, BLOCK_N);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("flatten_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("flatten_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

#[test]
fn test_flatten_backward_mlir_output() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::mlp::flatten::FlattenBackward::<f32>::new(BLOCK_B, BLOCK_N);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("flatten_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("flatten_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests
// ---------------------------------------------------------------------------

/// Forward: strided-row input → contiguous row-major output.
///
/// The input buffer holds [2*B, N] values in row-major order, but we only
/// read every other row (stride_ib = 2*N, stride_in = 1). This exercises the
/// non-unit outer-dimension stride path while keeping the inner (fastest-
/// varying) stride at 1 as required by the TMA descriptor hardware.
///
/// After flatten_forward, output[b, n] must equal raw_input[2*b, n].
#[test]
#[cfg(feature = "hardware")]
fn test_flatten_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    // padded is [PAD_ROWS, N]; expected_forward is [B, N] = even-indexed rows.
    let padded_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "flatten/padded.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "flatten/expected_forward.bin");
    let mut output_host = vec![0.0f32; B * N];

    let mut input_buf = device.buffer::<f32>(PAD_ROWS * N)?;
    let output_buf = device.buffer::<f32>(B * N)?;

    input_buf.to_device(&padded_host)?;

    let kernel = teeny_kernels::nn::mlp::flatten::FlattenForward::<f32>::new(BLOCK_B, BLOCK_N);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[flatten_forward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::mlp::flatten::FlattenForward<f32>,
    >(&ptx_path)?;

    let grid_x = (B as u32).div_ceil(BLOCK_B as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = teeny_runtime::launch_config_custom(
        [grid_x, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    // Every-other-row strides: stride_ib = 2*N (skip one full row), stride_in = 1.
    let args = (
        input_buf.as_device_ptr(),
        output_buf.as_device_ptr(),
        B as i32,
        N as i32,
        (2 * N) as i32, // stride_ib: every-other-row
        1i32,           // stride_in: contiguous columns
    );

    device.launch(&program, &cfg, args)?;
    output_buf.to_host(&mut output_host)?;

    for i in 0..(B * N) {
        assert!(
            (output_host[i] - expected[i]).abs() < 1e-5,
            "flatten_forward mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

/// Backward: contiguous row-major dy → strided-row dx.
///
/// The inverse of the forward test: the upstream gradient dy is contiguous
/// [B, N] row-major; we write it back to dx using every-other-row strides
/// (stride_dxb = 2*N, stride_dxn = 1) so that raw_dx[2*b, n] == dy[b, n].
/// Odd rows of raw_dx must remain zero (unwritten).
#[test]
#[cfg(feature = "hardware")]
fn test_flatten_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    // dy is [B, N]; expected_backward is [PAD_ROWS, N] with even rows = dy, odd = 0.
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "flatten/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "flatten/expected_backward.bin");
    let dx_init = vec![0.0f32; PAD_ROWS * N];

    let mut dy_buf = device.buffer::<f32>(B * N)?;
    let mut dx_buf = device.buffer::<f32>(PAD_ROWS * N)?;
    let mut dx_host = vec![0.0f32; PAD_ROWS * N];

    dy_buf.to_device(&dy_host)?;
    dx_buf.to_device(&dx_init)?;

    let kernel = teeny_kernels::nn::mlp::flatten::FlattenBackward::<f32>::new(BLOCK_B, BLOCK_N);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    println!("[flatten_backward] compiled PTX: {ptx_path}");

    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::mlp::flatten::FlattenBackward<f32>,
    >(&ptx_path)?;

    let grid_x = (B as u32).div_ceil(BLOCK_B as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = teeny_runtime::launch_config_custom(
        [grid_x, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );

    // Write to every-other-row of dx: stride_dxb = 2*N, stride_dxn = 1.
    let args = (
        dy_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        B as i32,
        N as i32,
        (2 * N) as i32, // stride_dxb
        1i32,           // stride_dxn
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(PAD_ROWS * N) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "flatten_backward mismatch at flat index {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}
