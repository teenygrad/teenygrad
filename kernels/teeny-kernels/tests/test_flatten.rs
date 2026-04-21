/*
 * Copyright (c) 2026 Teenygrad.
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
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::PathBuf;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::context::buffer::Buffer;
use teeny_core::context::device::Device;
use teeny_core::context::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

// Tensor shape: B rows (batch), N columns (features).
const B: usize = 64;
const N: usize = 96;
const BLOCK_B: i32 = 32;
const BLOCK_N: i32 = 32;

// Number of rows in the overallocated buffer used by the strided-row tests.
// Every other row is selected (stride_ib = 2*N).
const PAD_ROWS: usize = 2 * B;

/// Must match `.reqntid` in the generated PTX.
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_flatten_forward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::mlp::flatten::FlattenForward::<f32, BLOCK_B, BLOCK_N>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("flatten_forward_source", kernel.source());
    assert_debug_snapshot!("flatten_forward_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_flatten_backward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::mlp::flatten::FlattenBackward::<f32, BLOCK_B, BLOCK_N>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("flatten_backward_source", kernel.source());
    assert_debug_snapshot!("flatten_backward_mlir", mlir.trim());

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
#[cfg(feature = "cuda")]
fn test_flatten_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // Allocate a PAD_ROWS x N buffer; we will sample every other row.
    let padded = Array2::<f32>::random((PAD_ROWS, N), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let padded_host: Vec<f32> = padded.iter().copied().collect();

    // Expected output: even-indexed rows, row-major.
    let mut expected = vec![0.0f32; B * N];
    for b in 0..B {
        for n in 0..N {
            expected[b * N + n] = padded[[2 * b, n]];
        }
    }

    let mut input_buf = device.buffer::<f32>(PAD_ROWS * N)?;
    let output_buf = device.buffer::<f32>(B * N)?;
    let mut output_host = vec![0.0f32; B * N];

    input_buf.to_device(&padded_host)?;

    let kernel = teeny_kernels::mlp::flatten::FlattenForward::<f32, BLOCK_B, BLOCK_N>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[flatten_forward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::flatten::FlattenForward<f32, BLOCK_B, BLOCK_N>,
    >(&ptx)?;

    let grid_x = (B as u32).div_ceil(BLOCK_B as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    // Every-other-row strides: stride_ib = 2*N (skip one full row), stride_in = 1.
    let args = (
        input_buf.as_device_ptr() as *mut f32,
        output_buf.as_device_ptr() as *mut f32,
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
#[cfg(feature = "cuda")]
fn test_flatten_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_arr = Array2::<f32>::random((B, N), Uniform::new(-2.0f32, 2.0f32).unwrap());
    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();

    // dx buffer is PAD_ROWS x N, zero-initialised so odd rows stay zero.
    let dx_init = vec![0.0f32; PAD_ROWS * N];

    let mut dy_buf = device.buffer::<f32>(B * N)?;
    let mut dx_buf = device.buffer::<f32>(PAD_ROWS * N)?;
    let mut dx_host = vec![0.0f32; PAD_ROWS * N];

    dy_buf.to_device(&dy_host)?;
    dx_buf.to_device(&dx_init)?;

    let kernel = teeny_kernels::mlp::flatten::FlattenBackward::<f32, BLOCK_B, BLOCK_N>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[flatten_backward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::mlp::flatten::FlattenBackward<f32, BLOCK_B, BLOCK_N>,
    >(&ptx)?;

    let grid_x = (B as u32).div_ceil(BLOCK_B as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    // Write to every-other-row of dx: stride_dxb = 2*N, stride_dxn = 1.
    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        B as i32,
        N as i32,
        (2 * N) as i32, // stride_dxb
        1i32,           // stride_dxn
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    // Even rows must match dy; odd rows must remain zero.
    for b in 0..B {
        for n in 0..N {
            let gpu_even = dx_host[2 * b * N + n];
            let exp = dy_arr[[b, n]];
            assert!(
                (gpu_even - exp).abs() < 1e-5,
                "flatten_backward even-row mismatch at (b={b}, n={n}): gpu={gpu_even}, expected={exp}"
            );

            if b + 1 < B || n < N {
                let odd_row = 2 * b + 1;
                if odd_row < PAD_ROWS {
                    let gpu_odd = dx_host[odd_row * N + n];
                    assert!(
                        gpu_odd == 0.0,
                        "flatten_backward odd row {odd_row} should be zero at n={n}, got {gpu_odd}"
                    );
                }
            }
        }
    }

    Ok(())
}
