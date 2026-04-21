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

// 64 rows, 128 columns — BLOCK_SIZE == N_COLS (no masking required).
const N_ROWS: usize = 64;
const N_COLS: usize = 128;
const BLOCK_SIZE: i32 = 128;

/// Must match `.reqntid` in the generated PTX.
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// CPU reference helpers
// ---------------------------------------------------------------------------

/// Row-wise softmax: `y[i] = exp(x[i] - max(row)) / sum(exp(x - max(row)))`.
fn cpu_softmax(input: &Array2<f32>) -> Array2<f32> {
    let mut out = input.clone();
    for mut row in out.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        row.mapv_inplace(|v| (v - max).exp() / exp_sum);
    }
    out
}

/// Softmax backward: `dx_i = y_i * (dy_i - sum_j(y_j * dy_j))`.
fn cpu_softmax_backward(y: &Array2<f32>, dy: &Array2<f32>) -> Array2<f32> {
    let mut dx = Array2::zeros(y.raw_dim());
    for (((y_row, dy_row), mut dx_row)) in y.rows().into_iter().zip(dy.rows()).zip(dx.rows_mut()) {
        let dot: f32 = y_row.iter().zip(dy_row.iter()).map(|(&a, &b)| a * b).sum();
        for ((&yi, &dyi), dxi) in y_row.iter().zip(dy_row.iter()).zip(dx_row.iter_mut()) {
            *dxi = yi * (dyi - dot);
        }
    }
    dx
}

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_forward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxForward::<f32, BLOCK_SIZE>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("softmax_forward_source", kernel.source());
    assert_debug_snapshot!("softmax_forward_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_softmax_backward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxBackward::<f32, BLOCK_SIZE>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("softmax_backward_source", kernel.source());
    assert_debug_snapshot!("softmax_backward_mlir", mlir.trim());

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests
// ---------------------------------------------------------------------------

/// Forward: GPU softmax output must match the CPU reference row-by-row.
#[test]
#[cfg(feature = "cuda")]
fn test_softmax_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array2::<f32>::random((N_ROWS, N_COLS), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let input_host: Vec<f32> = input_arr.iter().copied().collect();
    let expected_arr = cpu_softmax(&input_arr);
    let expected: Vec<f32> = expected_arr.iter().copied().collect();

    let mut x_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let y_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut y_host = vec![0.0f32; N_ROWS * N_COLS];

    x_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxForward::<f32, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[softmax_forward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::softmax::SoftmaxForward<f32, BLOCK_SIZE>,
    >(&ptx)?;

    // Grid: one CTA per row.
    let cfg = CudaLaunchConfig {
        grid: [N_ROWS as u32, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        N_ROWS as i32,
        N_COLS as i32,
    );

    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..(N_ROWS * N_COLS) {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "softmax_forward mismatch at index {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    // Verify rows sum to 1.
    for r in 0..N_ROWS {
        let row_sum: f32 = y_host[r * N_COLS..(r + 1) * N_COLS].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "row {r} sums to {row_sum}, expected 1.0"
        );
    }

    Ok(())
}

/// Backward: GPU dx must match `y * (dy - sum(y * dy))` computed on the CPU.
#[test]
#[cfg(feature = "cuda")]
fn test_softmax_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // y must be a valid softmax output (all positive, rows sum to 1).
    let x_arr = Array2::<f32>::random((N_ROWS, N_COLS), Uniform::new(-3.0f32, 3.0f32).unwrap());
    let y_arr = cpu_softmax(&x_arr);
    let dy_arr = Array2::<f32>::random((N_ROWS, N_COLS), Uniform::new(-1.0f32, 1.0f32).unwrap());

    let y_host: Vec<f32> = y_arr.iter().copied().collect();
    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();
    let expected_arr = cpu_softmax_backward(&y_arr, &dy_arr);
    let expected: Vec<f32> = expected_arr.iter().copied().collect();

    let mut dy_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut y_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let dx_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut dx_host = vec![0.0f32; N_ROWS * N_COLS];

    dy_buf.to_device(&dy_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxBackward::<f32, BLOCK_SIZE>::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[softmax_backward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::softmax::SoftmaxBackward<f32, BLOCK_SIZE>,
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid: [N_ROWS as u32, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N_ROWS as i32,
        N_COLS as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(N_ROWS * N_COLS) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "softmax_backward mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}
