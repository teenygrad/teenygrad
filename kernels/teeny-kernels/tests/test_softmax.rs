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
use std::path::PathBuf;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{
    compiler::target::Capability, errors::Result, testing, device::CudaLaunchConfig,
};

const N_ROWS: usize = 64;
const N_COLS: usize = 128;
const BLOCK_SIZE: i32 = 128;

/// Must match `.reqntid` in the generated PTX.
const PTX_LAUNCH_THREADS_X: u32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_forward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxForward::<f32>::new(BLOCK_SIZE);
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

    let kernel = teeny_kernels::activation::softmax::SoftmaxBackward::<f32>::new(BLOCK_SIZE);
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

/// Forward: GPU softmax output must match the PyTorch reference row-by-row.
#[test]
#[cfg(feature = "cuda")]
fn test_softmax_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_host = load_fixture("softmax/x_forward.bin");
    let expected = load_fixture("softmax/expected_forward.bin");
    let mut y_host = vec![0.0f32; N_ROWS * N_COLS];

    let mut x_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let y_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;

    x_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[softmax_forward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::softmax::SoftmaxForward<f32>,
    >(&ptx)?;

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

/// Backward: GPU dx must match `y * (dy - sum(y * dy))` computed by PyTorch.
#[test]
#[cfg(feature = "cuda")]
fn test_softmax_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let y_host = load_fixture("softmax/y_backward.bin");
    let dy_host = load_fixture("softmax/dy_backward.bin");
    let expected = load_fixture("softmax/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N_ROWS * N_COLS];

    let mut dy_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut y_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let dx_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;

    dy_buf.to_device(&dy_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::activation::softmax::SoftmaxBackward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[softmax_backward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::softmax::SoftmaxBackward<f32>,
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
