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
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};
use teeny_kernels::testing::load_fixture;

const M: usize = 64;
const N: usize = 48;
const K: usize = 64;
const BLOCK_M: i32 = 32;
const BLOCK_N: i32 = 32;
const BLOCK_K: i32 = 32;
const GROUP_M: i32 = 8;

/// Must match `.reqntid` in the generated PTX.
const PTX_LAUNCH_THREADS_X: u32 = 128;

// TF32 tensor-core precision, not full f32: error scales with the magnitude
// of the accumulated sum (K=64 here), so atol + rtol * |expected| is used
// instead of a flat absolute tolerance.
const ATOL: f32 = 1e-1;
const RTOL: f32 = 2e-2;

fn tf32_close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() < ATOL + RTOL * expected.abs()
}

/// Naive host-side reference for `LinearForward`: `y = x @ w^T (+ bias)`.
/// `x` is (M, K) row-major, `w` is (N, K) row-major (PyTorch `nn.Linear`
/// weight layout — out_features first), `y` is (M, N) row-major.
#[cfg(feature = "cuda")]
fn linear_reference(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w[ni * k + ki];
            }
            if let Some(b) = bias {
                acc += b[ni];
            }
            y[mi * n + ni] = acc;
        }
    }
    y
}

#[test]
fn test_linear_mlir_without_bias_output() -> Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::mlp::linear::LinearForward::<f32>::new(
        false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_source", kernel.source());
    assert_debug_snapshot!("linear_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_linear_mlir_with_bias_output() -> Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::mlp::linear::LinearForward::<f32>::new(
        true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_with_bias_source", kernel.source());
    assert_debug_snapshot!("linear_with_bias_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_linear_backward_mlir_without_bias_output() -> Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::mlp::linear::LinearBackward::<f32>::new(
        false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_backward_source", kernel.source());
    assert_debug_snapshot!("linear_backward_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_linear_backward_mlir_with_bias_output() -> Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::mlp::linear::LinearBackward::<f32>::new(
        true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("linear_backward_with_bias_source", kernel.source());
    assert_debug_snapshot!("linear_backward_with_bias_mlir", mlir.trim());

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_forward_no_bias_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_host = load_fixture("linear/input.bin");
    let weight_host = load_fixture("linear/weight.bin");
    let bias_host = vec![0.0f32; N]; // no-bias kernel ignores the pointer value
    let expected = load_fixture("linear/expected_no_bias.bin");
    let mut output_host = vec![0.0f32; M * N];

    let mut in_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(M * N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::nn::mlp::linear::LinearForward::<f32>::new(
        false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true, false)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program =
        testing::load_program_from_ptx::<teeny_kernels::nn::mlp::linear::LinearForward<f32>>(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    println!(
        "[8/9] launching: grid={:?} block={:?} M={M} N={N} K={K}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        K as i32,
        1i32,
        N as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

    out_buf.to_host(&mut output_host)?;
    println!(
        "[9/9] copied results back: output[0]={} output[{}]={}",
        output_host[0],
        (M * N) - 1,
        output_host[(M * N) - 1]
    );

    for i in 0..(M * N) {
        assert!(
            tf32_close(output_host[i], expected[i]),
            "linear (no bias) mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_forward_with_bias_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_host = load_fixture("linear/input.bin");
    let weight_host = load_fixture("linear/weight.bin");
    let bias_host = load_fixture("linear/bias.bin");
    let expected = load_fixture("linear/expected_with_bias.bin");
    let mut output_host = vec![0.0f32; M * N];

    let mut in_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(M * N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::nn::mlp::linear::LinearForward::<f32>::new(
        true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true, false)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program =
        testing::load_program_from_ptx::<teeny_kernels::nn::mlp::linear::LinearForward<f32>>(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    println!(
        "[8/9] launching: grid={:?} block={:?} M={M} N={N} K={K}",
        cfg.grid, cfg.block,
    );

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        K as i32,
        1i32,
        N as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

    out_buf.to_host(&mut output_host)?;
    println!(
        "[9/9] copied results back: output[0]={} output[{}]={}",
        output_host[0],
        (M * N) - 1,
        output_host[(M * N) - 1]
    );

    for i in 0..(M * N) {
        assert!(
            tf32_close(output_host[i], expected[i]),
            "linear (with bias) mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

// ── Inline data + pipeline-stage logging ─────────────────────────────────────
//
// Same LinearForward kernel as above (the simplest tensor-core-eligible tl.dot
// call in the codebase — see LinearForward's single T::dot call), but with
// data generated inline instead of loaded from fixtures, and compiled with
// `compile_kernel(..., debug=true)` so teenyc's ttir/ttgpuir/llir/llvmir/ptx
// pipeline stages are logged to stderr as the compile runs. Run with
// `--nocapture` (and redirect stderr) to capture them, e.g.:
//
//   cargo test -p teeny-kernels --test test_linear --features cuda \
//     test_linear_forward_logs_pipeline_stages -- --nocapture 2>pipeline.log

#[test]
#[cfg(feature = "cuda")]
fn test_linear_forward_logs_pipeline_stages() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // Deterministic inline data instead of the .bin fixtures the tests above
    // use, so this test is fully self-contained.
    let input_host: Vec<f32> = (0..M * K).map(|i| (i as f32 % 13.0 - 6.0) * 0.05).collect();
    let weight_host: Vec<f32> = (0..N * K).map(|i| (i as f32 % 11.0 - 5.0) * 0.05).collect();
    let bias_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.01 - 0.1).collect();
    let expected = linear_reference(&input_host, &weight_host, Some(&bias_host), M, N, K);
    let mut output_host = vec![0.0f32; M * N];

    let mut in_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut bias_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(M * N)?;

    in_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::nn::mlp::linear::LinearForward::<f32>::new(
        true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);

    // force=true so teenyc actually runs (a cache hit would emit no pipeline logs).
    let ptx_path = compile_kernel(&kernel, &target, true, true)?;
    println!("compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program =
        testing::load_program_from_ptx::<teeny_kernels::nn::mlp::linear::LinearForward<f32>>(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32) * (N as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        bias_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        K as i32,
        1i32,
        N as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut output_host)?;

    for i in 0..(M * N) {
        assert!(
            tf32_close(output_host[i], expected[i]),
            "linear (inline data) mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_backward_without_bias_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_host = load_fixture("linear/input.bin");
    let weight_host = load_fixture("linear/weight.bin");
    let dy_host = load_fixture("linear/dy.bin");
    let expected_dx = load_fixture("linear/expected_dx.bin");
    let expected_dw = load_fixture("linear/expected_dw.bin");

    let mut dx_host = vec![0.0f32; M * K];
    let mut dw_host = vec![0.0f32; N * K];

    let mut x_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut dy_buf = device.buffer::<f32>(M * N)?;
    let dx_buf = device.buffer::<f32>(M * K)?;
    let dw_buf = device.buffer::<f32>(N * K)?;
    let db_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::nn::mlp::linear::LinearBackward::<f32>::new(
        false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true, false)?;
    println!("[linear_backward no bias] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::mlp::linear::LinearBackward<f32>,
    >(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32)
        * (N as u32).div_ceil(BLOCK_N as u32)
        * (K as u32).div_ceil(BLOCK_K as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        dy_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        dw_buf.as_device_ptr() as *mut f32,
        db_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        N as i32,
        1i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;
    dw_buf.to_host(&mut dw_host)?;

    for i in 0..(M * K) {
        assert!(
            tf32_close(dx_host[i], expected_dx[i]),
            "linear_backward (dx, no bias) mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            expected_dx[i]
        );
    }

    for i in 0..(N * K) {
        assert!(
            tf32_close(dw_host[i], expected_dw[i]),
            "linear_backward (dw, no bias) mismatch at index {i}: gpu={}, expected={}",
            dw_host[i],
            expected_dw[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_linear_backward_with_bias_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_host = load_fixture("linear/input.bin");
    let weight_host = load_fixture("linear/weight.bin");
    let dy_host = load_fixture("linear/dy.bin");
    let expected_dx = load_fixture("linear/expected_dx.bin");
    let expected_dw = load_fixture("linear/expected_dw.bin");
    let expected_db = load_fixture("linear/expected_db.bin");

    let mut dx_host = vec![0.0f32; M * K];
    let mut dw_host = vec![0.0f32; N * K];
    let mut db_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(M * K)?;
    let mut w_buf = device.buffer::<f32>(N * K)?;
    let mut dy_buf = device.buffer::<f32>(M * N)?;
    let dx_buf = device.buffer::<f32>(M * K)?;
    let dw_buf = device.buffer::<f32>(N * K)?;
    let db_buf = device.buffer::<f32>(N)?;

    x_buf.to_device(&input_host)?;
    w_buf.to_device(&weight_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::nn::mlp::linear::LinearBackward::<f32>::new(
        true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true, false)?;
    println!("[linear_backward with bias] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::mlp::linear::LinearBackward<f32>,
    >(&ptx)?;

    let grid_x = (M as u32).div_ceil(BLOCK_M as u32)
        * (N as u32).div_ceil(BLOCK_N as u32)
        * (K as u32).div_ceil(BLOCK_K as u32);
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        dy_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        dw_buf.as_device_ptr() as *mut f32,
        db_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        K as i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        N as i32,
        1i32,
        K as i32,
        1i32,
        1i32,
        K as i32,
        1i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;
    dw_buf.to_host(&mut dw_host)?;
    db_buf.to_host(&mut db_host)?;

    for i in 0..(M * K) {
        assert!(
            tf32_close(dx_host[i], expected_dx[i]),
            "linear_backward (dx, with bias) mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            expected_dx[i]
        );
    }

    for i in 0..(N * K) {
        assert!(
            tf32_close(dw_host[i], expected_dw[i]),
            "linear_backward (dw, with bias) mismatch at index {i}: gpu={}, expected={}",
            dw_host[i],
            expected_dw[i]
        );
    }

    for i in 0..N {
        assert!(
            tf32_close(db_host[i], expected_db[i]),
            "linear_backward (db) mismatch at index {i}: gpu={}, expected={}",
            db_host[i],
            expected_db[i]
        );
    }

    Ok(())
}
