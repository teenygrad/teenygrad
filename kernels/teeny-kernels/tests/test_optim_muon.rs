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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};
#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};

// Tile sizes used for snapshot / CUDA tests
const BLOCK_SIZE: i32 = 128;
const BLOCK_R: i32    = 32;
const BLOCK_K: i32    = 32;
const GROUP_R: i32    = 8;
const BLOCK_M: i32    = 32;
const BLOCK_N: i32    = 32;
const GROUP_M: i32    = 8;

// Matrix dimensions for CUDA correctness tests
const M: usize = 32;
const N: usize = 64;

const LR: f32  = 0.01;
const A: f32   = 1.5;
const B: f32   = -0.5;

// ── reference CPU helpers ─────────────────────────────────────────────────────

/// Dense matmul C[m,n] = A[m,k] @ B[k,n] (row-major).
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k { s += a[i * k + p] * b[p * n + j]; }
            c[i * n + j] = s;
        }
    }
    c
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_muon_frob_norm_sq_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::muon::MuonFrobNormSq::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("muon_frob_norm_sq_source", kernel.source());
    assert_debug_snapshot!("muon_frob_norm_sq_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_muon_ns_xtx_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);

    // !TRANSPOSE: T = X @ X.T
    let k_no_t = teeny_kernels::nn::optim::muon::MuonNsXtx::new(false, BLOCK_R, BLOCK_K, GROUP_R);
    let ptx_no_t = PathBuf::from(compile_kernel(&k_no_t, &target, true)?);
    let mlir_no_t = std::fs::read_to_string(ptx_no_t.with_extension("mlir"))?;
    assert_debug_snapshot!("muon_ns_xtx_source",    k_no_t.source());
    assert_debug_snapshot!("muon_ns_xtx_mlir",      mlir_no_t.trim());

    // TRANSPOSE: T = X.T @ X
    let k_t = teeny_kernels::nn::optim::muon::MuonNsXtx::new(true, BLOCK_R, BLOCK_K, GROUP_R);
    let ptx_t = PathBuf::from(compile_kernel(&k_t, &target, true)?);
    let mlir_t = std::fs::read_to_string(ptx_t.with_extension("mlir"))?;
    assert_debug_snapshot!("muon_ns_xtx_transpose_source", k_t.source());
    assert_debug_snapshot!("muon_ns_xtx_transpose_mlir",   mlir_t.trim());

    Ok(())
}

#[test]
fn test_muon_ns_step_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);

    // !TRANSPOSE: X ← a·X + b·(T·X)
    let k_no_t = teeny_kernels::nn::optim::muon::MuonNsStep::new(false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let ptx_no_t = PathBuf::from(compile_kernel(&k_no_t, &target, true)?);
    let mlir_no_t = std::fs::read_to_string(ptx_no_t.with_extension("mlir"))?;
    assert_debug_snapshot!("muon_ns_step_source", k_no_t.source());
    assert_debug_snapshot!("muon_ns_step_mlir",   mlir_no_t.trim());

    // TRANSPOSE: X ← a·X + b·(X·T)
    let k_t = teeny_kernels::nn::optim::muon::MuonNsStep::new(true, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let ptx_t = PathBuf::from(compile_kernel(&k_t, &target, true)?);
    let mlir_t = std::fs::read_to_string(ptx_t.with_extension("mlir"))?;
    assert_debug_snapshot!("muon_ns_step_transpose_source", k_t.source());
    assert_debug_snapshot!("muon_ns_step_transpose_mlir",   mlir_t.trim());

    Ok(())
}

#[test]
fn test_muon_update_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::optim::muon::MuonUpdate::new(BLOCK_SIZE);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("muon_update_source", kernel.source());
    assert_debug_snapshot!("muon_update_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA correctness tests ────────────────────────────────────────────────────

/// Verify muon_update: W_out[i] = W_in[i] - lr * G[i].
#[test]
#[cfg(feature = "cuda")]
fn test_muon_update_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let n = M * N;

    let params_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let grad:      Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let expected:  Vec<f32> = params_in.iter().zip(&grad).map(|(p, g)| p - LR * g).collect();

    let mut params_buf = env.device.buffer::<f32>(n)?;
    let mut grad_buf   = env.device.buffer::<f32>(n)?;
    params_buf.to_device(&params_in)?;
    grad_buf.to_device(&grad)?;

    let kernel = teeny_kernels::nn::optim::muon::MuonUpdate::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::muon::MuonUpdate>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(n, &program), (
        params_buf.as_device_ptr() as *mut f32,
        grad_buf.as_device_ptr() as *mut f32,
        n as i32,
        LR,
    ))?;

    let mut out = vec![0.0f32; n];
    params_buf.to_host(&mut out)?;
    for i in 0..n {
        assert!((out[i] - expected[i]).abs() < 1e-5,
            "muon_update at {i}: got={} expected={}", out[i], expected[i]);
    }
    Ok(())
}

/// Verify muon_frob_norm_sq: out = sum(x²).
#[test]
#[cfg(feature = "cuda")]
fn test_muon_frob_norm_sq_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let n = M * N;

    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - (n as f32) * 0.005).collect();
    let expected: f32 = x.iter().map(|v| v * v).sum();

    let mut x_buf   = env.device.buffer::<f32>(n)?;
    let mut out_buf = env.device.buffer::<f32>(1)?;
    x_buf.to_device(&x)?;
    out_buf.to_device(&[0.0f32])?;  // pre-zero the accumulator

    let kernel = teeny_kernels::nn::optim::muon::MuonFrobNormSq::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::muon::MuonFrobNormSq>(&ptx)?;
    env.device.launch(&program, &testing::launch_config_from_program(n, &program), (
        x_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        n as i32,
    ))?;

    let mut result = vec![0.0f32; 1];
    out_buf.to_host(&mut result)?;
    assert!((result[0] - expected).abs() / expected.abs() < 1e-4,
        "frob_norm_sq: got={} expected={}", result[0], expected);
    Ok(())
}

/// Verify muon_ns_xtx (!TRANSPOSE): T = X @ X.T.
#[test]
#[cfg(feature = "cuda")]
fn test_muon_ns_xtx_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    // X: [M, N], T_out: [M, M]
    let x: Vec<f32> = (0..M * N).map(|i| ((i as f32) * 0.03 - 1.0) * 0.1).collect();
    // CPU reference: T[i,j] = sum_k X[i,k]*X[j,k]
    let x_t: Vec<f32> = (0..N * M).map(|j_i| x[(j_i % M) * N + (j_i / M)]).collect();
    let expected = cpu_matmul(&x, &x_t, M, N, M);  // X @ X.T

    let mut x_buf = env.device.buffer::<f32>(M * N)?;
    let mut t_buf = env.device.buffer::<f32>(M * M)?;
    x_buf.to_device(&x)?;

    let kernel = teeny_kernels::nn::optim::muon::MuonNsXtx::new(false, BLOCK_R, BLOCK_K, GROUP_R);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::muon::MuonNsXtx>(&ptx)?;
    env.device.launch(
        &program,
        &testing::launch_config_with_grid(
            (M.div_ceil(BLOCK_R as usize)) * (M.div_ceil(BLOCK_R as usize)),
            &program,
        ),
        (
            x_buf.as_device_ptr() as *mut f32,
            t_buf.as_device_ptr() as *mut f32,
            M as i32,
            N as i32,
            N as i32,   // stride_xm
        ),
    )?;

    let mut t_out = vec![0.0f32; M * M];
    t_buf.to_host(&mut t_out)?;
    for i in 0..M * M {
        assert!((t_out[i] - expected[i]).abs() < 1e-3,
            "ns_xtx at {i}: got={} expected={}", t_out[i], expected[i]);
    }
    Ok(())
}

/// Verify muon_ns_step (!TRANSPOSE): X ← a·X + b·(T·X).
#[test]
#[cfg(feature = "cuda")]
fn test_muon_ns_step_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;

    // X: [M, N], T: [M, M]
    let x: Vec<f32> = (0..M * N).map(|i| ((i as f32) * 0.03 - 1.0) * 0.1).collect();
    let t: Vec<f32> = {
        let x_t: Vec<f32> = (0..N * M).map(|j_i| x[(j_i % M) * N + (j_i / M)]).collect();
        cpu_matmul(&x, &x_t, M, N, M)
    };
    // expected: a*X + b*(T@X)
    let tx = cpu_matmul(&t, &x, M, M, N);
    let expected: Vec<f32> = x.iter().zip(&tx).map(|(xi, txi)| A * xi + B * txi).collect();

    let mut x_buf = env.device.buffer::<f32>(M * N)?;
    let mut t_buf = env.device.buffer::<f32>(M * M)?;
    x_buf.to_device(&x)?;
    t_buf.to_device(&t)?;

    let kernel = teeny_kernels::nn::optim::muon::MuonNsStep::new(false, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::optim::muon::MuonNsStep>(&ptx)?;
    env.device.launch(
        &program,
        &testing::launch_config_with_grid(
            M.div_ceil(BLOCK_M as usize) * N.div_ceil(BLOCK_N as usize),
            &program,
        ),
        (
            t_buf.as_device_ptr() as *mut f32,
            x_buf.as_device_ptr() as *mut f32,
            M as i32,
            N as i32,
            M as i32,   // stride_tm (T is [M,M])
            N as i32,   // stride_xm (X is [M,N])
            A,
            B,
        ),
    )?;

    let mut x_out = vec![0.0f32; M * N];
    x_buf.to_host(&mut x_out)?;
    for i in 0..M * N {
        let tol = 1e-3_f32 * expected[i].abs().max(1.0);
        assert!((x_out[i] - expected[i]).abs() < tol,
            "ns_step at {i}: got={} expected={}", x_out[i], expected[i]);
    }
    Ok(())
}
