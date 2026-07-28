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
use teeny_core::device::Device;
#[cfg(feature = "cuda")]
use teeny_core::device::buffer::Buffer;
#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};

use teeny_kernels::math::gemm::{MatmulBackwardDa, MatmulBackwardDb, MatmulForward};

const BLOCK_K: i32 = 128;
const TOL: f32 = 1e-3;

// ── Source + MLIR snapshots ───────────────────────────────────────────────────

#[test]
fn test_matmul_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = MatmulForward::<f32>::new(BLOCK_K);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("matmul_forward_source", kernel.source());
    assert_debug_snapshot!("matmul_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_matmul_backward_da_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = MatmulBackwardDa::<f32>::new(BLOCK_K);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("matmul_backward_da_source", kernel.source());
    assert_debug_snapshot!("matmul_backward_da_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_matmul_backward_db_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = MatmulBackwardDb::<f32>::new(BLOCK_K);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("matmul_backward_db_source", kernel.source());
    assert_debug_snapshot!("matmul_backward_db_mlir", mlir.trim());
    Ok(())
}

// ── GPU forward test ──────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // 4x4 @ 4x4 = 4x4 with inline CPU reference
    let m = 4usize;
    let n = 4usize;
    let k = 4usize;

    let a: Vec<f32> = (0..m * k).map(|i| i as f32 / (m * k) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 / (k * n) as f32).collect();

    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[mi * k + ki] * b[ki * n + ni];
            }
            expected[mi * n + ni] = sum;
        }
    }

    let mut a_buf = device.buffer::<f32>(m * k)?;
    let mut b_buf = device.buffer::<f32>(k * n)?;
    let c_buf = device.buffer::<f32>(m * n)?;
    let mut c_out = vec![0.0f32; m * n];

    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;

    let kernel = MatmulForward::<f32>::new(BLOCK_K);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<MatmulForward<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = CudaLaunchConfig {
        grid: [(m * n) as u32, 1, 1],
        block: [BLOCK_K as u32, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            a_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            c_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    c_buf.to_host(&mut c_out)?;
    for i in 0..m * n {
        assert!(
            (c_out[i] - expected[i]).abs() < TOL,
            "matmul fwd mismatch at i={i}: gpu={} expected={}",
            c_out[i],
            expected[i]
        );
    }
    Ok(())
}

// ── GPU backward: dA ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_backward_da_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // dA[M,K] = dC[M,N] @ B^T[N,K]  =>  dA[m,k] = sum_n dC[m,n] * B[k,n]
    let m = 4usize;
    let n = 4usize;
    let k = 4usize;

    let b: Vec<f32> = (0..k * n).map(|i| i as f32 / (k * n) as f32).collect();
    let dc: Vec<f32> = vec![1.0f32; m * n];

    let mut expected_da = vec![0.0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut sum = 0.0f32;
            for ni in 0..n {
                sum += dc[mi * n + ni] * b[ki * n + ni];
            }
            expected_da[mi * k + ki] = sum;
        }
    }

    let mut dc_buf = device.buffer::<f32>(m * n)?;
    let mut b_buf = device.buffer::<f32>(k * n)?;
    let da_buf = device.buffer::<f32>(m * k)?;
    let mut da_out = vec![0.0f32; m * k];

    dc_buf.to_device(&dc)?;
    b_buf.to_device(&b)?;

    let kernel = MatmulBackwardDa::<f32>::new(BLOCK_K);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<MatmulBackwardDa<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = CudaLaunchConfig {
        grid: [(m * k) as u32, 1, 1],
        block: [BLOCK_K as u32, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            dc_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            da_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    da_buf.to_host(&mut da_out)?;
    for i in 0..m * k {
        assert!(
            (da_out[i] - expected_da[i]).abs() < TOL,
            "matmul bwd_da mismatch at i={i}: gpu={} expected={}",
            da_out[i],
            expected_da[i]
        );
    }
    Ok(())
}

// ── GPU backward: dB ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_backward_db_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // dB[K,N] = A^T[K,M] @ dC[M,N]  =>  dB[k,n] = sum_m A[m,k] * dC[m,n]
    let m = 4usize;
    let n = 4usize;
    let k = 4usize;

    let a: Vec<f32> = (0..m * k).map(|i| i as f32 / (m * k) as f32).collect();
    let dc: Vec<f32> = vec![1.0f32; m * n];

    let mut expected_db = vec![0.0f32; k * n];
    for ki in 0..k {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for mi in 0..m {
                sum += a[mi * k + ki] * dc[mi * n + ni];
            }
            expected_db[ki * n + ni] = sum;
        }
    }

    let mut dc_buf = device.buffer::<f32>(m * n)?;
    let mut a_buf = device.buffer::<f32>(m * k)?;
    let db_buf = device.buffer::<f32>(k * n)?;
    let mut db_out = vec![0.0f32; k * n];

    dc_buf.to_device(&dc)?;
    a_buf.to_device(&a)?;

    let kernel = MatmulBackwardDb::<f32>::new(BLOCK_K);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<MatmulBackwardDb<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = CudaLaunchConfig {
        grid: [(k * n) as u32, 1, 1],
        block: [BLOCK_K as u32, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            dc_buf.as_device_ptr() as *mut f32,
            a_buf.as_device_ptr() as *mut f32,
            db_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    db_buf.to_host(&mut db_out)?;
    for i in 0..k * n {
        assert!(
            (db_out[i] - expected_db[i]).abs() < TOL,
            "matmul bwd_db mismatch at i={i}: gpu={} expected={}",
            db_out[i],
            expected_db[i]
        );
    }
    Ok(())
}
