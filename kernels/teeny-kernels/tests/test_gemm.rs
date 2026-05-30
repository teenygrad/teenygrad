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
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};

const TOL: f32 = 1e-3;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── Source snapshot tests ─────────────────────────────────────────────────────

#[test]
fn test_matmul_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::math::gemm::MatmulForward::<f32>::new(128);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("matmul_forward_source", kernel.source());
    Ok(())
}

// ── CUDA execution tests ──────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // Simple 4x4 matmul: A[4,4] @ B[4,4] = C[4,4]
    let m = 4usize;
    let n = 4usize;
    let k = 4usize;

    let a: Vec<f32> = (0..m*k).map(|i| i as f32 / (m*k) as f32).collect();
    let b: Vec<f32> = (0..k*n).map(|i| i as f32 / (k*n) as f32).collect();

    // Expected: compute reference on CPU
    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0;
            for ki in 0..k {
                sum += a[mi * k + ki] * b[ki * n + ni];
            }
            expected[mi * n + ni] = sum;
        }
    }

    let mut a_buf = device.buffer::<f32>(m*k)?;
    let mut b_buf = device.buffer::<f32>(k*n)?;
    let c_buf = device.buffer::<f32>(m*n)?;
    let mut c_out = vec![0.0f32; m*n];

    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;

    let kernel = teeny_kernels::math::gemm::MatmulForward::<f32>::new(128);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::math::gemm::MatmulForward<f32>,
    >(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = CudaLaunchConfig {
        grid: [(m * n) as u32, 1, 1],
        block: [128, 1, 1],
        cluster: [1, 1, 1],
    };
    use teeny_core::device::Device;
    device.launch(&program, &cfg, (
        a_buf.as_device_ptr() as *mut f32,
        b_buf.as_device_ptr() as *mut f32,
        c_buf.as_device_ptr() as *mut f32,
        m as i32,
        n as i32,
        k as i32,
    ))?;

    use teeny_core::device::buffer::Buffer;
    c_buf.to_host(&mut c_out)?;
    for i in 0..m*n {
        assert!(
            (c_out[i] - expected[i]).abs() < TOL,
            "matmul mismatch at i={i}: gpu={} expected={}",
            c_out[i], expected[i]
        );
    }
    Ok(())
}
