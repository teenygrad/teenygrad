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

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

const TOL: f32 = 1e-4;

// ── Source snapshot tests ─────────────────────────────────────────────────────

#[test]
fn test_elemwise_mul_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_binary::ElemwiseMulForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_mul_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_sub_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_binary::ElemwiseSubForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_sub_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_div_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_binary::ElemwiseDivForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_div_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_equal_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_binary::ElemwiseEqualForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_equal_forward_source", kernel.source());
    Ok(())
}

// ── CUDA execution tests ──────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_elemwise_mul_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let a = load_fixture("elemwise_binary/a.bin");
    let b = load_fixture("elemwise_binary/b.bin");
    let expected = load_fixture("elemwise_binary/expected_mul.bin");
    let n = a.len();

    let mut a_buf = device.buffer::<f32>(n)?;
    let mut b_buf = device.buffer::<f32>(n)?;
    let out_buf = device.buffer::<f32>(n)?;
    let mut out = vec![0.0f32; n];
    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;

    let kernel = teeny_kernels::nn::tensor::elemwise_binary::ElemwiseMulForward::<f32>::new(1024);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::elemwise_binary::ElemwiseMulForward<f32>,
    >(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = CudaLaunchConfig {
        grid: [(n as u32).div_ceil(1024), 1, 1],
        block: [1024, 1, 1],
        cluster: [1, 1, 1],
    };
    use teeny_core::device::Device;
    device.launch(&program, &cfg, (
        a_buf.as_device_ptr() as *mut f32,
        b_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        n as i32,
    ))?;

    use teeny_core::device::buffer::Buffer;
    out_buf.to_host(&mut out)?;
    for i in 0..n {
        assert!(
            (out[i] - expected[i]).abs() < TOL,
            "mul mismatch at i={i}: gpu={} expected={}",
            out[i], expected[i]
        );
    }
    Ok(())
}
