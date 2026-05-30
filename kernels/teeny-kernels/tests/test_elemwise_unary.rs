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

// ── Source snapshot tests (no CUDA required) ──────────────────────────────────

#[test]
fn test_elemwise_abs_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseAbsForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_abs_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_neg_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseNegForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_neg_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_sqrt_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseSqrtForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_sqrt_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_exp_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseExpForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_exp_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_log_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseLogForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_log_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_sin_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseSinForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_sin_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_cos_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseCosForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_cos_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_elemwise_atan_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseAtanForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("elemwise_atan_forward_source", kernel.source());
    Ok(())
}

// ── CUDA execution tests ──────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_elemwise_abs_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x = load_fixture("elemwise_unary/x.bin");
    let expected = load_fixture("elemwise_unary/expected_abs.bin");
    let n = x.len();

    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;

    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseAbsForward::<f32>::new(1024);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::elemwise_unary::ElemwiseAbsForward<f32>,
    >(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = CudaLaunchConfig {
        grid: [(n as u32).div_ceil(1024), 1, 1],
        block: [1024, 1, 1],
        cluster: [1, 1, 1],
    };
    use teeny_core::device::Device;
    device.launch(&program, &cfg, (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        n as i32,
    ))?;

    use teeny_core::device::buffer::Buffer;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "abs mismatch at i={i}: gpu={} expected={}",
            y_out[i], expected[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_elemwise_neg_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x = load_fixture("elemwise_unary/x.bin");
    let expected = load_fixture("elemwise_unary/expected_neg.bin");
    let n = x.len();

    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;

    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseNegForward::<f32>::new(1024);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::elemwise_unary::ElemwiseNegForward<f32>,
    >(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(n as u32).div_ceil(1024), 1, 1],
        block: [1024, 1, 1],
        cluster: [1, 1, 1],
    };
    use teeny_core::device::Device;
    device.launch(&program, &cfg, (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        n as i32,
    ))?;

    use teeny_core::device::buffer::Buffer;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "neg mismatch at i={i}: gpu={} expected={}",
            y_out[i], expected[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_elemwise_sqrt_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x = load_fixture("elemwise_unary/x.bin");
    let expected = load_fixture("elemwise_unary/expected_sqrt.bin");
    let n = x.len();

    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;

    let kernel = teeny_kernels::nn::tensor::elemwise_unary::ElemwiseSqrtForward::<f32>::new(1024);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::tensor::elemwise_unary::ElemwiseSqrtForward<f32>,
    >(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let cfg = teeny_cuda::device::CudaLaunchConfig {
        grid: [(n as u32).div_ceil(1024), 1, 1],
        block: [1024, 1, 1],
        cluster: [1, 1, 1],
    };
    use teeny_core::device::Device;
    device.launch(&program, &cfg, (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        n as i32,
    ))?;

    use teeny_core::device::buffer::Buffer;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "sqrt mismatch at i={i}: gpu={} expected={}",
            y_out[i], expected[i]
        );
    }
    Ok(())
}
