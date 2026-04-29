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
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

const M: usize = 16;
const N: usize = 128;
const EPS: f32 = 1e-5;
const BLOCK_N: i32 = 256;
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
// Source snapshot tests (no CUDA required)
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_inference_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability as Cap;
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForwardInference::<f32>::new(BLOCK_N);
    let target = Target::new(Cap::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("layer_norm_inference_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_layer_norm_forward_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability as Cap;
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForward::<f32>::new(BLOCK_N);
    let target = Target::new(Cap::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("layer_norm_forward_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_layer_norm_backward_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability as Cap;
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormBackward::<f32>::new(BLOCK_N);
    let target = Target::new(Cap::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("layer_norm_backward_source", kernel.source());
    Ok(())
}

// ---------------------------------------------------------------------------
// MLIR snapshot tests (compile to MLIR, no GPU required)
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_inference_mlir() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability as Cap;
    let kernel = teeny_kernels::nn::norm::layernorm::LayerNormForwardInference::<f32>::new(BLOCK_N);
    let target = Target::new(Cap::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("layer_norm_inference_mlir", mlir.trim());
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests (requires GPU + fixtures from generate.py)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "cuda")]
fn test_layer_norm_inference_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("layernorm/x.bin");
    let weight_host = load_fixture("layernorm/weight.bin");
    let bias_host = load_fixture("layernorm/bias.bin");
    let expected = load_fixture("layernorm/expected_forward.bin");
    let mut y_host = vec![0.0f32; M * N];

    let mut x_buf = device.buffer::<f32>(M * N)?;
    let mut w_buf = device.buffer::<f32>(N)?;
    let mut b_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(M * N)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&weight_host)?;
    b_buf.to_device(&bias_host)?;

    let kernel =
        teeny_kernels::nn::norm::layernorm::LayerNormForwardInference::<f32>::new(BLOCK_N);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::norm::layernorm::LayerNormForwardInference<f32>,
    >(&ptx)?;

    let cfg = CudaLaunchConfig { grid: [M as u32, 1, 1], block: [PTX_LAUNCH_THREADS_X, 1, 1], cluster: [1, 1, 1] };
    device.launch(&program, &cfg, (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        b_buf.as_device_ptr() as *mut f32,
        M as i32,
        N as i32,
        EPS,
    ))?;

    y_buf.to_host(&mut y_host)?;
    for i in 0..M * N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "layer_norm_inference mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}
