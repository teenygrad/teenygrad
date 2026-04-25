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

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── MLIR snapshots ────────────────────────────────────────────────────────────

#[test]
fn test_gelu_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::activation::gelu::GeluForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("gelu_forward_source", kernel.source());
    assert_debug_snapshot!("gelu_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_mish_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::activation::gelu::MishForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("mish_forward_source", kernel.source());
    assert_debug_snapshot!("mish_forward_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: GELU ────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_gelu_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("gelu/x.bin");
    let expected = load_fixture("gelu/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::gelu::GeluForward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::gelu::GeluForward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32))?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 2e-5,
            "gelu_forward mismatch at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_gelu_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("gelu/x.bin");
    let dy_host  = load_fixture("gelu/dy.bin");
    let expected = load_fixture("gelu/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::gelu::GeluBackward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::gelu::GeluBackward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 2e-5,
            "gelu_backward mismatch at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Mish ────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_mish_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("mish/x.bin");
    let expected = load_fixture("mish/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::gelu::MishForward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::gelu::MishForward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32))?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 2e-5,
            "mish_forward mismatch at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_mish_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("mish/x.bin");
    let dy_host  = load_fixture("mish/dy.bin");
    let expected = load_fixture("mish/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::gelu::MishBackward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::gelu::MishBackward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 2e-5,
            "mish_backward mismatch at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}
