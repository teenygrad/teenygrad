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
fn test_leaky_relu_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::activation::misc::LeakyReluForward::<BLOCK_SIZE>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("leaky_relu_forward_source", kernel.source());
    assert_debug_snapshot!("leaky_relu_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_softsign_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::activation::misc::SoftsignForward::<BLOCK_SIZE>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("softsign_forward_source", kernel.source());
    assert_debug_snapshot!("softsign_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_softplus_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::activation::misc::SoftplusForward::<BLOCK_SIZE>::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("softplus_forward_source", kernel.source());
    assert_debug_snapshot!("softplus_forward_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA: LeakyReLU ───────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_leaky_relu_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("leaky_relu/x.bin");
    let expected = load_fixture("leaky_relu/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::LeakyReluForward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::LeakyReluForward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32, 0.01_f32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "leaky_relu_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_leaky_relu_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("leaky_relu/x.bin");
    let dy_host  = load_fixture("leaky_relu/dy.bin");
    let expected = load_fixture("leaky_relu/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::LeakyReluBackward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::LeakyReluBackward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32, 0.01_f32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "leaky_relu_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Threshold ───────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_threshold_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("threshold/x.bin");
    let expected = load_fixture("threshold/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::ThresholdForward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::ThresholdForward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32, 0.5_f32, 0.0_f32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "threshold_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_threshold_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("threshold/x.bin");
    let dy_host  = load_fixture("threshold/dy.bin");
    let expected = load_fixture("threshold/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::ThresholdBackward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::ThresholdBackward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32, 0.5_f32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "threshold_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Softsign ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_softsign_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("softsign/x.bin");
    let expected = load_fixture("softsign/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::SoftsignForward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::SoftsignForward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "softsign_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_softsign_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("softsign/x.bin");
    let dy_host  = load_fixture("softsign/dy.bin");
    let expected = load_fixture("softsign/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::SoftsignBackward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::SoftsignBackward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "softsign_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Softshrink ──────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_softshrink_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("softshrink/x.bin");
    let expected = load_fixture("softshrink/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::SoftshrinkForward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::SoftshrinkForward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32, 0.5_f32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "softshrink_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_softshrink_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("softshrink/x.bin");
    let dy_host  = load_fixture("softshrink/dy.bin");
    let expected = load_fixture("softshrink/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::SoftshrinkBackward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::SoftshrinkBackward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32, 0.5_f32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "softshrink_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Softplus ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_softplus_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("softplus/x.bin");
    let expected = load_fixture("softplus/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::SoftplusForward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::SoftplusForward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32, 1.0_f32, 20.0_f32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "softplus_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_softplus_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("softplus/x.bin");
    let dy_host  = load_fixture("softplus/dy.bin");
    let expected = load_fixture("softplus/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::misc::SoftplusBackward::<BLOCK_SIZE>::new();
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::misc::SoftplusBackward<BLOCK_SIZE>
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32, 1.0_f32, 20.0_f32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "softplus_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}
