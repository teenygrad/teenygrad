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

macro_rules! mlir_snap {
    ($test:ident, $KernelTy:ty, $src_name:expr, $mlir_name:expr) => {
        #[test]
        fn $test() -> anyhow::Result<()> {
            dotenv()?;
            let kernel = <$KernelTy>::new(BLOCK_SIZE);
            let target = Target::new(Capability::Sm90);
            let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
            let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
            assert_debug_snapshot!($src_name,  kernel.source());
            assert_debug_snapshot!($mlir_name, mlir.trim());
            Ok(())
        }
    };
}

mlir_snap!(test_hardtanh_mlir,    teeny_kernels::activation::hard::HardtanhForward,    "hardtanh_forward_source",    "hardtanh_forward_mlir");
mlir_snap!(test_relu6_mlir,       teeny_kernels::activation::hard::Relu6Forward,       "relu6_forward_source",       "relu6_forward_mlir");
mlir_snap!(test_hardsigmoid_mlir, teeny_kernels::activation::hard::HardsigmoidForward, "hardsigmoid_forward_source", "hardsigmoid_forward_mlir");
mlir_snap!(test_hardswish_mlir,   teeny_kernels::activation::hard::HardswishForward,   "hardswish_forward_source",   "hardswish_forward_mlir");
mlir_snap!(test_hardshrink_mlir,  teeny_kernels::activation::hard::HardshrinkForward,  "hardshrink_forward_source",  "hardshrink_forward_mlir");

// ── CUDA: Hardtanh ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_hardtanh_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardtanh/x.bin");
    let expected = load_fixture("hardtanh/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardtanhForward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardtanhForward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32,
        N as i32, -1.0_f32, 1.0_f32,
    ))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "hardtanh_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hardtanh_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardtanh/x.bin");
    let dy_host  = load_fixture("hardtanh/dy.bin");
    let expected = load_fixture("hardtanh/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardtanhBackward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardtanhBackward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32, -1.0_f32, 1.0_f32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "hardtanh_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: ReLU6 ───────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_relu6_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("relu6/x.bin");
    let expected = load_fixture("relu6/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::Relu6Forward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::Relu6Forward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "relu6_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_relu6_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("relu6/x.bin");
    let dy_host  = load_fixture("relu6/dy.bin");
    let expected = load_fixture("relu6/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::Relu6Backward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::Relu6Backward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "relu6_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Hardsigmoid ─────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_hardsigmoid_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardsigmoid/x.bin");
    let expected = load_fixture("hardsigmoid/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardsigmoidForward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardsigmoidForward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "hardsigmoid_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hardsigmoid_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardsigmoid/x.bin");
    let dy_host  = load_fixture("hardsigmoid/dy.bin");
    let expected = load_fixture("hardsigmoid/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardsigmoidBackward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardsigmoidBackward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "hardsigmoid_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Hardswish ───────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_hardswish_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardswish/x.bin");
    let expected = load_fixture("hardswish/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardswishForward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardswishForward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "hardswish_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hardswish_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardswish/x.bin");
    let dy_host  = load_fixture("hardswish/dy.bin");
    let expected = load_fixture("hardswish/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardswishBackward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardswishBackward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "hardswish_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}

// ── CUDA: Hardshrink ──────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_hardshrink_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardshrink/x.bin");
    let expected = load_fixture("hardshrink/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = env.device.buffer::<f32>(N)?;
    let y_buf = env.device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardshrinkForward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardshrinkForward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE),
        (x_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32, N as i32, 0.5_f32))?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!((y_host[i] - expected[i]).abs() < 1e-5,
            "hardshrink_forward at {i}: got={} expected={}", y_host[i], expected[i]);
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hardshrink_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let x_host   = load_fixture("hardshrink/x.bin");
    let dy_host  = load_fixture("hardshrink/dy.bin");
    let expected = load_fixture("hardshrink/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = env.device.buffer::<f32>(N)?;
    let mut x_buf  = env.device.buffer::<f32>(N)?;
    let dx_buf     = env.device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::activation::hard::HardshrinkBackward::new(BLOCK_SIZE);
    let ptx = std::fs::read(compile_kernel(&kernel, &Target::new(env.capability), true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::activation::hard::HardshrinkBackward
    >(&ptx)?;
    env.device.launch(&program, &testing::launch_config(N, BLOCK_SIZE), (
        dy_buf.as_device_ptr() as *mut f32, x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32, N as i32, 0.5_f32,
    ))?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!((dx_host[i] - expected[i]).abs() < 1e-5,
            "hardshrink_backward at {i}: got={} expected={}", dx_host[i], expected[i]);
    }
    Ok(())
}
