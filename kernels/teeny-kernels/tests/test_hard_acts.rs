/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

// ── MLIR snapshots ────────────────────────────────────────────────────────────

macro_rules! mlir_snap {
    ($test:ident, $KernelTy:ty, $src_name:expr, $mlir_name:expr) => {
        #[test]
        fn $test() -> anyhow::Result<()> {
            dotenv().ok();
            let kernel = <$KernelTy>::new(BLOCK_SIZE);
            let target = teeny_runtime::reference_target();
            let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
                &kernel, &target, true, false,
            )?);
            let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
            assert_debug_snapshot!(
                format!("{}_{}", $src_name, teeny_runtime::BACKEND_NAME),
                kernel.source()
            );
            assert_debug_snapshot!(
                format!("{}_{}", $mlir_name, teeny_runtime::BACKEND_NAME),
                mlir.trim()
            );
            Ok(())
        }
    };
}

mlir_snap!(
    test_hardtanh_mlir,
    teeny_kernels::nn::activation::hard::HardtanhForward<f32>,
    "hardtanh_forward_source",
    "hardtanh_forward_mlir"
);
mlir_snap!(
    test_relu6_mlir,
    teeny_kernels::nn::activation::hard::Relu6Forward<f32>,
    "relu6_forward_source",
    "relu6_forward_mlir"
);
mlir_snap!(
    test_hardsigmoid_mlir,
    teeny_kernels::nn::activation::hard::HardsigmoidForward<f32>,
    "hardsigmoid_forward_source",
    "hardsigmoid_forward_mlir"
);
mlir_snap!(
    test_hardswish_mlir,
    teeny_kernels::nn::activation::hard::HardswishForward<f32>,
    "hardswish_forward_source",
    "hardswish_forward_mlir"
);
mlir_snap!(
    test_hardshrink_mlir,
    teeny_kernels::nn::activation::hard::HardshrinkForward<f32>,
    "hardshrink_forward_source",
    "hardshrink_forward_mlir"
);

// ── CUDA: Hardtanh ────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_hardtanh_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardtanh/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardtanh/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardtanhForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardtanhForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N as i32,
            -1.0_f32,
            1.0_f32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "hardtanh_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_hardtanh_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardtanh/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardtanh/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardtanh/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardtanhBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardtanhBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
            -1.0_f32,
            1.0_f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "hardtanh_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: ReLU6 ───────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_relu6_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu6/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu6/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::Relu6Forward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::Relu6Forward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "relu6_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_relu6_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu6/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu6/dy.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu6/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::Relu6Backward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::Relu6Backward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "relu6_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Hardsigmoid ─────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_hardsigmoid_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardsigmoid/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "hardsigmoid/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardsigmoidForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardsigmoidForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "hardsigmoid_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_hardsigmoid_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardsigmoid/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardsigmoid/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "hardsigmoid/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardsigmoidBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardsigmoidBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "hardsigmoid_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Hardswish ───────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_hardswish_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardswish/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardswish/expected_forward.bin");
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardswishForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardswishForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), N as i32),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "hardswish_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_hardswish_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardswish/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardswish/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "hardswish/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardswishBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardswishBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "hardswish_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

// ── CUDA: Hardshrink ──────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_hardshrink_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardshrink/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "hardshrink/expected_forward.bin",
    );
    let mut y_host = vec![0.0f32; N];

    let mut x_buf = device.buffer::<f32>(N)?;
    let y_buf = device.buffer::<f32>(N)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardshrinkForward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardshrinkForward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            N as i32,
            0.5_f32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;
    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-5,
            "hardshrink_forward at {i}: got={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_hardshrink_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardshrink/x.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "hardshrink/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "hardshrink/expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x_buf.to_device(&x_host)?;

    let kernel = teeny_kernels::nn::activation::hard::HardshrinkBackward::<f32>::new(BLOCK_SIZE);
    let ptx_path = teeny_runtime::compile_kernel(
        &kernel,
        &teeny_runtime::default_target(&device)?,
        true,
        false,
    )?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::hard::HardshrinkBackward<f32>,
    >(&ptx_path)?;
    device.launch(
        &program,
        &teeny_runtime::launch_config(N, &program),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            N as i32,
            0.5_f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_host)?;
    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "hardshrink_backward at {i}: got={} expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
