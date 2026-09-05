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

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use std::path::PathBuf;
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const B: usize = 2;
#[cfg(feature = "hardware")]
const C: usize = 4;
#[cfg(feature = "hardware")]
const L: usize = 8;
const PAD_LEFT: i32 = 2;
const PAD_RIGHT: i32 = 3;
#[cfg(feature = "hardware")]
const OL: usize = L + PAD_LEFT as usize + PAD_RIGHT as usize; // 13
const BLOCK_OL: i32 = 16;
#[cfg(feature = "hardware")]
const VALUE: f32 = 1.5;

#[cfg(feature = "hardware")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

#[test]
fn test_constant_pad1d_forward_mlir_output() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    dotenv().ok();
    let kernel = teeny_kernels::nn::pad::constant_pad1d::ConstantPad1dForward::<f32>::new(
        PAD_LEFT, PAD_RIGHT, BLOCK_OL,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "constant_pad1d_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "constant_pad1d_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_constant_pad1d_backward_mlir_output() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    dotenv().ok();
    let kernel = teeny_kernels::nn::pad::constant_pad1d::ConstantPad1dBackward::<f32>::new(
        PAD_LEFT, PAD_RIGHT, BLOCK_OL,
    );
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "constant_pad1d_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "constant_pad1d_backward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_constant_pad1d_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let input_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "constant_pad1d/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "constant_pad1d/expected_forward.bin",
    );
    let mut output_host = vec![0.0f32; B * C * OL];

    let mut input_buf = device.buffer::<f32>(B * C * L)?;
    let output_buf = device.buffer::<f32>(B * C * OL)?;
    input_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::nn::pad::constant_pad1d::ConstantPad1dForward::<f32>::new(
        PAD_LEFT, PAD_RIGHT, BLOCK_OL,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::pad::constant_pad1d::ConstantPad1dForward<f32>,
    >(&ptx_path)?;

    let num_ol_tiles = OL.div_ceil(BLOCK_OL as usize);
    let grid_x = (B * C * num_ol_tiles) as u32;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_x, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );
    let args = (
        input_buf.as_device_ptr(),
        output_buf.as_device_ptr(),
        B as i32,
        C as i32,
        L as i32,
        OL as i32,
        VALUE,
    );
    device.launch(&program, &cfg, args)?;
    output_buf.to_host(&mut output_host)?;

    for i in 0..(B * C * OL) {
        assert!(
            (output_host[i] - expected[i]).abs() < 1e-5,
            "constant_pad1d_forward mismatch at {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_constant_pad1d_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "constant_pad1d/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "constant_pad1d/expected_backward.bin",
    );
    let zeros = vec![0.0f32; B * C * L];
    let mut dx_host = vec![0.0f32; B * C * L];

    let mut dy_buf = device.buffer::<f32>(B * C * OL)?;
    let mut dx_buf = device.buffer::<f32>(B * C * L)?;
    dy_buf.to_device(&dy_host)?;
    dx_buf.to_device(&zeros)?;

    let kernel = teeny_kernels::nn::pad::constant_pad1d::ConstantPad1dBackward::<f32>::new(
        PAD_LEFT, PAD_RIGHT, BLOCK_OL,
    );
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::pad::constant_pad1d::ConstantPad1dBackward<f32>,
    >(&ptx_path)?;

    let num_ol_tiles = OL.div_ceil(BLOCK_OL as usize);
    let grid_x = (B * C * num_ol_tiles) as u32;
    let cfg = teeny_runtime::launch_config_custom(
        [grid_x, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );
    let args = (
        dy_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        B as i32,
        C as i32,
        L as i32,
        OL as i32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C * L) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "constant_pad1d_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
