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
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{
    compiler::target::Capability, errors::Result, testing, device::CudaLaunchConfig,
};

const B: usize = 2;
const C: usize = 4;
const H: usize = 6;
const W: usize = 8;
const PT: i32 = 1;
const PB: i32 = 2;
const PL: i32 = 2;
const PR: i32 = 3;
const OH: usize = H + PT as usize + PB as usize; // 9
const OW: usize = W + PL as usize + PR as usize; // 13
const BLOCK_OW: i32 = 16;

const PTX_LAUNCH_THREADS_X: u32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn test_circular_pad2d_forward_mlir_output() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    dotenv()?;
    let kernel = teeny_kernels::pad::circular_pad2d::CircularPad2dForward::<f32>::new(PT, PB, PL, PR, BLOCK_OW);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("circular_pad2d_forward_source", kernel.source());
    assert_debug_snapshot!("circular_pad2d_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_circular_pad2d_backward_mlir_output() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    dotenv()?;
    let kernel = teeny_kernels::pad::circular_pad2d::CircularPad2dBackward::<f32>::new(PT, PB, PL, PR, BLOCK_OW);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("circular_pad2d_backward_source", kernel.source());
    assert_debug_snapshot!("circular_pad2d_backward_mlir", mlir.trim());
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_circular_pad2d_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_host = load_fixture("circular_pad2d/x.bin");
    let expected = load_fixture("circular_pad2d/expected_forward.bin");
    let mut output_host = vec![0.0f32; B * C * OH * OW];

    let mut input_buf = device.buffer::<f32>(B * C * H * W)?;
    let output_buf = device.buffer::<f32>(B * C * OH * OW)?;
    input_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::pad::circular_pad2d::CircularPad2dForward::<f32>::new(PT, PB, PL, PR, BLOCK_OW);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::pad::circular_pad2d::CircularPad2dForward<f32>,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_x = (B * C * OH * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    let args = (
        input_buf.as_device_ptr() as *mut f32,
        output_buf.as_device_ptr() as *mut f32,
        B as i32,
        C as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );
    device.launch(&program, &cfg, args)?;
    output_buf.to_host(&mut output_host)?;

    for i in 0..(B * C * OH * OW) {
        assert!(
            (output_host[i] - expected[i]).abs() < 1e-5,
            "circular_pad2d_forward mismatch at {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_circular_pad2d_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host = load_fixture("circular_pad2d/dy.bin");
    let expected = load_fixture("circular_pad2d/expected_backward.bin");
    let zeros = vec![0.0f32; B * C * H * W];
    let mut dx_host = vec![0.0f32; B * C * H * W];

    let mut dy_buf = device.buffer::<f32>(B * C * OH * OW)?;
    let mut dx_buf = device.buffer::<f32>(B * C * H * W)?;
    dy_buf.to_device(&dy_host)?;
    dx_buf.to_device(&zeros)?;

    let kernel = teeny_kernels::pad::circular_pad2d::CircularPad2dBackward::<f32>::new(PT, PB, PL, PR, BLOCK_OW);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::pad::circular_pad2d::CircularPad2dBackward<f32>,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_x = (B * C * OH * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        B as i32,
        C as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );
    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C * H * W) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "circular_pad2d_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
