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
use ndarray::Array4;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::PathBuf;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

// Spatial dimensions: 2×2 non-overlapping pool on a 8×8 feature map → 4×4 output.
const B: usize = 2;
const C: usize = 4;
const H: usize = 8;
const W: usize = 8;
const KH: i32 = 2;
const KW: i32 = 2;
const STRIDE_H: i32 = 2;
const STRIDE_W: i32 = 2;
const OH: usize = (H - KH as usize) / STRIDE_H as usize + 1; // 4
const OW: usize = (W - KW as usize) / STRIDE_W as usize + 1; // 4
const BLOCK_OW: i32 = 4;

/// Must match `.reqntid` in the generated PTX.
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_avgpool2d_forward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::pool::avgpool2d::Avgpool2dForward::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("avgpool2d_forward_source", kernel.source());
    assert_debug_snapshot!("avgpool2d_forward_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_avgpool2d_backward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::pool::avgpool2d::Avgpool2dBackward::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("avgpool2d_backward_source", kernel.source());
    assert_debug_snapshot!("avgpool2d_backward_mlir", mlir.trim());

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests
// ---------------------------------------------------------------------------

/// Forward: verify GPU output matches CPU reference for 2×2 avgpool with stride 2.
///
/// For each output (b, c, oh, ow):
///   output[b,c,oh,ow] = mean( input[b, c, oh*2:oh*2+2, ow*2:ow*2+2] )
#[test]
#[cfg(feature = "cuda")]
fn test_avgpool2d_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let input_arr = Array4::<f32>::random((B, C, H, W), Uniform::new(-5.0f32, 5.0f32).unwrap());
    let input_host: Vec<f32> = input_arr.iter().copied().collect();

    // CPU reference.
    let mut expected = vec![0.0f32; B * C * OH * OW];
    for b in 0..B {
        for c in 0..C {
            for oh in 0..OH {
                for ow in 0..OW {
                    let mut sum = 0.0f32;
                    for kh in 0..KH as usize {
                        for kw in 0..KW as usize {
                            let ih = oh * STRIDE_H as usize + kh;
                            let iw = ow * STRIDE_W as usize + kw;
                            sum += input_arr[[b, c, ih, iw]];
                        }
                    }
                    expected[b * C * OH * OW + c * OH * OW + oh * OW + ow] = sum / (KH * KW) as f32;
                }
            }
        }
    }

    let mut input_buf = device.buffer::<f32>(B * C * H * W)?;
    let output_buf = device.buffer::<f32>(B * C * OH * OW)?;
    let mut output_host = vec![0.0f32; B * C * OH * OW];

    input_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::pool::avgpool2d::Avgpool2dForward::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[avgpool2d_forward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::pool::avgpool2d::Avgpool2dForward<f32, KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW>,
    >(&ptx)?;

    // Grid: B * C * OH * ceil(OW / BLOCK_OW).
    let num_ow_tiles = (OW as u32).div_ceil(BLOCK_OW as u32);
    let grid_x = (B * C * OH) as u32 * num_ow_tiles;
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
            (output_host[i] - expected[i]).abs() < 1e-4,
            "avgpool2d_forward mismatch at index {i}: gpu={}, expected={}",
            output_host[i],
            expected[i]
        );
    }

    Ok(())
}

/// Backward: verify GPU dx matches CPU reference for 2×2 avgpool with stride 2.
///
/// Since STRIDE = KH = KW = 2 (non-overlapping), each input element maps to
/// exactly one output, so:
///   dx[b,c,ih,iw] = dy[b,c,ih/2,iw/2] / (KH*KW)
#[test]
#[cfg(feature = "cuda")]
fn test_avgpool2d_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_arr = Array4::<f32>::random((B, C, OH, OW), Uniform::new(-2.0f32, 2.0f32).unwrap());
    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();

    // CPU reference: uniform scatter.
    let mut expected = vec![0.0f32; B * C * H * W];
    for b in 0..B {
        for c in 0..C {
            for oh in 0..OH {
                for ow in 0..OW {
                    let grad = dy_arr[[b, c, oh, ow]] / (KH * KW) as f32;
                    for kh in 0..KH as usize {
                        for kw in 0..KW as usize {
                            let ih = oh * STRIDE_H as usize + kh;
                            let iw = ow * STRIDE_W as usize + kw;
                            expected[b * C * H * W + c * H * W + ih * W + iw] += grad;
                        }
                    }
                }
            }
        }
    }

    let mut dy_buf = device.buffer::<f32>(B * C * OH * OW)?;
    let mut dx_host = vec![0.0f32; B * C * H * W];
    let zeros = vec![0.0f32; B * C * H * W];

    dy_buf.to_device(&dy_host)?;
    // dx must be zero-initialised before launching (atomics accumulate).
    let mut dx_zero_buf = device.buffer::<f32>(B * C * H * W)?;
    dx_zero_buf.to_device(&zeros)?;

    let kernel = teeny_kernels::pool::avgpool2d::Avgpool2dBackward::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[avgpool2d_backward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::pool::avgpool2d::Avgpool2dBackward<
            f32,
            KH,
            KW,
            STRIDE_H,
            STRIDE_W,
            BLOCK_OW,
        >,
    >(&ptx)?;

    let num_ow_tiles = (OW as u32).div_ceil(BLOCK_OW as u32);
    let grid_x = (B * C * OH) as u32 * num_ow_tiles;
    let cfg = CudaLaunchConfig {
        grid: [grid_x, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        dx_zero_buf.as_device_ptr() as *mut f32,
        B as i32,
        C as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_zero_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C * H * W) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "avgpool2d_backward mismatch at index {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}
