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

// Small dimensions for fast tests.
const B: usize = 1;
const C_IN: usize = 2;
const C_OUT: usize = 4;
const H: usize = 8;
const W: usize = 8;
const KH: i32 = 3;
const KW: i32 = 3;
const STRIDE_H: i32 = 1;
const STRIDE_W: i32 = 1;
const OH: usize = (H - KH as usize) / STRIDE_H as usize + 1; // 6
const OW: usize = (W - KW as usize) / STRIDE_W as usize + 1; // 6
const BLOCK_OW: i32 = 8;

const PTX_LAUNCH_THREADS_X: u32 = 128;

// ---------------------------------------------------------------------------
// CPU reference helpers
// ---------------------------------------------------------------------------

fn cpu_conv2d_forward(x: &Array4<f32>, w: &ndarray::Array4<f32>) -> Array4<f32> {
    let mut y = Array4::<f32>::zeros([B, C_OUT, OH, OW]);
    for b in 0..B {
        for c_out in 0..C_OUT {
            for oh in 0..OH {
                for ow in 0..OW {
                    let mut acc = 0.0f32;
                    for c_in in 0..C_IN {
                        for kh in 0..KH as usize {
                            for kw in 0..KW as usize {
                                let ih = oh * STRIDE_H as usize + kh;
                                let iw = ow * STRIDE_W as usize + kw;
                                acc += x[[b, c_in, ih, iw]] * w[[c_out, c_in, kh, kw]];
                            }
                        }
                    }
                    y[[b, c_out, oh, ow]] = acc;
                }
            }
        }
    }
    y
}

fn cpu_conv2d_backward_dx(dy: &Array4<f32>, w: &ndarray::Array4<f32>) -> Array4<f32> {
    let mut dx = Array4::<f32>::zeros([B, C_IN, H, W]);
    for b in 0..B {
        for c_out in 0..C_OUT {
            for oh in 0..OH {
                for ow in 0..OW {
                    for c_in in 0..C_IN {
                        for kh in 0..KH as usize {
                            for kw in 0..KW as usize {
                                let ih = oh * STRIDE_H as usize + kh;
                                let iw = ow * STRIDE_W as usize + kw;
                                dx[[b, c_in, ih, iw]] +=
                                    dy[[b, c_out, oh, ow]] * w[[c_out, c_in, kh, kw]];
                            }
                        }
                    }
                }
            }
        }
    }
    dx
}

fn cpu_conv2d_backward_dw(dy: &Array4<f32>, x: &Array4<f32>) -> ndarray::Array4<f32> {
    let mut dw = ndarray::Array4::<f32>::zeros([C_OUT, C_IN, KH as usize, KW as usize]);
    for b in 0..B {
        for c_out in 0..C_OUT {
            for oh in 0..OH {
                for ow in 0..OW {
                    for c_in in 0..C_IN {
                        for kh in 0..KH as usize {
                            for kw in 0..KW as usize {
                                let ih = oh * STRIDE_H as usize + kh;
                                let iw = ow * STRIDE_W as usize + kw;
                                dw[[c_out, c_in, kh, kw]] +=
                                    dy[[b, c_out, oh, ow]] * x[[b, c_in, ih, iw]];
                            }
                        }
                    }
                }
            }
        }
    }
    dw
}

// ---------------------------------------------------------------------------
// MLIR snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_forward_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::conv::conv2d::Conv2dForward::<
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

    assert_debug_snapshot!("conv2d_forward_source", kernel.source());
    assert_debug_snapshot!("conv2d_forward_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_conv2d_backward_dx_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::conv::conv2d::Conv2dBackwardDx::<
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

    assert_debug_snapshot!("conv2d_backward_dx_source", kernel.source());
    assert_debug_snapshot!("conv2d_backward_dx_mlir", mlir.trim());

    Ok(())
}

#[test]
fn test_conv2d_backward_dw_mlir_output() -> Result<()> {
    dotenv()?;

    let kernel = teeny_kernels::conv::conv2d::Conv2dBackwardDw::<
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

    assert_debug_snapshot!("conv2d_backward_dw_source", kernel.source());
    assert_debug_snapshot!("conv2d_backward_dw_mlir", mlir.trim());

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_arr = Array4::<f32>::random((B, C_IN, H, W), Uniform::new(-1.0f32, 1.0f32).unwrap());
    let w_arr = ndarray::Array4::<f32>::random(
        (C_OUT, C_IN, KH as usize, KW as usize),
        Uniform::new(-0.5f32, 0.5f32).unwrap(),
    );

    let x_host: Vec<f32> = x_arr.iter().copied().collect();
    let w_host: Vec<f32> = w_arr.iter().copied().collect();
    let expected_arr = cpu_conv2d_forward(&x_arr, &w_arr);
    let expected: Vec<f32> = expected_arr.iter().copied().collect();

    let mut x_buf = device.buffer::<f32>(B * C_IN * H * W)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KH as usize * KW as usize)?;
    let y_buf = device.buffer::<f32>(B * C_OUT * OH * OW)?;
    let mut y_host = vec![0.0f32; B * C_OUT * OH * OW];

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;

    let kernel = teeny_kernels::conv::conv2d::Conv2dForward::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[conv2d_forward] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::conv::conv2d::Conv2dForward<f32, KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW>,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OH * num_ow_tiles;
    let cfg = CudaLaunchConfig {
        grid: [grid_size as u32, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..(B * C_OUT * OH * OW) {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "conv2d_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_backward_dx_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_arr = Array4::<f32>::random((B, C_OUT, OH, OW), Uniform::new(-1.0f32, 1.0f32).unwrap());
    let w_arr = ndarray::Array4::<f32>::random(
        (C_OUT, C_IN, KH as usize, KW as usize),
        Uniform::new(-0.5f32, 0.5f32).unwrap(),
    );

    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();
    let w_host: Vec<f32> = w_arr.iter().copied().collect();
    let expected_arr = cpu_conv2d_backward_dx(&dy_arr, &w_arr);
    let expected: Vec<f32> = expected_arr.iter().copied().collect();

    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OH * OW)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KH as usize * KW as usize)?;
    let dx_buf = device.buffer::<f32>(B * C_IN * H * W)?;
    let mut dx_host = vec![0.0f32; B * C_IN * H * W];

    dy_buf.to_device(&dy_host)?;
    w_buf.to_device(&w_host)?;

    let kernel = teeny_kernels::conv::conv2d::Conv2dBackwardDx::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::conv::conv2d::Conv2dBackwardDx<f32, KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW>,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OH * num_ow_tiles;
    let cfg = CudaLaunchConfig {
        grid: [grid_size as u32, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(B * C_IN * H * W) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "conv2d_backward_dx mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_backward_dw_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_arr = Array4::<f32>::random((B, C_IN, H, W), Uniform::new(-1.0f32, 1.0f32).unwrap());
    let dy_arr = Array4::<f32>::random((B, C_OUT, OH, OW), Uniform::new(-1.0f32, 1.0f32).unwrap());

    let x_host: Vec<f32> = x_arr.iter().copied().collect();
    let dy_host: Vec<f32> = dy_arr.iter().copied().collect();
    let expected_arr = cpu_conv2d_backward_dw(&dy_arr, &x_arr);
    let expected: Vec<f32> = expected_arr.iter().copied().collect();

    let mut x_buf = device.buffer::<f32>(B * C_IN * H * W)?;
    let mut dy_buf = device.buffer::<f32>(B * C_OUT * OH * OW)?;
    let dw_buf = device.buffer::<f32>(C_OUT * C_IN * KH as usize * KW as usize)?;
    let mut dw_host = vec![0.0f32; C_OUT * C_IN * KH as usize * KW as usize];

    x_buf.to_device(&x_host)?;
    dy_buf.to_device(&dy_host)?;

    let kernel = teeny_kernels::conv::conv2d::Conv2dBackwardDw::<
        f32,
        KH,
        KW,
        STRIDE_H,
        STRIDE_W,
        BLOCK_OW,
    >::new();
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<
        teeny_kernels::conv::conv2d::Conv2dBackwardDw<f32, KH, KW, STRIDE_H, STRIDE_W, BLOCK_OW>,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid_size = B * C_OUT * OH * num_ow_tiles;
    let cfg = CudaLaunchConfig {
        grid: [grid_size as u32, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        dw_buf.as_device_ptr() as *mut f32,
        B as i32,
        C_IN as i32,
        C_OUT as i32,
        H as i32,
        W as i32,
        OH as i32,
        OW as i32,
    );

    device.launch(&program, &cfg, args)?;
    dw_buf.to_host(&mut dw_host)?;

    for i in 0..(C_OUT * C_IN * KH as usize * KW as usize) {
        assert!(
            (dw_host[i] - expected[i]).abs() < 1e-3,
            "conv2d_backward_dw mismatch at {i}: gpu={}, expected={}",
            dw_host[i],
            expected[i]
        );
    }

    Ok(())
}
