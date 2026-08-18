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

//! Tests for the hand-written fused Conv2d+BatchNorm2d+SiLU kernels
//! (`nn::fused::conv2d_bn_silu{,_gemm,_tiled}`).
//!
//! These are constructed directly, not through `Graph`/`TritonLowering` —
//! Anduin does not fuse this pattern (see `nn/fused/mod.rs`), so there is no
//! graph-level path to exercise. Each variant is checked against the same
//! PyTorch fixture (1x1/stride=1/pad=0/groups=1 conv, so all three variants,
//! including the GEMM path, are legal for this shape).

use std::path::PathBuf;

use insta::assert_debug_snapshot;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Capability, target::Target};
use teeny_kernels::nn::fused::{
    conv2d_bn_silu::Conv2dBnSiluForward, conv2d_bn_silu_gemm::Conv2dBnSiluGemmForward,
    conv2d_bn_silu_tiled::Conv2dBnSiluTiledForward, prefold_bn_affine,
};
use teeny_kernels::testing::load_fixture;

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "cuda")]
use teeny_cuda::{device::CudaLaunchConfig, errors::Result, testing};

// Fixture shape: NCHW B=1, C_IN=32, C_OUT=64, H=W=8, 1x1/stride=1/pad=0/groups=1
// (see tests/fixtures/generate.py). M = OH*OW = 64 is exactly BLOCK_M/BLOCK_OW-
// aligned for every variant below, so none of the padded-output kernels need
// host-side depadding here.
const NB: usize = 1;
const C_IN: usize = 32;
const C_OUT: usize = 64;
const HH: usize = 8;
const WW: usize = 8;
const EPS: f32 = 1e-5;

const BLOCK_OW_SCALAR: i32 = 8;
const BLOCK_OW_TILED: i32 = 8;
const BLOCK_N_TILED: i32 = 16;
const BLOCK_M_GEMM: i32 = 32;
const BLOCK_N_GEMM: i32 = 32;
const BLOCK_K_GEMM: i32 = 32;
const GROUP_M_GEMM: i32 = 8;

const PTX_LAUNCH_THREADS_X: u32 = 128;

/// Load the fixture's raw BN params and prefold them into (scale, shift), the
/// form all three kernels expect (see `prefold_bn_affine`'s doc comment).
fn load_bn_scale_shift() -> (Vec<f32>, Vec<f32>) {
    let gamma = load_fixture("conv2d_bn_silu/bn_weight.bin");
    let beta = load_fixture("conv2d_bn_silu/bn_bias.bin");
    let mean = load_fixture("conv2d_bn_silu/bn_running_mean.bin");
    let var = load_fixture("conv2d_bn_silu/bn_running_var.bin");
    prefold_bn_affine(&gamma, &beta, &mean, &var, EPS)
}

// ---------------------------------------------------------------------------
// MLIR/source snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_bn_silu_forward_mlir_output() -> anyhow::Result<()> {
    let kernel = Conv2dBnSiluForward::new(1, 1, 1, 1, 0, 0, 1, BLOCK_OW_SCALAR);
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("conv2d_bn_silu_forward_source", kernel.source());
    assert_debug_snapshot!("conv2d_bn_silu_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_conv2d_bn_silu_tiled_forward_mlir_output() -> anyhow::Result<()> {
    let kernel = Conv2dBnSiluTiledForward::new(1, 1, 1, 1, 0, 0, BLOCK_OW_TILED, BLOCK_N_TILED);
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("conv2d_bn_silu_tiled_forward_source", kernel.source());
    assert_debug_snapshot!("conv2d_bn_silu_tiled_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_conv2d_bn_silu_gemm_forward_mlir_output() -> anyhow::Result<()> {
    let kernel =
        Conv2dBnSiluGemmForward::new(BLOCK_M_GEMM, BLOCK_N_GEMM, BLOCK_K_GEMM, GROUP_M_GEMM);
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("conv2d_bn_silu_gemm_forward_source", kernel.source());
    assert_debug_snapshot!("conv2d_bn_silu_gemm_forward_mlir", mlir.trim());
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA integration tests against the PyTorch fixture
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("conv2d_bn_silu/x.bin");
    let w_host = load_fixture("conv2d_bn_silu/w.bin");
    let expected = load_fixture("conv2d_bn_silu/expected_forward.bin");
    let (bn_scale, bn_shift) = load_bn_scale_shift();
    let mut y_host = vec![0.0f32; NB * C_OUT * HH * WW];

    let mut x_buf = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf = device.buffer::<f32>(w_host.len())?;
    let mut s_buf = device.buffer::<f32>(C_OUT)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf = device.buffer::<f32>(NB * C_OUT * HH * WW)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel = Conv2dBnSiluForward::new(1, 1, 1, 1, 0, 0, 1, BLOCK_OW_SCALAR);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<Conv2dBnSiluForward>(&ptx)?;

    let num_ow_tiles = WW.div_ceil(BLOCK_OW_SCALAR as usize);
    let grid = (NB * C_OUT * HH * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            w_buf.as_device_ptr() as *mut f32,
            s_buf.as_device_ptr() as *mut f32,
            sh_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            NB as i32,
            C_IN as i32,
            C_OUT as i32,
            HH as i32,
            WW as i32,
            HH as i32,
            WW as i32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..NB * C_OUT * HH * WW {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-3,
            "conv2d_bn_silu_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_tiled_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("conv2d_bn_silu/x.bin");
    let w_host = load_fixture("conv2d_bn_silu/w.bin");
    let expected = load_fixture("conv2d_bn_silu/expected_forward.bin");
    let (bn_scale, bn_shift) = load_bn_scale_shift();

    // y_col_stride == WW here (WW is already BLOCK_OW_TILED-aligned), so the
    // padded and tight NCHW layouts coincide — no depadding needed.
    let y_col_stride = WW.max(BLOCK_OW_TILED as usize).next_multiple_of(4);
    assert_eq!(y_col_stride, WW, "test shape must not require depadding");
    let mut y_host = vec![0.0f32; NB * C_OUT * HH * WW];

    let mut x_buf = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf = device.buffer::<f32>(w_host.len())?;
    let mut s_buf = device.buffer::<f32>(C_OUT)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf = device.buffer::<f32>(NB * C_OUT * HH * y_col_stride)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel = Conv2dBnSiluTiledForward::new(1, 1, 1, 1, 0, 0, BLOCK_OW_TILED, BLOCK_N_TILED);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<Conv2dBnSiluTiledForward>(&ptx)?;

    let num_ow_tiles = WW.div_ceil(BLOCK_OW_TILED as usize);
    let num_n_tiles = C_OUT.div_ceil(BLOCK_N_TILED as usize);
    let grid = (NB * HH * num_n_tiles * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            w_buf.as_device_ptr() as *mut f32,
            s_buf.as_device_ptr() as *mut f32,
            sh_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            NB as i32,
            C_IN as i32,
            C_OUT as i32,
            HH as i32,
            WW as i32,
            HH as i32,
            WW as i32,
            y_col_stride as i32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..NB * C_OUT * HH * WW {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-3,
            "conv2d_bn_silu_tiled_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_gemm_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host = load_fixture("conv2d_bn_silu/x.bin");
    // GEMM kernel expects weight shape [C_OUT, C_IN] (1x1 conv, no KH/KW dims) —
    // same flat layout as the fixture's [C_OUT, C_IN, 1, 1] weight.
    let w_host = load_fixture("conv2d_bn_silu/w.bin");
    let expected = load_fixture("conv2d_bn_silu/expected_forward.bin");
    let (bn_scale, bn_shift) = load_bn_scale_shift();

    let m = HH * WW;
    let y_row_stride = m.next_multiple_of(BLOCK_M_GEMM as usize);
    assert_eq!(y_row_stride, m, "test shape must not require depadding");
    let mut y_host = vec![0.0f32; NB * C_OUT * m];

    let mut x_buf = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf = device.buffer::<f32>(w_host.len())?;
    let mut s_buf = device.buffer::<f32>(C_OUT)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf = device.buffer::<f32>(NB * C_OUT * y_row_stride)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel =
        Conv2dBnSiluGemmForward::new(BLOCK_M_GEMM, BLOCK_N_GEMM, BLOCK_K_GEMM, GROUP_M_GEMM);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<Conv2dBnSiluGemmForward>(&ptx)?;

    let num_pm = m.div_ceil(BLOCK_M_GEMM as usize);
    let num_pn = C_OUT.div_ceil(BLOCK_N_GEMM as usize);
    let grid = (NB * num_pm * num_pn) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            w_buf.as_device_ptr() as *mut f32,
            s_buf.as_device_ptr() as *mut f32,
            sh_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            NB as i32,
            C_IN as i32,
            C_OUT as i32,
            m as i32,
            y_row_stride as i32,
        ),
    )?;
    y_buf.to_host(&mut y_host)?;

    // TF32 tensor-core dot has reduced mantissa precision (10-bit vs 23-bit
    // for fp32) — atol + rtol scales the tolerance with output magnitude.
    const ATOL: f32 = 1e-2;
    const RTOL: f32 = 1e-2;
    for i in 0..NB * C_OUT * m {
        let tol = ATOL + RTOL * expected[i].abs();
        assert!(
            (y_host[i] - expected[i]).abs() < tol,
            "conv2d_bn_silu_gemm_forward mismatch at {i}: gpu={}, expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}
