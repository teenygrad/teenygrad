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

// PSA layout-op backward kernel tests.
//
// Tests psa_pack_qkv_backward, psa_extract_v_backward, and
// psa_merge_attn_backward via MLIR snapshot and CUDA integration.
//
// Dimensions:
//   B=1, NUM_HEADS=1, KEY_DIM=4, H=2, W=2 → N=4, BH=1
//   qkv_h = 1*4*4 = 16 channels,  c = 1*2*4 = 8 channels

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

// ── Test dimensions ───────────────────────────────────────────────────────────
const KEY_DIM: i32    = 4;
const KEY_DIM_U: usize = 4;
const NUM_HEADS: usize = 1;
const B: usize         = 1;
const HH: usize        = 2; // spatial height (avoid clash with `H`)
const WW: usize        = 2; // spatial width
const N: usize         = HH * WW;  // 4
const BH: usize        = B * NUM_HEADS; // 1
const QKV_H: usize     = NUM_HEADS * 4 * KEY_DIM_U; // 16
const C: usize         = NUM_HEADS * 2 * KEY_DIM_U; // 8

// Buffer element counts
const N_PACKED: usize = 4 * BH * N * KEY_DIM_U;  // 64  [4,BH,N,KD]
const N_QKV:   usize = B * QKV_H * HH * WW;       // 64  [B,qkv_h,H,W]
const N_V:     usize = B * C * HH * WW;            // 32  [B,c,H,W]
const N_FLAT:  usize = BH * N * KEY_DIM_U;         // 16  [BH,N,KD]

const PTX_LAUNCH_THREADS_X: u32 = 128;

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_psa_pack_qkv_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::attention::psa::PsaPackQkvBackward::new(KEY_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("psa_pack_qkv_backward_source", kernel.source());
    assert_debug_snapshot!("psa_pack_qkv_backward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_psa_extract_v_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::attention::psa::PsaExtractVBackward::new(KEY_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("psa_extract_v_backward_source", kernel.source());
    assert_debug_snapshot!("psa_extract_v_backward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_psa_merge_attn_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::attention::psa::PsaMergeAttnBackward::new(KEY_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("psa_merge_attn_backward_source", kernel.source());
    assert_debug_snapshot!("psa_merge_attn_backward_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

// psa_pack_qkv_backward with d_packed = ones → d_qkv = ones.
//
// Each (section, bh, n, d) in d_packed maps to a unique channel position in
// d_qkv, so every element receives exactly one atomic_add of 1.0.
#[test]
#[cfg(feature = "cuda")]
fn test_psa_pack_qkv_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let d_packed_host = vec![1.0_f32; N_PACKED];
    let mut d_packed_buf = device.buffer::<f32>(N_PACKED)?;
    d_packed_buf.to_device(&d_packed_host)?;
    let d_qkv_buf = device.buffer::<f32>(N_QKV)?; // zero-initialised

    let kernel = teeny_kernels::nn::attention::psa::PsaPackQkvBackward::new(KEY_DIM);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::attention::psa::PsaPackQkvBackward,
    >(&ptx)?;

    // grid = [4 * BH * N, 1, 1]
    let cfg = CudaLaunchConfig {
        grid:    [(4 * BH * N) as u32, 1, 1],
        block:   [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(&program, &cfg, (
        d_packed_buf.as_device_ptr() as *mut f32,  // d_packed_ptr
        d_qkv_buf.as_device_ptr() as *mut f32,     // d_qkv_ptr
        QKV_H as i32,
        HH as i32,
        WW as i32,
        B as i32,
        NUM_HEADS as i32,
    ))?;

    let mut d_qkv_host = vec![0.0_f32; N_QKV];
    d_qkv_buf.to_host(&mut d_qkv_host)?;

    for (i, &v) in d_qkv_host.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "psa_pack_qkv_backward: d_qkv[{i}] = {v}, expected 1.0",
        );
    }

    Ok(())
}

// psa_extract_v_backward with d_v = ones.
//
// Only V_lo and V_hi channels of d_qkv are written (sections 2 and 3 per head).
// For NUM_HEADS=1, KEY_DIM=4: V channels are 8..16 (channels 8-15).
// Expected: d_qkv[channel*N + n] = 1.0 for channel ∈ [8,16), else 0.0.
#[test]
#[cfg(feature = "cuda")]
fn test_psa_extract_v_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let d_v_host = vec![1.0_f32; N_V];
    let mut d_v_buf = device.buffer::<f32>(N_V)?;
    d_v_buf.to_device(&d_v_host)?;
    let d_qkv_buf = device.buffer::<f32>(N_QKV)?; // zero-initialised

    let kernel = teeny_kernels::nn::attention::psa::PsaExtractVBackward::new(KEY_DIM);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::attention::psa::PsaExtractVBackward,
    >(&ptx)?;

    // grid = [BH * N, 1, 1]
    let cfg = CudaLaunchConfig {
        grid:    [(BH * N) as u32, 1, 1],
        block:   [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(&program, &cfg, (
        d_v_buf.as_device_ptr() as *mut f32,    // d_v_ptr
        d_qkv_buf.as_device_ptr() as *mut f32,  // d_qkv_ptr
        QKV_H as i32,
        C as i32,
        HH as i32,
        WW as i32,
        NUM_HEADS as i32,
    ))?;

    let mut d_qkv_host = vec![0.0_f32; N_QKV];
    d_qkv_buf.to_host(&mut d_qkv_host)?;

    // For head h=0: V_lo channels = [2*KEY_DIM_U .. 3*KEY_DIM_U) = [8..12)
    //               V_hi channels = [3*KEY_DIM_U .. 4*KEY_DIM_U) = [12..16)
    let v_chan_start = 2 * KEY_DIM_U; // inclusive
    let v_chan_end   = 4 * KEY_DIM_U; // exclusive

    for chan in 0..QKV_H {
        for n in 0..N {
            let idx = chan * N + n;
            let expected = if chan >= v_chan_start && chan < v_chan_end { 1.0 } else { 0.0 };
            assert!(
                (d_qkv_host[idx] - expected).abs() < 1e-6,
                "psa_extract_v_backward: d_qkv[chan={chan}, n={n}] = {}, expected {expected}",
                d_qkv_host[idx],
            );
        }
    }

    Ok(())
}

// psa_merge_attn_backward with d_merged = ones → d_lo = d_hi = ones.
//
// The merge backward simply routes each (h, d, n) position in the merged NCHW
// output back to the flat [BH, N, KEY_DIM] lo/hi buffers. With uniform input,
// every element of d_lo and d_hi receives exactly 1.0.
#[test]
#[cfg(feature = "cuda")]
fn test_psa_merge_attn_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // d_merged has shape [B, c, H, W] = [1, 8, 2, 2] → 32 elements
    let d_merged_host = vec![1.0_f32; N_V]; // N_V = B*C*H*W = 32
    let mut d_merged_buf = device.buffer::<f32>(N_V)?;
    d_merged_buf.to_device(&d_merged_host)?;
    let d_lo_buf = device.buffer::<f32>(N_FLAT)?; // zero-initialised
    let d_hi_buf = device.buffer::<f32>(N_FLAT)?; // zero-initialised

    let kernel = teeny_kernels::nn::attention::psa::PsaMergeAttnBackward::new(KEY_DIM);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::attention::psa::PsaMergeAttnBackward,
    >(&ptx)?;

    // grid = [BH * N, 1, 1]
    let cfg = CudaLaunchConfig {
        grid:    [(BH * N) as u32, 1, 1],
        block:   [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(&program, &cfg, (
        d_merged_buf.as_device_ptr() as *mut f32, // d_merged_ptr
        d_lo_buf.as_device_ptr() as *mut f32,     // d_lo_ptr
        d_hi_buf.as_device_ptr() as *mut f32,     // d_hi_ptr
        C as i32,
        HH as i32,
        WW as i32,
        NUM_HEADS as i32,
    ))?;

    let mut d_lo_host = vec![0.0_f32; N_FLAT];
    let mut d_hi_host = vec![0.0_f32; N_FLAT];
    d_lo_buf.to_host(&mut d_lo_host)?;
    d_hi_buf.to_host(&mut d_hi_host)?;

    for (i, &v) in d_lo_host.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "psa_merge_attn_backward: d_lo[{i}] = {v}, expected 1.0",
        );
    }
    for (i, &v) in d_hi_host.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "psa_merge_attn_backward: d_hi[{i}] = {v}, expected 1.0",
        );
    }

    Ok(())
}
