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

//! Tests for the GEMM-based fused Conv2d(1×1) + BatchNorm2d + SiLU kernel.

use std::rc::Rc;

use teeny_core::{
    graph::{DtypeRepr, Graph, Op, SymTensor},
    model::LoweringMode,
};
use teeny_kernels::graph::TritonLowering;

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "cuda")]
use teeny_cuda::{device::CudaLaunchConfig, errors::Result, testing};

// 1×1 convolution: KH=KW=1, STRIDE=1, PAD=0, G=1, C_OUT≥32 → GEMM kernel.
const NB: usize = 1;
const C_IN: usize = 32;
const C_OUT: usize = 64; // ≥ 32 → GEMM kernel
const HH: usize = 8;
const WW: usize = 8;
const M: usize = HH * WW; // 64 spatial positions
const BLOCK_M: i32 = 32;
const BLOCK_N: i32 = 32;
const BLOCK_K: i32 = 32;
const GROUP_M: i32 = 8;

fn conv1x1_reference(x: &[f32], w: &[f32], b: usize, c_in: usize, c_out: usize, h: usize, w_sz: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; b * c_out * h * w_sz];
    for bi in 0..b {
        for co in 0..c_out {
            for hi in 0..h {
                for wi in 0..w_sz {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        let xi = ((bi * c_in + ci) * h + hi) * w_sz + wi;
                        let wj = co * c_in + ci; // weight layout [C_OUT, C_IN]
                        acc += x[xi] * w[wj];
                    }
                    y[((bi * c_out + co) * h + hi) * w_sz + wi] = acc;
                }
            }
        }
    }
    y
}

fn bn_affine_silu_reference(conv_out: &[f32], bn_scale: &[f32], bn_shift: &[f32], c_out: usize, oh: usize, ow: usize) -> Vec<f32> {
    let hw = oh * ow;
    conv_out.iter().enumerate().map(|(idx, &val)| {
        let c = (idx / hw) % c_out;
        let bn_out = bn_scale[c] * val + bn_shift[c];
        bn_out / (1.0 + (-bn_out).exp())
    }).collect()
}

fn build_gemm_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(NB), Some(C_IN), Some(HH), Some(WW)],
    );
    let conv_shape = vec![Some(NB), Some(C_OUT), Some(HH), Some(WW)];
    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: C_IN, out_channels: C_OUT,
            kernel_h: 1, kernel_w: 1,
            stride_h: 1, stride_w: 1,
            padding_h: 0, padding_w: 0,
            groups: 1, has_bias: false,
        },
        vec![input.node_id], DtypeRepr::F32, conv_shape.clone(),
    );
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d { num_features: C_OUT, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true },
        vec![conv_id], DtypeRepr::F32, conv_shape.clone(),
    );
    let _ = input.graph.borrow_mut().add_node(
        Op::Silu, vec![bn_id], DtypeRepr::F32, conv_shape,
    );
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_gemm_lowering_selects_gemm_kernel() {
    let graph = build_gemm_graph().optimise();
    let lowering = TritonLowering::new();
    let (dag, _) = lowering.lower_with_mapping(&graph, LoweringMode::Inference).expect("lowering");
    let fused = dag.node(1);
    assert!(
        fused.value.name().contains("conv2d_bn_silu_gemm"),
        "expected GEMM kernel name, got: {}", fused.value.name()
    );
}

#[test]
fn test_gemm_kernel_source_snapshot() {
    let graph = build_gemm_graph().optimise();
    let lowering = TritonLowering::new();
    let (dag, _) = lowering.lower_with_mapping(&graph, LoweringMode::Inference).expect("lowering");
    let src = dag.node(1).value.forward_kernel_source();
    insta::assert_snapshot!("test_conv2d_bn_silu_gemm__gemm_forward_source", src);
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_gemm_matches_reference() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host: Vec<f32> = (0..NB * C_IN * HH * WW).map(|i| (i as f32 % 17.0 - 8.0) * 0.1).collect();
    // GEMM kernel expects weight shape [C_OUT, C_IN] (1×1 conv, no KH/KW dims)
    let w_host: Vec<f32> = (0..C_OUT * C_IN).map(|i| (i as f32 % 13.0 - 6.0) * 0.05).collect();
    let bn_scale: Vec<f32> = (0..C_OUT).map(|i| 0.8 + i as f32 * 0.05).collect();
    let bn_shift: Vec<f32> = (0..C_OUT).map(|i| i as f32 * 0.1 - 0.15).collect();
    let mut y_gpu = vec![0.0f32; NB * C_OUT * HH * WW];

    // Reference: 1×1 conv (w layout [C_OUT, C_IN]) + BN + SiLU
    let conv_out = conv1x1_reference(&x_host, &w_host, NB, C_IN, C_OUT, HH, WW);
    let expected = bn_affine_silu_reference(&conv_out, &bn_scale, &bn_shift, C_OUT, HH, WW);

    let mut x_buf  = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf  = device.buffer::<f32>(C_OUT * C_IN)?;
    let mut s_buf  = device.buffer::<f32>(C_OUT)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf      = device.buffer::<f32>(NB * C_OUT * HH * WW)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel = teeny_kernels::nn::fused::conv2d_bn_silu_gemm::Conv2dBnSiluGemmForward::new(
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::fused::conv2d_bn_silu_gemm::Conv2dBnSiluGemmForward,
    >(&ptx)?;

    let num_pm = M.div_ceil(BLOCK_M as usize);
    let num_pn = C_OUT.div_ceil(BLOCK_N as usize);
    let grid = (NB * num_pm * num_pn) as u32;
    let cfg = CudaLaunchConfig { grid: [grid, 1, 1], block: [128, 1, 1], cluster: [1, 1, 1] };

    device.launch(&program, &cfg, (
        x_buf.as_device_ptr()  as *mut f32,
        w_buf.as_device_ptr()  as *mut f32,
        s_buf.as_device_ptr()  as *mut f32,
        sh_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr()  as *mut f32,
        NB as i32,
        C_IN as i32,
        C_OUT as i32,
        M as i32,
    ))?;
    y_buf.to_host(&mut y_gpu)?;

    // TF32 has reduced mantissa precision (10-bit vs 23-bit for fp32).
    // Use a relaxed tolerance (1e-2) for TF32 tensor-core results.
    for i in 0..NB * C_OUT * HH * WW {
        assert!(
            (y_gpu[i] - expected[i]).abs() < 1e-2,
            "mismatch at [{}]: gpu={:.6} ref={:.6}", i, y_gpu[i], expected[i],
        );
    }
    Ok(())
}

/// Test the exact model.6.m.0.cv2 configuration: C_OUT=32 (exactly 1 N-tile),
/// C_IN=64 (2 K-tiles), M=1600 (40×40). This reproduces the failing scenario.
#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_gemm_c_out32_c_in64_m1600() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    const NB2: usize = 1;
    const C_IN2: usize = 64;
    const C_OUT2: usize = 32; // exactly 1 N-tile — the failing case
    const M2: usize = 1600;   // 40×40 spatial

    let x_host: Vec<f32> = (0..NB2 * C_IN2 * M2)
        .map(|i| (i as f32 % 17.0 - 8.0) * 0.1)
        .collect();
    let w_host: Vec<f32> = (0..C_OUT2 * C_IN2)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();
    // Large bn_scale (like channel 2 of model.6.m.0.cv2) to amplify errors
    let bn_scale: Vec<f32> = (0..C_OUT2)
        .map(|i| if i == 2 { 8.216f32 } else { 1.0 + i as f32 * 0.05 })
        .collect();
    let bn_shift: Vec<f32> = (0..C_OUT2).map(|i| i as f32 * 0.1 - 0.15).collect();
    let mut y_gpu = vec![0.0f32; NB2 * C_OUT2 * M2];

    let conv_out = conv1x1_reference(&x_host, &w_host, NB2, C_IN2, C_OUT2, 1, M2);
    let expected = bn_affine_silu_reference(&conv_out, &bn_scale, &bn_shift, C_OUT2, 1, M2);

    let mut x_buf  = device.buffer::<f32>(NB2 * C_IN2 * M2)?;
    let mut w_buf  = device.buffer::<f32>(C_OUT2 * C_IN2)?;
    let mut s_buf  = device.buffer::<f32>(C_OUT2)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT2)?;
    let y_buf      = device.buffer::<f32>(NB2 * C_OUT2 * M2)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel = teeny_kernels::nn::fused::conv2d_bn_silu_gemm::Conv2dBnSiluGemmForward::new(
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::fused::conv2d_bn_silu_gemm::Conv2dBnSiluGemmForward,
    >(&ptx)?;

    let num_pm = M2.div_ceil(BLOCK_M as usize);
    let num_pn = C_OUT2.div_ceil(BLOCK_N as usize);
    let grid = (NB2 * num_pm * num_pn) as u32;
    let cfg = CudaLaunchConfig { grid: [grid, 1, 1], block: [128, 1, 1], cluster: [1, 1, 1] };

    device.launch(&program, &cfg, (
        x_buf.as_device_ptr()  as *mut f32,
        w_buf.as_device_ptr()  as *mut f32,
        s_buf.as_device_ptr()  as *mut f32,
        sh_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr()  as *mut f32,
        NB2 as i32,
        C_IN2 as i32,
        C_OUT2 as i32,
        M2 as i32,
    ))?;
    y_buf.to_host(&mut y_gpu)?;

    let mut max_err = 0.0f32;
    for i in 0..NB2 * C_OUT2 * M2 {
        let err = (y_gpu[i] - expected[i]).abs();
        if err > max_err { max_err = err; }
    }
    eprintln!("C_OUT=32/C_IN=64/M=1600 max_err={max_err:.6e}");
    for i in 0..NB2 * C_OUT2 * M2 {
        assert!(
            (y_gpu[i] - expected[i]).abs() < 1e-2,
            "mismatch at [{}]: gpu={:.6} ref={:.6} diff={:.6e}",
            i, y_gpu[i], expected[i], (y_gpu[i] - expected[i]).abs()
        );
    }
    Ok(())
}
