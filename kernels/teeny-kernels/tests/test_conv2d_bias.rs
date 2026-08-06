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

//! Tests for `conv2d_bias_forward` — a Conv2d + per-channel bias fused into one
//! kernel launch, used (in inference mode) instead of a separate Conv2d + NCHW
//! bias-add pair. See spinorml-ia5.

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

const NB: usize = 1;
const C_IN: usize = 3;
const C_OUT: usize = 4;
const HH: usize = 9;
const WW: usize = 9;
const KH: i32 = 3;
const KW: i32 = 3;
const STRIDE_H: i32 = 1;
const STRIDE_W: i32 = 1;
const PAD_H: i32 = 1;
const PAD_W: i32 = 1;
const BLOCK_OW: i32 = 16;
const OH: usize = (HH + 2 * PAD_H as usize - KH as usize) / STRIDE_H as usize + 1; // 9
const OW: usize = (WW + 2 * PAD_W as usize - KW as usize) / STRIDE_W as usize + 1; // 9

fn build_conv2d_bias_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(NB), Some(C_IN), Some(HH), Some(WW)],
    );
    let out_shape = vec![Some(NB), Some(C_OUT), Some(OH), Some(OW)];
    let _ = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: C_IN,
            out_channels: C_OUT,
            kernel_h: KH as usize,
            kernel_w: KW as usize,
            stride_h: STRIDE_H as usize,
            stride_w: STRIDE_W as usize,
            padding_h: PAD_H as usize,
            padding_w: PAD_W as usize,
            groups: 1,
            has_bias: true,
        },
        vec![input.node_id],
        DtypeRepr::F32,
        out_shape,
    );
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_conv2d_bias_inference_lowers_to_one_fused_kernel() {
    let graph = build_conv2d_bias_graph();
    let lowering = TritonLowering::new();
    let (dag, _) = lowering
        .lower_with_mapping(&graph, LoweringMode::Inference)
        .expect("lowering");

    // Input + one fused conv2d_bias node — not split into conv + bias-add.
    assert_eq!(dag.len(), 2);
    assert!(
        dag.node(1).value.name().contains("conv2d_bias_forward"),
        "expected conv2d_bias_forward kernel name, got: {}",
        dag.node(1).value.name()
    );
}

#[test]
#[cfg(feature = "training")]
fn test_conv2d_bias_training_still_splits_into_two_kernels() {
    // conv2d_bias_forward has no backward pass, so training must keep using the
    // existing Conv2d + NchwBiasAdd split (each has its own backward kernel).
    let graph = build_conv2d_bias_graph();
    let lowering = TritonLowering::new();
    let (dag, _) = lowering
        .lower_with_mapping(&graph, LoweringMode::Training)
        .expect("lowering");

    // Input + Conv2d(no bias) + NchwBiasAdd = 3 nodes.
    assert_eq!(dag.len(), 3);
    assert!(!dag.node(1).value.name().contains("conv2d_bias_forward"));
}

#[cfg(feature = "cuda")]
fn conv2d_bias_reference(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    b: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w_sz: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; b * c_out * oh * ow];
    for bi in 0..b {
        for co in 0..c_out {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut acc = bias[co];
                    for ci in 0..c_in {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = (ohi + khi) as isize - pad_h as isize;
                                let iw = (owi + kwi) as isize - pad_w as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w_sz as isize {
                                    let xi =
                                        ((bi * c_in + ci) * h + ih as usize) * w_sz + iw as usize;
                                    let wi = ((co * c_in + ci) * kh + khi) * kw + kwi;
                                    acc += x[xi] * w[wi];
                                }
                            }
                        }
                    }
                    y[((bi * c_out + co) * oh + ohi) * ow + owi] = acc;
                }
            }
        }
    }
    y
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bias_forward_matches_reference() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host: Vec<f32> = (0..NB * C_IN * HH * WW)
        .map(|i| (i as f32 % 17.0 - 8.0) * 0.1)
        .collect();
    let w_host: Vec<f32> = (0..C_OUT * C_IN * KH as usize * KW as usize)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();
    let bias_host: Vec<f32> = (0..C_OUT).map(|i| i as f32 * 0.1 - 0.2).collect();

    let expected = conv2d_bias_reference(
        &x_host,
        &w_host,
        &bias_host,
        NB,
        C_IN,
        C_OUT,
        HH,
        WW,
        KH as usize,
        KW as usize,
        PAD_H as usize,
        PAD_W as usize,
        OH,
        OW,
    );

    let mut x_buf = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KH as usize * KW as usize)?;
    let mut bias_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf = device.buffer::<f32>(NB * C_OUT * OH * OW)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    bias_buf.to_device(&bias_host)?;

    let kernel = teeny_kernels::nn::conv::conv2d::Conv2dBiasForward::<f32>::new(
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, 1, BLOCK_OW,
    );
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::conv::conv2d::Conv2dBiasForward<f32>,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let grid = (NB * C_OUT * OH * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [128, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            w_buf.as_device_ptr() as *mut f32,
            bias_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            NB as i32,
            C_IN as i32,
            C_OUT as i32,
            HH as i32,
            WW as i32,
            OH as i32,
            OW as i32,
        ),
    )?;

    let mut y_host = vec![0.0f32; NB * C_OUT * OH * OW];
    y_buf.to_host(&mut y_host)?;

    for i in 0..NB * C_OUT * OH * OW {
        assert!(
            (y_host[i] - expected[i]).abs() < 1e-4,
            "conv2d_bias_forward mismatch at {i}: gpu={} expected={}",
            y_host[i],
            expected[i]
        );
    }
    Ok(())
}
