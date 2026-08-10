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

//! Tests for the channel-tiled fused Conv2d + BatchNorm2d + SiLU kernel.

use std::rc::Rc;

use teeny_core::{
    graph::{DtypeRepr, Graph, Op, SymTensor},
    model::LoweringMode,
};
use teeny_kernels::graph::{Anduin, GraphOptimizer, TritonLowering};

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::{compile_kernel, target::Target};
#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "cuda")]
use teeny_cuda::{device::CudaLaunchConfig, errors::Result, testing};

// Dimensions — chosen so that C_OUT ≥ 16 (triggers tiled kernel) and
// C_OW is not a multiple of BLOCK_OW to exercise masking.
const NB: usize = 1;
const C_IN: usize = 4;
const C_OUT: usize = 32; // ≥ 16 → tiled kernel
const HH: usize = 7;
const WW: usize = 7;
const KH: i32 = 3;
const KW: i32 = 3;
const STRIDE_H: i32 = 1;
const STRIDE_W: i32 = 1;
const PAD_H: i32 = 1;
const PAD_W: i32 = 1;
const BLOCK_OW: i32 = 16;
const BLOCK_N: i32 = 16;
const OH: usize = (HH + 2 * PAD_H as usize - KH as usize) / STRIDE_H as usize + 1; // 7
const OW: usize = (WW + 2 * PAD_W as usize - KW as usize) / STRIDE_W as usize + 1; // 7

fn conv2d_reference(
    x: &[f32],
    w: &[f32],
    b: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w_sz: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
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
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = (ohi * stride_h + khi) as isize - pad_h as isize;
                                let iw = (owi * stride_w + kwi) as isize - pad_w as isize;
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

fn bn_affine_silu_reference(
    conv_out: &[f32],
    bn_scale: &[f32],
    bn_shift: &[f32],
    c_out: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let hw = oh * ow;
    conv_out
        .iter()
        .enumerate()
        .map(|(idx, &val)| {
            let c = (idx / hw) % c_out;
            let bn_out = bn_scale[c] * val + bn_shift[c];
            bn_out / (1.0 + (-bn_out).exp())
        })
        .collect()
}

fn build_tiled_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(NB as usize), Some(C_IN), Some(HH), Some(WW)],
    );
    let conv_shape = vec![Some(NB), Some(C_OUT), Some(OH), Some(OW)];
    let conv_id = input.graph.borrow_mut().add_node(
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
            has_bias: false,
        },
        vec![input.node_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d {
            num_features: C_OUT,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        },
        vec![conv_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Silu, vec![bn_id], DtypeRepr::F32, conv_shape);
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_tiled_lowering_produces_tiled_kernel() {
    let graph = Anduin.optimize(&build_tiled_graph()).unwrap();
    let lowering = TritonLowering::new();
    let (dag, _) = lowering
        .lower_with_mapping(&graph, LoweringMode::Inference)
        .expect("lowering");
    let fused = dag.node(1);
    assert!(
        fused.value.name().contains("conv2d_bn_silu_tiled"),
        "expected tiled kernel name, got: {}",
        fused.value.name()
    );
}

#[test]
fn test_tiled_kernel_source_snapshot() {
    let graph = Anduin.optimize(&build_tiled_graph()).unwrap();
    let lowering = TritonLowering::new();
    let (dag, _) = lowering
        .lower_with_mapping(&graph, LoweringMode::Inference)
        .expect("lowering");
    let src = dag.node(1).value.forward_kernel_source();
    insta::assert_snapshot!("test_conv2d_bn_silu_tiled__tiled_forward_source", src);
}

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_tiled_matches_reference() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x_host: Vec<f32> = (0..NB * C_IN * HH * WW)
        .map(|i| (i as f32 % 17.0 - 8.0) * 0.1)
        .collect();
    let w_host: Vec<f32> = (0..C_OUT * C_IN * KH as usize * KW as usize)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();
    let bn_scale: Vec<f32> = (0..C_OUT).map(|i| 0.8 + i as f32 * 0.05).collect();
    let bn_shift: Vec<f32> = (0..C_OUT).map(|i| i as f32 * 0.1 - 0.15).collect();
    // y_col_stride = max(OW, BLOCK_OW).next_multiple_of(4): avoids TMA overlap and misalignment.
    let y_col_stride = OW.max(BLOCK_OW as usize).next_multiple_of(4); // max(7,16)=16
    let y_total = NB * C_OUT * OH * y_col_stride;

    let conv_out = conv2d_reference(
        &x_host,
        &w_host,
        NB,
        C_IN,
        C_OUT,
        HH,
        WW,
        KH as usize,
        KW as usize,
        STRIDE_H as usize,
        STRIDE_W as usize,
        PAD_H as usize,
        PAD_W as usize,
        OH,
        OW,
    );
    let expected = bn_affine_silu_reference(&conv_out, &bn_scale, &bn_shift, C_OUT, OH, OW);

    let mut x_buf = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KH as usize * KW as usize)?;
    let mut s_buf = device.buffer::<f32>(C_OUT)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf = device.buffer::<f32>(y_total)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel = teeny_kernels::nn::fused::conv2d_bn_silu_tiled::Conv2dBnSiluTiledForward::new(
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, BLOCK_OW, BLOCK_N,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::fused::conv2d_bn_silu_tiled::Conv2dBnSiluTiledForward,
    >(&ptx)?;

    let num_ow_tiles = OW.div_ceil(BLOCK_OW as usize);
    let num_n_tiles = C_OUT.div_ceil(BLOCK_N as usize);
    let grid = (NB * OH * num_n_tiles * num_ow_tiles) as u32;
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
            s_buf.as_device_ptr() as *mut f32,
            sh_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            NB as i32,
            C_IN as i32,
            C_OUT as i32,
            HH as i32,
            WW as i32,
            OH as i32,
            OW as i32,
            y_col_stride as i32,
        ),
    )?;
    let mut y_gpu_flat = vec![0.0f32; y_total];
    y_buf.to_host(&mut y_gpu_flat)?;

    // Layout: y_flat[(b*C_OUT + co) * OH * y_col_stride + oh * y_col_stride + ow]
    for bi in 0..NB {
        for co in 0..C_OUT {
            for ohi in 0..OH {
                for owi in 0..OW {
                    let gpu_val = y_gpu_flat
                        [(bi * C_OUT + co) * OH * y_col_stride + ohi * y_col_stride + owi];
                    let ref_idx = ((bi * C_OUT + co) * OH + ohi) * OW + owi;
                    assert!(
                        (gpu_val - expected[ref_idx]).abs() < 1e-4,
                        "mismatch at [{},{},{},{}]: gpu={:.6} ref={:.6}",
                        bi,
                        co,
                        ohi,
                        owi,
                        gpu_val,
                        expected[ref_idx],
                    );
                }
            }
        }
    }
    Ok(())
}
