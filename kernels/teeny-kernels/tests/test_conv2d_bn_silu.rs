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

//! Tests for the fused Conv2d + BatchNorm2d + SiLU op.
//!
//! These tests verify:
//! 1. `Graph::optimise()` correctly detects the Conv2d→BN→SiLU pattern and
//!    replaces it with a single `Op::Conv2dBnSilu` node.
//! 2. `TritonLowering` can lower `Op::Conv2dBnSilu` to a `KernelExecutable`
//!    with the expected kernel source.
//! 3. The GPU fused kernel produces numerical output matching a pure-Rust
//!    reference that runs the three operations separately.

use std::rc::Rc;

use teeny_core::{
    graph::{DtypeRepr, Graph, Op, SymTensor},
    model::LoweringMode,
};
use teeny_kernels::graph::TritonLowering;

// ── CUDA numerical test setup ─────────────────────────────────────────────────

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "cuda")]
use teeny_cuda::{device::CudaLaunchConfig, errors::Result, testing};

// Dimensions for the numerical test — kept small so it runs in milliseconds.
const NB: usize = 1; // batch
const C_IN: usize = 2;
const C_OUT: usize = 4;
const HH: usize = 6; // input spatial height
const WW: usize = 6; // input spatial width
const KH: i32 = 3;
const KW: i32 = 3;
const STRIDE_H: i32 = 1;
const STRIDE_W: i32 = 1;
const PAD_H: i32 = 1;
const PAD_W: i32 = 1;
const G: i32 = 1; // groups
const BLOCK_OW: i32 = 8;
const EPS: f32 = 1e-5;
const OH: usize = (HH + 2 * PAD_H as usize - KH as usize) / STRIDE_H as usize + 1; // 6
const OW: usize = (WW + 2 * PAD_W as usize - KW as usize) / STRIDE_W as usize + 1; // 6

// ── Pure-Rust reference implementations ──────────────────────────────────────

/// Conv2d forward on the CPU.  No bias, NCHW layout, groups=1.
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

/// Apply BN affine (inference-mode, precomputed scale/shift) then SiLU
/// element-wise.  Input is NCHW with B=1.
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
            // For B=1 NCHW layout, channel = idx / (oh*ow)
            let c = (idx / hw) % c_out;
            let bn_out = bn_scale[c] * val + bn_shift[c];
            // SiLU: y = x * sigmoid(x)
            bn_out / (1.0 + (-bn_out).exp())
        })
        .collect()
}

/// Build: `Input → Conv2d(no bias) → BatchNorm2d → Silu`
fn build_conv_bn_silu_graph() -> Graph {
    let (input, graph_rc) =
        SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(8), Some(8)]);

    // Conv2d: 3 → 16 channels, 3×3 kernel, stride 1, same-padding, no bias
    let conv_shape = vec![Some(1), Some(16), Some(8), Some(8)];
    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: 3,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 1,
            padding_w: 1,
            groups: 1,
            has_bias: false,
        },
        vec![input.node_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );

    // BatchNorm2d: 16 features
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d {
            num_features: 16,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        },
        vec![conv_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );

    // Silu
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Silu, vec![bn_id], DtypeRepr::F32, conv_shape);

    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[test]
fn test_conv_bn_silu_graph_has_four_nodes() {
    let graph = build_conv_bn_silu_graph();
    assert_eq!(graph.nodes.len(), 4, "expected Input + Conv2d + BN + Silu");
    assert!(matches!(graph.nodes[0].op, Op::Input));
    assert!(matches!(
        graph.nodes[1].op,
        Op::Conv2d {
            has_bias: false,
            ..
        }
    ));
    assert!(matches!(graph.nodes[2].op, Op::BatchNorm2d { .. }));
    assert!(matches!(graph.nodes[3].op, Op::Silu));
}

#[test]
fn test_optimise_fuses_conv_bn_silu() {
    let graph = build_conv_bn_silu_graph();
    let opt = graph.optimise();

    assert_eq!(
        opt.nodes.len(),
        2,
        "expected Input + Conv2dBnSilu after fusion"
    );
    assert!(matches!(opt.nodes[0].op, Op::Input));
    assert!(
        matches!(
            opt.nodes[1].op,
            Op::Conv2dBnSilu {
                in_channels: 3,
                out_channels: 16,
                kernel_h: 3,
                kernel_w: 3,
                ..
            }
        ),
        "node 1 should be Conv2dBnSilu, got: {:?}",
        opt.nodes[1].op
    );
}

#[test]
fn test_optimise_preserves_output_shape() {
    let graph = build_conv_bn_silu_graph();
    let original_output_shape = graph.nodes.last().unwrap().shape.clone();

    let opt = graph.optimise();
    let fused_shape = &opt.nodes[1].shape;

    assert_eq!(
        fused_shape, &original_output_shape,
        "fused node shape must match original Silu output shape"
    );
}

#[test]
fn test_optimise_rewires_inputs() {
    let graph = build_conv_bn_silu_graph();
    let opt = graph.optimise();

    // The fused node should take the Input node (index 0) as its only input.
    assert_eq!(
        opt.nodes[1].inputs,
        vec![0],
        "Conv2dBnSilu should consume the Input node directly"
    );
}

#[test]
fn test_lowering_produces_fused_kernel() {
    let graph = build_conv_bn_silu_graph();
    let opt = graph.optimise();

    let lowering = TritonLowering::new();
    let (dag, _mapping) = lowering
        .lower_with_mapping(&opt, LoweringMode::Inference)
        .expect("lowering should succeed");

    assert_eq!(
        dag.len(),
        2,
        "DAG should have Input node + fused kernel node"
    );

    let fused_node = dag.node(1);
    assert!(
        fused_node.value.name().contains("conv2d_bn_silu"),
        "expected fused kernel name to contain 'conv2d_bn_silu', got: {}",
        fused_node.value.name()
    );
    assert!(
        !fused_node.value.forward_kernel_source().is_empty(),
        "fused kernel should have non-empty source"
    );
}

#[test]
fn test_optimise_no_fusion_when_conv_has_bias() {
    let (input, graph_rc) =
        SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(8), Some(8)]);
    let conv_shape = vec![Some(1), Some(16), Some(8), Some(8)];
    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: 3,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 1,
            padding_w: 1,
            groups: 1,
            has_bias: true, // bias present — should NOT fuse
        },
        vec![input.node_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d {
            num_features: 16,
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
    let graph = Rc::try_unwrap(graph_rc).ok().unwrap().into_inner();

    let opt = graph.optimise();
    // No fusion; all 4 nodes survive.
    assert_eq!(opt.nodes.len(), 4, "Conv2d-with-bias should not be fused");
    assert!(
        !matches!(opt.nodes.last().unwrap().op, Op::Conv2dBnSilu { .. }),
        "last node should still be Silu, not Conv2dBnSilu"
    );
}

#[test]
fn test_optimise_no_fusion_when_conv_has_multiple_consumers() {
    let (input, graph_rc) =
        SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(8), Some(8)]);
    let conv_shape = vec![Some(1), Some(16), Some(8), Some(8)];
    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: 3,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 1,
            padding_w: 1,
            groups: 1,
            has_bias: false,
        },
        vec![input.node_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d {
            num_features: 16,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        },
        vec![conv_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    // Second consumer of conv (e.g. a skip connection via Relu)
    let _relu_id = input.graph.borrow_mut().add_node(
        Op::Relu,
        vec![conv_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Silu, vec![bn_id], DtypeRepr::F32, conv_shape);

    drop(input);
    let graph = Rc::try_unwrap(graph_rc).ok().unwrap().into_inner();

    let opt = graph.optimise();
    // Conv has 2 consumers — must not fuse.
    assert!(
        !opt.nodes
            .iter()
            .any(|n| matches!(n.op, Op::Conv2dBnSilu { .. })),
        "should not fuse when Conv2d has multiple consumers"
    );
}

// ── CUDA numerical correctness test ──────────────────────────────────────────
//
// Runs the fused GPU kernel and compares its output element-by-element against
// a pure-Rust reference that executes Conv2d, BN affine, and SiLU separately.

#[test]
#[cfg(feature = "cuda")]
fn test_conv2d_bn_silu_matches_reference() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // ── Deterministic test data (no fixtures needed) ──────────────────────
    let x_host: Vec<f32> = (0..NB * C_IN * HH * WW)
        .map(|i| (i as f32 % 17.0 - 8.0) * 0.1)
        .collect();
    let w_host: Vec<f32> = (0..C_OUT * C_IN * KH as usize * KW as usize)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();
    // Precomputed BN affine constants: bn_scale = gamma/sqrt(var+eps),
    // bn_shift = beta - bn_scale * mean  (exact formula used at inference).
    let bn_scale: Vec<f32> = (0..C_OUT).map(|i| 0.8 + i as f32 * 0.1).collect();
    let bn_shift: Vec<f32> = (0..C_OUT).map(|i| i as f32 * 0.1 - 0.15).collect();
    let mut y_gpu = vec![0.0f32; NB * C_OUT * OH * OW];

    // ── CPU reference ─────────────────────────────────────────────────────
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

    // ── GPU fused kernel ──────────────────────────────────────────────────
    let mut x_buf = device.buffer::<f32>(NB * C_IN * HH * WW)?;
    let mut w_buf = device.buffer::<f32>(C_OUT * C_IN * KH as usize * KW as usize)?;
    let mut s_buf = device.buffer::<f32>(C_OUT)?;
    let mut sh_buf = device.buffer::<f32>(C_OUT)?;
    let y_buf = device.buffer::<f32>(NB * C_OUT * OH * OW)?;

    x_buf.to_device(&x_host)?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&bn_scale)?;
    sh_buf.to_device(&bn_shift)?;

    let kernel = teeny_kernels::nn::fused::conv2d_bn_silu::Conv2dBnSiluForward::new(
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, G, BLOCK_OW,
    );
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::fused::conv2d_bn_silu::Conv2dBnSiluForward,
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
        ),
    )?;
    y_buf.to_host(&mut y_gpu)?;

    // ── Compare ───────────────────────────────────────────────────────────
    let n = NB * C_OUT * OH * OW;
    for i in 0..n {
        assert!(
            (y_gpu[i] - expected[i]).abs() < 1e-4,
            "mismatch at element {i}: fused_gpu={:.6} reference={:.6}",
            y_gpu[i],
            expected[i],
        );
    }
    Ok(())
}

// ── Pipeline-stage logging ────────────────────────────────────────────────────
//
// Verifies `LlvmCompiler::with_log_level` end-to-end: at `Debug`, `teenyc` should
// log every MLIR pipeline stage (ttir, ttgpuir, llir, llvmir, ptx/asm) for this
// fused kernel, relayed through this process's own `tracing`.

#[cfg(feature = "cuda")]
mod pipeline_logging {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use teeny_compiler::compiler::backend::llvm::compiler::{LlvmCompiler, LogLevel};
    use teeny_core::compiler::Compiler;

    use super::*;

    /// A `tracing_subscriber` writer that appends every formatted event into a
    /// shared buffer instead of stdout/stderr, so the test can inspect what was
    /// logged.
    #[derive(Clone, Default)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_conv2d_bn_silu_logs_pipeline_stages() {
        dotenv().ok();
        let env = testing::setup_cuda_env().expect("CUDA environment required for this test");

        let kernel = teeny_kernels::nn::fused::conv2d_bn_silu::Conv2dBnSiluForward::new(
            KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, G, BLOCK_OW,
        );
        let target = Target::new(env.capability);

        let teenyc_path =
            teeny_compiler::compiler::find_teenyc().expect("find_teenyc should locate teenyc");
        let cache_dir = teeny_compiler::compiler::default_cache_dir();
        let compiler = LlvmCompiler::new(teenyc_path, cache_dir)
            .expect("construct LlvmCompiler")
            .with_target_cpu(target.capability.to_string())
            .with_log_level(LogLevel::Debug);

        let captured = CapturingWriter::default();
        let writer_for_subscriber = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(move || writer_for_subscriber.clone())
            .finish();

        // Scoped to this thread only, so it doesn't clobber a subscriber another
        // test running concurrently in this binary may have installed. `force:
        // true` guarantees `teenyc` actually runs (a cache hit would emit nothing).
        let logs = tracing::subscriber::with_default(subscriber, || {
            compiler
                .compile(&kernel, &target, true)
                .expect("compile should succeed");
            String::from_utf8_lossy(&captured.0.lock().unwrap()).into_owned()
        });

        assert!(
            logs.contains("tt.") || logs.contains("ttg."),
            "expected Triton dialect ops (ttir/ttgpuir) in the logged pipeline stages:\n{logs}"
        );
        assert!(
            logs.contains("define ") || logs.contains("llvm.func"),
            "expected LLVM IR in the logged pipeline stages:\n{logs}"
        );
        assert!(
            logs.contains(".visible .entry") || logs.contains(".version"),
            "expected PTX/ASM in the logged pipeline stages:\n{logs}"
        );
    }
}
