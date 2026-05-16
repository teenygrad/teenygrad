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

use std::rc::Rc;

use teeny_core::{
    graph::{DtypeRepr, Graph, Op, SymTensor},
    model::LoweringMode,
};
use teeny_kernels::graph::TritonLowering;

/// Build: `Input → Conv2d(no bias) → BatchNorm2d → Silu`
fn build_conv_bn_silu_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(1), Some(3), Some(8), Some(8)],
    );

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
    let _ = input.graph.borrow_mut().add_node(
        Op::Silu,
        vec![bn_id],
        DtypeRepr::F32,
        conv_shape,
    );

    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[test]
fn test_conv_bn_silu_graph_has_four_nodes() {
    let graph = build_conv_bn_silu_graph();
    assert_eq!(graph.nodes.len(), 4, "expected Input + Conv2d + BN + Silu");
    assert!(matches!(graph.nodes[0].op, Op::Input));
    assert!(matches!(graph.nodes[1].op, Op::Conv2d { has_bias: false, .. }));
    assert!(matches!(graph.nodes[2].op, Op::BatchNorm2d { .. }));
    assert!(matches!(graph.nodes[3].op, Op::Silu));
}

#[test]
fn test_optimise_fuses_conv_bn_silu() {
    let graph = build_conv_bn_silu_graph();
    let opt = graph.optimise();

    assert_eq!(opt.nodes.len(), 2, "expected Input + Conv2dBnSilu after fusion");
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

    assert_eq!(fused_shape, &original_output_shape,
        "fused node shape must match original Silu output shape");
}

#[test]
fn test_optimise_rewires_inputs() {
    let graph = build_conv_bn_silu_graph();
    let opt = graph.optimise();

    // The fused node should take the Input node (index 0) as its only input.
    assert_eq!(opt.nodes[1].inputs, vec![0],
        "Conv2dBnSilu should consume the Input node directly");
}

#[test]
fn test_lowering_produces_fused_kernel() {
    let graph = build_conv_bn_silu_graph();
    let opt = graph.optimise();

    let lowering = TritonLowering::new();
    let (dag, _mapping) = lowering
        .lower_with_mapping(&opt, LoweringMode::Inference)
        .expect("lowering should succeed");

    assert_eq!(dag.len(), 2, "DAG should have Input node + fused kernel node");

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
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(1), Some(3), Some(8), Some(8)],
    );
    let conv_shape = vec![Some(1), Some(16), Some(8), Some(8)];
    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: 3, out_channels: 16, kernel_h: 3, kernel_w: 3,
            stride_h: 1, stride_w: 1, padding_h: 1, padding_w: 1,
            groups: 1,
            has_bias: true,  // bias present — should NOT fuse
        },
        vec![input.node_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d { num_features: 16, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true },
        vec![conv_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let _ = input.graph.borrow_mut().add_node(
        Op::Silu,
        vec![bn_id],
        DtypeRepr::F32,
        conv_shape,
    );

    drop(input);
    let graph = Rc::try_unwrap(graph_rc).ok().unwrap().into_inner();

    let opt = graph.optimise();
    // No fusion; all 4 nodes survive.
    assert_eq!(opt.nodes.len(), 4,
        "Conv2d-with-bias should not be fused");
    assert!(!matches!(opt.nodes.last().unwrap().op, Op::Conv2dBnSilu { .. }),
        "last node should still be Silu, not Conv2dBnSilu");
}

#[test]
fn test_optimise_no_fusion_when_conv_has_multiple_consumers() {
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(1), Some(3), Some(8), Some(8)],
    );
    let conv_shape = vec![Some(1), Some(16), Some(8), Some(8)];
    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: 3, out_channels: 16, kernel_h: 3, kernel_w: 3,
            stride_h: 1, stride_w: 1, padding_h: 1, padding_w: 1,
            groups: 1, has_bias: false,
        },
        vec![input.node_id],
        DtypeRepr::F32,
        conv_shape.clone(),
    );
    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d { num_features: 16, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true },
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
    let _ = input.graph.borrow_mut().add_node(
        Op::Silu,
        vec![bn_id],
        DtypeRepr::F32,
        conv_shape,
    );

    drop(input);
    let graph = Rc::try_unwrap(graph_rc).ok().unwrap().into_inner();

    let opt = graph.optimise();
    // Conv has 2 consumers — must not fuse.
    assert!(!opt.nodes.iter().any(|n| matches!(n.op, Op::Conv2dBnSilu { .. })),
        "should not fuse when Conv2d has multiple consumers");
}
