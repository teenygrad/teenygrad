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

//! Guards the other half of the tiling epic's contract (teenygrad-39jd).
//!
//! `tests/test_tile_declarations.rs` checks that a converted *kernel* declares
//! its axes. This checks that the *graph layer* actually hands that declaration
//! on, because a kernel can describe itself perfectly while `TritonLowering`
//! drops the spec and leaves the op a scheduling boundary — which is exactly
//! what `exec_from` did when it hardcoded `tile_spec = None`, for eighteen of
//! the twenty flat elementwise activations.
//!
//! Without this test the next conversion rung can silently re-open that gap:
//! nothing else fails when a lowered op's `tile_spec()` is `None`, it just
//! quietly stops fusing.

use teeny_core::graph::{DtypeRepr, Graph, op::Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// Every flat elementwise activation that routes through `exec_from`, with a
/// representative parameterisation for the ones carrying fields.
fn flat_elementwise_ops() -> Vec<(&'static str, Op)> {
    vec![
        ("Elu", Op::Elu { alpha: 1.0 }),
        ("Selu", Op::Selu),
        ("Celu", Op::Celu { alpha: 1.0 }),
        ("Gelu", Op::Gelu),
        ("Mish", Op::Mish),
        (
            "Hardtanh",
            Op::Hardtanh {
                min_val: -1.0,
                max_val: 1.0,
            },
        ),
        ("Relu6", Op::Relu6),
        ("Hardsigmoid", Op::Hardsigmoid),
        ("Hardswish", Op::Hardswish),
        ("Hardshrink", Op::Hardshrink { lambda: 0.5 }),
        (
            "LeakyRelu",
            Op::LeakyRelu {
                negative_slope: 0.01,
            },
        ),
        (
            "Threshold",
            Op::Threshold {
                threshold: 0.0,
                value: 0.0,
            },
        ),
        ("Softsign", Op::Softsign),
        ("Softshrink", Op::Softshrink { lambda: 0.5 }),
        (
            "Softplus",
            Op::Softplus {
                beta: 1.0,
                threshold: 20.0,
            },
        ),
        ("Sigmoid", Op::Sigmoid),
        ("Silu", Op::Silu),
        ("LogSigmoid", Op::LogSigmoid),
        ("Tanh", Op::Tanh),
        ("Tanhshrink", Op::Tanhshrink),
        // Relu builds its executable through `make_num_kernel!` rather than
        // `exec_from`, so it exercises the other path to the same guarantee.
        ("Relu", Op::Relu),
    ]
}

#[test]
fn test_flat_elementwise_ops_lower_with_a_tile_spec() {
    // Rank 2 deliberately: a flat spec's axis spans every dim, and a rank-1
    // node would let a spec that only ever matches 1-D pass unnoticed — the
    // exact way `RELU_TILE_SPEC` was once silently inert.
    let shape = vec![Some(8), Some(16)];
    let mut missing: Vec<&str> = Vec::new();
    let mut checked = 0usize;

    for (name, op) in flat_elementwise_ops() {
        let mut graph = Graph::new();
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        graph.add_node(op, vec![input], DtypeRepr::F32, shape.clone());

        let dag = TritonLowering::new()
            .lower(&graph, LoweringMode::Inference)
            .unwrap_or_else(|e| panic!("{name}: lowering should not fail: {e}"));

        // Node 0 is the Input; node 1 is the op under test.
        let exec = &dag.node(1).value;
        match exec.tile_spec() {
            None => missing.push(name),
            Some(spec) => {
                checked += 1;
                spec.validate().unwrap_or_else(|e| {
                    panic!("{name}: lowered spec must be self-consistent: {e}")
                });
                assert_eq!(
                    spec.inputs.len(),
                    1,
                    "{name}: a unary elementwise op declares one input"
                );
                for tensor in spec.inputs.iter().chain(spec.outputs.iter()) {
                    assert_eq!(
                        tensor.rank,
                        shape.len(),
                        "{name}: the spec's rank must follow the node's real rank, not 1"
                    );
                }
            }
        }
    }

    assert!(
        checked + missing.len() >= 20,
        "only {} ops were examined; the list has drifted from the `exec_from` call sites",
        checked + missing.len()
    );
    assert!(
        missing.is_empty(),
        "these ops lower without a tile_spec, so TileGraph::propagate treats them as hard \
         boundaries even though their kernels declare axes: {missing:?}\n\n\
         Pass the derived `XForward::<f32>::tile_spec(node.shape.len())` at the `exec_from` \
         call site in graph/mod.rs (teenygrad-39jd).",
    );
}
