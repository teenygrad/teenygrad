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

//! `propagate_graph` — Welder's `Propagate` (OSDI'23), walking a [`Graph`]
//! backward from a chosen output tile shape (teenygrad-3w0.8).
//!
//! Builds on [`teeny_triton::propagate_within_kernel`], which does the real
//! work *within* one kernel's own declared `KernelTileSpec`. This module
//! only adds the graph walk on top: at each node, resolve that node's own
//! per-input requirements, then recurse into whichever producer node feeds
//! each input.
//!
//! **Coverage is intentionally conservative.** `tile_spec()` coverage across
//! this codebase is sparse (as of teenygrad-3w0.8: `elu_forward`,
//! `matmul_forward`, `conv2d_forward`, `flash_attention2_forward` — no
//! common op like `Op::Relu`/`Op::Add` has one), and there is no existing
//! mapping from a graph edge (`GraphNode.inputs[i]`) to a specific
//! `KernelTileSpec.inputs[j]` entry — only the *convention* (shared by every
//! `RuntimeOp::pack_args` impl in this tree) that both are ordered by
//! declaration order. So this pass degrades to a hard boundary (stops,
//! doesn't guess or panic) whenever:
//! - `tile_spec_for_op` returns `None` for a node's op (no spec available), or
//! - a node's `inputs.len()` doesn't match its spec's `inputs.len()`
//!   (positional correspondence isn't safely assumable).

use std::collections::HashMap;

use teeny_core::graph::{Graph, Op};
use teeny_triton::{KernelTileSpec, propagate_within_kernel};

/// Walk `graph` backward from `target_idx`, propagating `chosen_output`
/// (concrete sizes for `output_param`'s axes, in declared axis order) into
/// every reachable ancestor's own resolved `extent_param -> size` map.
///
/// `tile_spec_for_op` resolves an arbitrary `Op` to its `KernelTileSpec`, if
/// one is declared — caller-supplied rather than a built-in global lookup,
/// since no such registry exists yet and building one now would either be
/// mostly-empty or overclaim coverage (see the module doc).
///
/// Returns one entry per node index actually reached (including
/// `target_idx` itself); a node with no tile spec, or whose input count
/// doesn't align with its spec, is the last entry on its path — its
/// producers are never visited.
pub fn propagate_graph(
    graph: &Graph,
    target_idx: usize,
    output_param: &str,
    chosen_output: &[i64],
    tile_spec_for_op: impl Fn(&Op) -> Option<KernelTileSpec>,
) -> HashMap<usize, HashMap<&'static str, i64>> {
    let mut results: HashMap<usize, HashMap<&'static str, i64>> = HashMap::new();
    // Work-list carries the *required shape* for a node's own output. The
    // target's output *name* comes from the caller (`output_param`); every
    // other node's output name is its own business, not something the
    // consumer's edge can dictate — a producer names its own output
    // whatever it likes (e.g. Relu's `y_ptr` feeding Sigmoid's `x_ptr` is
    // the same tensor under two different kernel-local names), so we look
    // it up from the *producer's own* spec once we get there, not from the
    // consumer's `TensorTileSpec::param`. `None` marks "look it up"; `Some`
    // is only ever the seed for `target_idx`.
    let mut work: Vec<(usize, Option<String>, Vec<i64>)> = vec![(
        target_idx,
        Some(output_param.to_string()),
        chosen_output.to_vec(),
    )];

    while let Some((idx, out_param_override, out_shape)) = work.pop() {
        let Some(spec) = tile_spec_for_op(&graph.nodes[idx].op) else {
            continue; // hard boundary: no declared tile shape for this op
        };
        let out_param: &str = match &out_param_override {
            Some(p) => p,
            None => {
                // Producer node: use its own sole declared output. A
                // multi-output producer is a hard boundary too — nothing
                // here says *which* output this edge corresponds to.
                let [only_output] = spec.outputs else {
                    continue;
                };
                only_output.param
            }
        };
        let resolved = propagate_within_kernel(&spec, out_param, &out_shape);
        results.insert(idx, resolved.iter().map(|(&k, &v)| (k, v)).collect());

        let node_inputs = &graph.nodes[idx].inputs;
        if node_inputs.len() != spec.inputs.len() {
            continue; // hard boundary: positional correspondence unsafe
        }
        for (&producer_idx, input_spec) in node_inputs.iter().zip(spec.inputs.iter()) {
            // This input's own required shape, in its own axis order —
            // looked up from what we just resolved for its extent_param
            // names. An axis Propagate couldn't resolve (e.g. a reduction
            // axis with no output-side counterpart) has no entry here
            // either; skip this producer rather than push a bogus shape.
            let Some(producer_output_shape): Option<Vec<i64>> = input_spec
                .axes
                .iter()
                .map(|axis| resolved.get(axis.extent_param).copied())
                .collect()
            else {
                continue;
            };
            work.push((producer_idx, None, producer_output_shape));
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use teeny_core::graph::{DtypeRepr, SymTensor};
    use teeny_triton::{KernelTileSpec, TensorTileSpec, TileAxisBinding};

    fn shape_1d(n: usize) -> Vec<Option<usize>> {
        vec![None, Some(n)]
    }

    /// `x -> Relu -> Sigmoid` — a 3-node graph (Input, Relu, Sigmoid), same
    /// `SymTensor`/`add_node` construction every other graph test in this
    /// tree uses (e.g. `test_anduin_tile_fuse.rs`).
    fn build_relu_sigmoid_graph() -> Graph {
        let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(64));
        let relu = input.graph.borrow_mut().add_node(
            Op::Relu,
            vec![input.node_id],
            DtypeRepr::F32,
            shape_1d(64),
        );
        let _ = input.graph.borrow_mut().add_node(
            Op::Sigmoid,
            vec![relu],
            DtypeRepr::F32,
            shape_1d(64),
        );
        drop(input);
        Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
    }

    /// A flat, elu-shaped `KernelTileSpec` any unary op can share for this
    /// test — real coverage (only `elu_forward` et al.) doesn't include
    /// `Relu`/`Sigmoid`, so the lookup below is a test-local stand-in, not a
    /// claim that these ops have real specs today.
    fn flat_unary_spec() -> KernelTileSpec {
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 1,
            axes: &[TileAxisBinding {
                dim: 0,
                block_const: "BLOCK_SIZE",
                extent_param: "n_elements",
                window: None,
            }],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
            loop_spec: None,
        }
    }

    #[test]
    fn propagate_graph_resolves_across_two_hops() {
        let graph = build_relu_sigmoid_graph();
        // nodes: [0]=Input, [1]=Relu, [2]=Sigmoid (target)
        let sigmoid_idx = graph.nodes.len() - 1;

        let results = propagate_graph(&graph, sigmoid_idx, "y_ptr", &[64], |op| {
            matches!(op, Op::Relu | Op::Sigmoid).then(flat_unary_spec)
        });

        // Sigmoid and Relu both have a spec, so both should be reached and
        // resolve n_elements=64; Input (op Op::Input) has no spec, so the
        // walk stops there (present only as Relu's unresolved producer, not
        // in `results`).
        assert_eq!(results.len(), 2, "Sigmoid + Relu; Input is a hard boundary");
        assert_eq!(results[&sigmoid_idx].get("n_elements"), Some(&64));
        let relu_idx = sigmoid_idx - 1;
        assert_eq!(results[&relu_idx].get("n_elements"), Some(&64));
    }

    #[test]
    fn propagate_graph_stops_at_first_node_without_a_tile_spec() {
        let graph = build_relu_sigmoid_graph();
        let sigmoid_idx = graph.nodes.len() - 1;

        // Only Sigmoid has a spec in this lookup -- Relu doesn't, so the
        // walk must stop there without panicking or guessing.
        let results = propagate_graph(&graph, sigmoid_idx, "y_ptr", &[64], |op| {
            matches!(op, Op::Sigmoid).then(flat_unary_spec)
        });

        assert_eq!(results.len(), 1);
        assert!(results.contains_key(&sigmoid_idx));
    }

    #[test]
    fn propagate_graph_target_with_no_spec_returns_empty() {
        let graph = build_relu_sigmoid_graph();
        let sigmoid_idx = graph.nodes.len() - 1;
        let results = propagate_graph(&graph, sigmoid_idx, "y_ptr", &[64], |_op| None);
        assert!(results.is_empty());
    }
}
