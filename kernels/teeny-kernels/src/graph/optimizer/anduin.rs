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

//! Anduin — first Triton graph optimizer

use teeny_core::graph::{CustomData, DtypeRepr, Graph, GraphNode, Op};
use teeny_triton::PointwiseFuseProbe;

use crate::errors::Result;
use crate::graph::optimizer::GraphOptimizer;
use crate::graph::optimizer::ops::{
    PointwiseFuse, is_bool_terminal_only, is_pointwise_fuse_dtype, probe_pointwise_op,
};

/// Anduin: Triton-side graph rewrites before lowering.
///
/// Current strategies:
/// - fuse linear unary pointwise chains into [`PointwiseFuse`] custom ops
#[derive(Debug, Default, Clone, Copy)]
pub struct Anduin;

impl GraphOptimizer for Anduin {
    fn name(&self) -> &str {
        "anduin"
    }

    fn optimize(&self, graph: &Graph) -> Result<Graph> {
        Ok(fuse_pointwise_chains(graph))
    }
}

/// Collapse adjacent single-consumer pointwise ops into [`PointwiseFuse`].
fn fuse_pointwise_chains(graph: &Graph) -> Graph {
    let mut graph = graph.clone();
    loop {
        let (next, changed) = fuse_pointwise_chain_pass(&graph);
        graph = next;
        if !changed {
            break;
        }
    }
    graph
}

/// Expand a node into pointwise members + shared probe, if fusible.
fn pointwise_parts(op: &Op, dtype: DtypeRepr) -> Option<(Vec<Op>, PointwiseFuseProbe)> {
    match op {
        Op::Custom { data } => data
            .downcast_ref::<PointwiseFuse>()
            .map(|pf| (pf.members.clone(), pf.probe)),
        other => {
            let probe = probe_pointwise_op(other, dtype)?;
            Some((vec![other.clone()], probe))
        }
    }
}

fn fuse_pointwise_chain_pass(graph: &Graph) -> (Graph, bool) {
    let n = graph.nodes.len();

    let mut n_consumers = vec![0usize; n];
    for node in &graph.nodes {
        for &inp in &node.inputs {
            n_consumers[inp] += 1;
        }
    }

    let mut dead = vec![false; n];
    let mut node_override: Vec<Option<(Op, Vec<usize>)>> = vec![None; n];
    let mut changed = false;

    // `child_idx` indexes multiple parallel collections.
    #[allow(clippy::needless_range_loop)]
    for child_idx in 0..n {
        if changed {
            break;
        }
        let child_dtype = graph.nodes[child_idx].dtype;
        let Some((child_members, child_probe)) =
            pointwise_parts(&graph.nodes[child_idx].op, child_dtype)
        else {
            continue;
        };
        if child_members.is_empty() {
            continue;
        }
        if graph.nodes[child_idx].inputs.len() != 1 {
            continue;
        }
        let parent_idx = graph.nodes[child_idx].inputs[0];
        if n_consumers[parent_idx] != 1 {
            continue;
        }

        let parent_dtype = graph.nodes[parent_idx].dtype;
        // A fused chain lowers every member through one shared dtype
        // (PointwiseFuse::dtype); parent and child must already agree, and
        // that dtype must be one PointwiseFuse can actually emit -- otherwise
        // this either silently drops the parent's real dtype or panics later
        // in PointwiseFuse::lower() on an unsupported dtype.
        if parent_dtype != child_dtype || !is_pointwise_fuse_dtype(child_dtype) {
            continue;
        }
        let Some((mut members, parent_probe)) =
            pointwise_parts(&graph.nodes[parent_idx].op, parent_dtype)
        else {
            continue;
        };
        if !parent_probe.compatible(child_probe) {
            continue;
        }
        // Append child's members (single op or an existing PointwiseFuse).
        members.extend(child_members);
        if members.len() < 2 {
            continue;
        }
        // Bool-producing ops (IsNaN, IsInf) change the element type mid-chain;
        // a later member reading their output as the chain's float dtype
        // would be wrong, so they may only be the chain's last member.
        if members[..members.len() - 1]
            .iter()
            .any(is_bool_terminal_only)
        {
            continue;
        }

        let fused = PointwiseFuse::new(members, child_dtype, parent_probe);

        dead[parent_idx] = true;
        node_override[child_idx] = Some((
            Op::Custom {
                data: CustomData::new(fused),
            },
            graph.nodes[parent_idx].inputs.clone(),
        ));
        changed = true;
    }

    if !changed {
        return (graph.clone(), false);
    }

    let mut old_to_new = vec![0usize; n];
    let mut new_count = 0usize;
    for i in 0..n {
        if !dead[i] {
            old_to_new[i] = new_count;
            new_count += 1;
        }
    }

    let mut new_graph = Graph::new();
    for old_idx in 0..n {
        if dead[old_idx] {
            continue;
        }
        let node = &graph.nodes[old_idx];
        let (op, inputs) = if let Some((fused_op, fused_inputs)) = node_override[old_idx].clone() {
            let mapped = fused_inputs.iter().map(|&i| old_to_new[i]).collect();
            (fused_op, mapped)
        } else {
            let mapped = node.inputs.iter().map(|&i| old_to_new[i]).collect();
            (node.op.clone(), mapped)
        };
        let new_idx = new_graph.nodes.len();
        new_graph.nodes.push(GraphNode {
            op,
            inputs,
            dtype: node.dtype,
            shape: node.shape.clone(),
        });
        if let Some(name) = graph.names.get(&old_idx) {
            new_graph.names.insert(new_idx, name.clone());
        }
    }

    (new_graph, true)
}
