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
    PointwiseFuse, ReduceFuse, SharedTransposeFuse, TileFuse, choose_reduce_fuse_block_inner,
    choose_shared_transpose_fuse_block_size, elem_bytes, is_bool_terminal_only,
    is_pointwise_fuse_dtype, is_reduce_fuse_member, is_reduce_fuse_reducible, is_tile_fuse_tail,
    probe_pointwise_op,
};

/// Anduin: Triton-side graph rewrites before lowering.
///
/// Current strategies (run in order, each to its own fixpoint):
/// - fuse linear unary pointwise chains into [`PointwiseFuse`] custom ops
/// - fuse a unary chain + a second, unchained input through a binary tail op
///   into [`TileFuse`] custom ops (teenygrad-3w0's fan-in case — see its
///   module doc) — tried second so it can pick up `PointwiseFuse` chains the
///   first pass already built as its branch
/// - fuse a unary chain feeding directly into a row-reduction into
///   [`ReduceFuse`] custom ops (teenygrad-3w0.9's case-4 fusion — see its
///   module doc) — tried after `PointwiseFuse` chains exist to pick up as
///   the reduction's ancestor; the fused kernel's CTA size (`BLOCK_INNER`)
///   is chosen by a real cost-driven search over the calibrated
///   `CostModel`/`mem_traffic` (teenygrad-3w0.11's
///   `choose_reduce_fuse_block_inner`), not a fixed constant — Welder's
///   `SubGraphTiling` tile-size search, applied to the one fusion op with a
///   clean, closed-form cost story
/// - fuse a unary chain feeding directly into a rank-2 transpose into
///   [`SharedTransposeFuse`] custom ops (teenygrad-3w0.10's `SetConnect`
///   demonstration — see its module doc) — tried last; the shared-memory
///   tier: `T::trans` already stages through shared memory transparently,
///   so this splices the chain into registers ahead of it rather than
///   materializing to global memory first. Tile shape (`BLOCK_M`,
///   `BLOCK_N`) is chosen by a real cost-driven search over the calibrated
///   `CostModel`, including the shared-memory-capacity term — Welder's
///   `SubGraphTiling` search again, this time genuinely two-dimensional
///   (grid size *and* per-CTA shared-memory footprint both vary with the
///   tile shape)
///
/// Conv/GEMM epilogue fusion (teenygrad-1bf.8) is **not** done via hand-written
/// fused kernels; that path was removed. Case 8 stays a hard boundary against
/// merging conv/GEMM main loops with [`PointwiseFuse`] / [`TileFuse`].
///
/// `PointwiseFuse`/`TileFuse` deliberately stay on today's fixed-`1024`
/// heuristic (teenygrad-3w0.11 non-goal): their block size isn't a local
/// constant to parameterize — it's hardcoded table-wide across every unary
/// op's entry in `graph/mod.rs`'s `Op` → kernel lowering match, and their
/// grid genuinely varies with block size (unlike `ReduceFuse`'s fixed
/// `[n_outer, 1, 1]` grid), making it a non-closed-form search — a
/// materially bigger, separate effort.
#[derive(Debug, Default, Clone, Copy)]
pub struct Anduin;

impl GraphOptimizer for Anduin {
    fn name(&self) -> &str {
        "anduin"
    }

    fn optimize(&self, graph: &Graph) -> Result<Graph> {
        let graph = fuse_pointwise_chains(graph);
        let graph = fuse_fan_in_chains(&graph);
        let graph = fuse_reduction_chains(&graph);
        Ok(fuse_transpose_chains(&graph))
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
    (rebuild_graph(graph, &dead, &node_override), true)
}

/// Compact `graph`, dropping `dead` nodes and applying `node_override`s
/// (new op + new — old-indexed — inputs) to the survivors, remapping every
/// input reference to the new, compacted indices. Shared by both Anduin
/// passes ([`fuse_pointwise_chain_pass`], [`fuse_fan_in_pass`]).
fn rebuild_graph(
    graph: &Graph,
    dead: &[bool],
    node_override: &[Option<(Op, Vec<usize>)>],
) -> Graph {
    let n = graph.nodes.len();
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

    new_graph
}

/// Collapse a fan-in `(unary chain, second input) -> binary tail` region into
/// [`TileFuse`] custom ops — teenygrad-3w0's case-2 fusion (e.g. `relu(x) + z`).
fn fuse_fan_in_chains(graph: &Graph) -> Graph {
    let mut graph = graph.clone();
    loop {
        let (next, changed) = fuse_fan_in_pass(&graph);
        graph = next;
        if !changed {
            break;
        }
    }
    graph
}

fn fuse_fan_in_pass(graph: &Graph) -> (Graph, bool) {
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

    #[allow(clippy::needless_range_loop)]
    for tail_idx in 0..n {
        if changed {
            break;
        }
        let tail_op = graph.nodes[tail_idx].op.clone();
        if !is_tile_fuse_tail(&tail_op) {
            continue;
        }
        if graph.nodes[tail_idx].inputs.len() != 2 {
            continue;
        }
        let tail_dtype = graph.nodes[tail_idx].dtype;
        if !is_pointwise_fuse_dtype(tail_dtype) {
            continue;
        }
        let lhs = graph.nodes[tail_idx].inputs[0];
        let rhs = graph.nodes[tail_idx].inputs[1];

        // Try each side as the unary-chain branch, the other as the
        // unchained second input. `Op::Add` is commutative so either
        // orientation is a valid fusion; pick whichever side actually
        // qualifies as a single-consumer pointwise chain.
        for (chain_idx, pass_idx) in [(lhs, rhs), (rhs, lhs)] {
            if n_consumers[chain_idx] != 1 {
                continue;
            }
            if graph.nodes[chain_idx].inputs.len() != 1 {
                continue;
            }
            let chain_dtype = graph.nodes[chain_idx].dtype;
            if chain_dtype != tail_dtype {
                continue;
            }
            let Some((branch_members, branch_probe)) =
                pointwise_parts(&graph.nodes[chain_idx].op, chain_dtype)
            else {
                continue;
            };
            if branch_members.is_empty() {
                continue;
            }

            let fused = TileFuse::new(branch_members, tail_op.clone(), tail_dtype, branch_probe);
            dead[chain_idx] = true;
            node_override[tail_idx] = Some((
                Op::Custom {
                    data: CustomData::new(fused),
                },
                vec![graph.nodes[chain_idx].inputs[0], pass_idx],
            ));
            changed = true;
            break;
        }
    }

    if !changed {
        return (graph.clone(), false);
    }
    (rebuild_graph(graph, &dead, &node_override), true)
}

/// Expand a node into reduce-fusable chain members, if every member is in
/// [`is_reduce_fuse_member`]'s set — narrower than [`pointwise_parts`]'s
/// general pointwise-fusable set (`ReduceFuse` v1 only splices zero-extra-
/// param ops; see `reduce_fuse.rs`'s module doc).
fn reduce_chain_parts(op: &Op) -> Option<Vec<Op>> {
    match op {
        Op::Custom { data } => {
            let pf = data.downcast_ref::<PointwiseFuse>()?;
            pf.members
                .iter()
                .all(is_reduce_fuse_member)
                .then(|| pf.members.clone())
        }
        other if is_reduce_fuse_member(other) => Some(vec![other.clone()]),
        _ => None,
    }
}

/// Collapse a `(unary chain) -> reduction` region into [`ReduceFuse`] custom
/// ops — teenygrad-3w0.9's case-4 fusion (e.g. `reduce_sum(relu(x))`).
fn fuse_reduction_chains(graph: &Graph) -> Graph {
    let mut graph = graph.clone();
    loop {
        let (next, changed) = fuse_reduction_chain_pass(&graph);
        graph = next;
        if !changed {
            break;
        }
    }
    graph
}

fn fuse_reduction_chain_pass(graph: &Graph) -> (Graph, bool) {
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

    #[allow(clippy::needless_range_loop)]
    for reduce_idx in 0..n {
        if changed {
            break;
        }
        let reduce_op = graph.nodes[reduce_idx].op.clone();
        if !is_reduce_fuse_reducible(&reduce_op) {
            continue;
        }
        if graph.nodes[reduce_idx].inputs.len() != 1 {
            continue;
        }
        let reduce_dtype = graph.nodes[reduce_idx].dtype;
        if !is_pointwise_fuse_dtype(reduce_dtype) {
            continue;
        }
        let chain_idx = graph.nodes[reduce_idx].inputs[0];
        if n_consumers[chain_idx] != 1 {
            continue;
        }
        let chain_dtype = graph.nodes[chain_idx].dtype;
        if chain_dtype != reduce_dtype {
            continue;
        }
        let Some(members) = reduce_chain_parts(&graph.nodes[chain_idx].op) else {
            continue;
        };

        // teenygrad-3w0.11: cost-driven BLOCK_INNER, not a fixed constant.
        // The chain's own output shape is the reduction's input shape,
        // `[..., n_inner]` (batch/`n_outer` dynamic, mirroring `graph/mod.rs`'s
        // `pick_gemm_tile_sizes` shape-reading convention) -- no shape, no
        // fusion, same as any other failed eligibility check here.
        let Some(n_inner) = graph.nodes[chain_idx].shape.last().copied().flatten() else {
            continue;
        };
        let Some(block_inner) = choose_reduce_fuse_block_inner(n_inner as i64) else {
            continue;
        };

        let fused = ReduceFuse::new(members, reduce_op, reduce_dtype, block_inner);
        dead[chain_idx] = true;
        node_override[reduce_idx] = Some((
            Op::Custom {
                data: CustomData::new(fused),
            },
            graph.nodes[chain_idx].inputs.clone(),
        ));
        changed = true;
    }

    if !changed {
        return (graph.clone(), false);
    }
    (rebuild_graph(graph, &dead, &node_override), true)
}

/// True when `op` is `Op::Transpose` with a `perm` `SharedTransposeFuse`
/// v1 can terminate a chain with — rank-2 only, same limitation
/// `transpose_2d_forward`/the free `Op::Transpose` lowering both already
/// enforce.
fn is_shared_transpose_fuse_terminal(op: &Op) -> bool {
    matches!(op, Op::Transpose { perm } if perm.is_empty() || perm.as_slice() == [1, 0])
}

/// Collapse a `(unary chain) -> rank-2 transpose` region into
/// [`SharedTransposeFuse`] custom ops — teenygrad-3w0.10's `SetConnect`
/// demonstration (e.g. `transpose(relu(x))`).
fn fuse_transpose_chains(graph: &Graph) -> Graph {
    let mut graph = graph.clone();
    loop {
        let (next, changed) = fuse_transpose_chain_pass(&graph);
        graph = next;
        if !changed {
            break;
        }
    }
    graph
}

fn fuse_transpose_chain_pass(graph: &Graph) -> (Graph, bool) {
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

    #[allow(clippy::needless_range_loop)]
    for transpose_idx in 0..n {
        if changed {
            break;
        }
        let transpose_op = graph.nodes[transpose_idx].op.clone();
        if !is_shared_transpose_fuse_terminal(&transpose_op) {
            continue;
        }
        if graph.nodes[transpose_idx].inputs.len() != 1 {
            continue;
        }
        let transpose_dtype = graph.nodes[transpose_idx].dtype;
        if !is_pointwise_fuse_dtype(transpose_dtype) {
            continue;
        }
        let chain_idx = graph.nodes[transpose_idx].inputs[0];
        if n_consumers[chain_idx] != 1 {
            continue;
        }
        let chain_dtype = graph.nodes[chain_idx].dtype;
        if chain_dtype != transpose_dtype {
            continue;
        }
        let Some(members) = reduce_chain_parts(&graph.nodes[chain_idx].op) else {
            continue;
        };

        // teenygrad-3w0.10: cost-driven (BLOCK_M, BLOCK_N), and only when
        // both M and N are statically known (rank-2) and some candidate
        // tile evenly divides both -- transpose_2d_forward's own alignment
        // requirement, inherited unchanged by the fused kernel.
        let shape = &graph.nodes[chain_idx].shape;
        let (Some(Some(m)), Some(Some(n))) = (shape.first(), shape.get(1)) else {
            continue;
        };
        let Some(elem_bytes) = elem_bytes(transpose_dtype) else {
            continue;
        };
        let Some((block_m, block_n)) =
            choose_shared_transpose_fuse_block_size(*m as i64, *n as i64, elem_bytes)
        else {
            continue;
        };

        let fused = SharedTransposeFuse::new(members, transpose_dtype, block_m, block_n);
        dead[chain_idx] = true;
        node_override[transpose_idx] = Some((
            Op::Custom {
                data: CustomData::new(fused),
            },
            graph.nodes[chain_idx].inputs.clone(),
        ));
        changed = true;
    }

    if !changed {
        return (graph.clone(), false);
    }
    (rebuild_graph(graph, &dead, &node_override), true)
}
