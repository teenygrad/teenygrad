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

//! [`schedule_graph`] — Welder §3.2's `GraphConnecting` (Fig. 7), under a
//! better name: `GraphConnecting` reads like a type, not an action
//! (teenygrad-1nr.5).
//!
//! For every edge, in topological node order, tries every memory level
//! declared in a [`HardwareProfile`], tentatively [`TileGraph::set_connect`]s
//! it there, extracts the resulting fused subgraph
//! ([`TileGraph::extract_subgraph`]) and scores it with a [`Profiler`], and
//! leaves the edge connected at whichever level scored lowest (best). The
//! "schedule" this produces is the resulting `connect_level` left on every
//! edge, mutated in place.
//!
//! ## Deviation from the paper's `Min(d.Profile(configs))`
//!
//! Welder profiles every one of `SubGraphTiling`'s returned candidate
//! `TileConfig`s for a level and takes the best. [`Profiler`] here doesn't
//! (yet) take a [`TileConfig`] — it scores a node set structurally (see its
//! own doc comment) — so this asks [`TileGraph::sub_graph_tiling`] for only
//! its single best-ranked candidate (`top_k = 1`) per level, and profiles
//! the *structural* cost of the extracted subgraph at that level instead.
//! That signal still responds correctly to which level is being tried
//! (different levels change both which nodes get extracted and each
//! boundary edge's own bandwidth lookup), so it's a reasoned, working
//! stand-in, not a silent gap — teaching `Profiler` to score a specific
//! `TileConfig` directly is a real follow-up.
//!
//! ## Scope: scheduling only
//!
//! This does not rewrite the DAG into actually-fused kernels — turning a
//! finished schedule into that is blocked on reworking `#[tiled_kernel]` to
//! compose `Tile<D>` functions instead of pointer-in/pointer-out kernels
//! (teenygrad-1nr.1; `pid` decode doesn't currently compose across a fused
//! call). `Anduin::optimize` still needs that piece before it can return a
//! rewritten `Dag` from a schedule produced here.

use teeny_core::device::hardware::HardwareProfile;

use crate::errors::Result;
use crate::graph::EdgeId;

use super::profile::Profiler;
use super::tile_graph::{SubGraphTilingResult, TileGraph};

/// Number of ranked candidates `schedule_graph` asks
/// [`TileGraph::sub_graph_tiling`] for at each candidate memory level.
/// Only the best (rank 0) is used to score the level — see the module doc
/// comment on why a single candidate is enough for this outer search.
const TOP_K: usize = 1;

/// Welder §3.2's `GraphConnecting` (Fig. 7) — see the module doc comment.
/// Mutates `tile_graph` in place: the schedule is the resulting
/// `connect_level` on every edge, plus the winning
/// [`SubGraphTilingResult`] recorded per edge via
/// [`TileGraph::record_resolved_tiling`] (readable back through
/// [`TileGraph::resolved_tiling`]) — the concrete tile shapes that scoring
/// was based on, not just which memory level won. An earlier version of
/// this function computed that result and then discarded it, keeping only
/// `best_level`; §3.3's `Trace::trace_graph` needs the shapes too, so it's
/// cached here (the winning level's `sub_graph_tiling` call is already
/// being made regardless) rather than recomputed.
pub fn schedule_graph(
    tile_graph: &mut TileGraph,
    hardware: &HardwareProfile,
    profiler: &dyn Profiler,
) -> Result<()> {
    let lowest_level = hardware.memory_levels.first().unwrap().kind;

    // Initialize all edges to the lowest level
    for edge_id in 0..tile_graph.num_edges() {
        tile_graph.set_connect(EdgeId(edge_id), lowest_level);
    }

    for node in tile_graph.topological_sort() {
        let edges: Vec<_> = tile_graph
            .children(node)
            .into_iter()
            .map(|(_, id)| id)
            .collect();

        for edge_id in edges {
            let mut best_level = tile_graph.connect_level(edge_id);
            let mut best_latency = f64::INFINITY;
            let mut best_result: Option<SubGraphTilingResult> = None;

            for memory_level in &hardware.memory_levels {
                let level = memory_level.kind;
                tile_graph.set_connect(edge_id, level);

                let subgraph = tile_graph.extract_subgraph(node, None);
                let mut candidates =
                    tile_graph.sub_graph_tiling(&subgraph, node, None, hardware, TOP_K)?;
                if candidates.is_empty() {
                    continue;
                }

                let latency = profiler.profile(tile_graph, &subgraph, hardware);
                if latency < best_latency {
                    best_latency = latency;
                    best_level = level;
                    best_result = Some(candidates.remove(0));
                }
            }

            tile_graph.set_connect(edge_id, best_level);
            if let Some(result) = best_result {
                tile_graph.record_resolved_tiling(edge_id, result);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use teeny_core::graph::{DtypeRepr, Graph, Op};
    use teeny_core::model::LoweringMode;

    use crate::graph::TritonLowering;
    use crate::graph::optimizer::anduin::Anduin;
    use crate::testing::hardware_profile::orin_nano;

    #[test]
    fn test_schedule_pointwise_ops() {
        let mut graph = Graph::new();
        let shape = vec![Some(2048), Some(4096)];

        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
        let _silu = graph.add_node(Op::Silu, vec![relu], DtypeRepr::F32, shape.clone());

        // Anchor the "won't fit on a single SM" claim above against a real
        // two-level hardware profile: the full [2048, 4096] F32 tile is
        // bigger than shared memory but comfortably smaller than device
        // memory.
        let profile = orin_nano();

        let lowering = TritonLowering::default();
        let (dag, _, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .unwrap();

        let (_, traces) = Anduin::schedule(&dag, &profile).unwrap();
        eprintln!("traces: {:?}", traces);
        todo!("test schedule");
    }
}
