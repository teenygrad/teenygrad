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

use std::collections::{BinaryHeap, HashSet};

use teeny_core::device::hardware::{HardwareProfile, MemoryLevel};

use crate::errors::Result;

use super::profile::Profiler;
use super::tile_graph::TileGraph;
use super::tile_graph::node::NodeId;

struct TileConfig {
    config: HashMap<EdgeId, MemoryLevel>,
}

struct Subtile {}

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
    top_k: usize,
) -> Result<()> {
    let lowest_level = hardware
        .memory_levels
        .iter()
        .min_by_key(|level| level.level)?;

    let mut config = TileConfig {
        config: HashMap::new(),
    };

    for node in tile_graph.nodes.iter_mut() {
        for edge in node.out_edges.iter_mut() {
            for level in hardware.memory_levels.iter() {
                edge.set_connect(*level);
                let subgrap = extract_subgraph(tile_graph, node.id, *lowest_level)?;
                todo!()
            }
        }
    }

    todo!()
}

fn subgraph_tiling(
    tile_graph: &TileGraph,
    subgraph: &[NodeId],
    hardware: &HardwareProfile,
    memory_level: MemoryLevel,
    config: &mut TileConfig,
    top_k: usize,
) -> Result<Vec<TileConfig>> {
    let mut configs = TopK::new(top_k);

    for subtile in enumerate_subtiles(tile_graph, subgraph, config)? {
        let config = propagate(tile_graph, subgraph, &subtile)?;
        let footprint = memory_footprint(tile_graph, subgraph)?;
        if footprint > memory_level.capacity {
            continue;
        }
        configs.push((footprint, config));
    }

    let results = Vec::new();
    let next_memory_level = hardware
        .memory_levels
        .iter()
        .filter(|m| m.level > memory_level.level)
        .min_by_key(|m| m.level)?;
    for config in configs.into_sorted_desc() {
        let subgraphs: HashSet<HashSet<NodeId>> = subgraph
            .iter()
            .map(|node| extract_subgraph(tile_graph, *node, *next_memory_level))
            .collect()?;

        for subgraph in subgraphs {
            let g = subgraph.iter().copied().collect::<Vec<_>>();
            let subgraph_configs = subgraph_tiling(
                tile_graph,
                &g,
                hardware,
                *next_memory_level,
                &mut config,
                top_k,
            )?;
            // results[config] = subgraph_config;
            todo!()
        }
    }

    Ok(results)
}

fn extract_subgraph(
    tile_graph: &TileGraph,
    node_id: NodeId,
    memory_level: MemoryLevel,
) -> Result<HashSet<NodeId>> {
    let mut nodes = HashSet::new();
    let node = tile_graph.node(node_id)?;

    for edge in node.in_edges.iter().chain(node.out_edges.iter()) {
        if let Some(edge_memory_level) = edge.memory_level {
            if edge_memory_level.level > memory_level.level {
                let other_node = if edge.src_node_id == node_id {
                    edge.dst_node_id
                } else {
                    edge.src_node_id
                };

                extract_subgraph(tile_graph, other_node, memory_level)?;
                nodes.extend(extract_subgraph);
            }
        }
    }

    Ok(nodes)
}

fn enumerate_subtiles(
    tile_graph: &TileGraph,
    nodes: &[NodeId],
    config: &TileConfig,
) -> Result<Vec<Subtile>> {
    todo!()
}

fn propagate(tile_graph: &TileGraph, nodes: &[NodeId], config: &Subtile) -> Result<TileConfig> {
    todo!()
}

fn memory_footprint(tile_graph: &TileGraph, subgraph: &[NodeId]) -> Result<u64> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TritonLowering;
    use crate::graph::optimizer::anduin::profile::SimpleProfiler;
    use crate::graph::optimizer::anduin::tile_graph::{
        TileGraph,
        node::{Edge, EdgeId, Node, NodeId, NodeKind},
    };
    use teeny_core::{
        device::hardware::{HardwareProfile, MemoryLevel, MemoryLevelKind},
        graph::{DtypeRepr, Graph, Shape},
        model::{ExecutableOp, LoweringMode},
    };
    use teeny_test::graph::DecomposedSoftmaxBuilder;
    use teeny_test::hardware_profile::orin_nano;

    // A dummy Profiler that does nothing
    struct DummyProfiler;
    impl Profiler for DummyProfiler {
        fn profile(
            &self,
            tile_graph: &TileGraph,
            nodes: &[NodeId],
            hardware: &HardwareProfile,
        ) -> f64 {
            todo!()
        }
    }

    /// The connect level of the edge `src -> dst` on `dst`'s input `port`.
    /// The edge is stored twice (`src.out_edges` and `dst.in_edges`, see
    /// ANALYSIS.md §12.1 fix #14), so this also asserts that both copies agree.
    fn edge_level(
        tile_graph: &TileGraph,
        src: NodeId,
        dst: NodeId,
        port: usize,
    ) -> Option<MemoryLevelKind> {
        let in_edge = tile_graph
            .node(dst)
            .unwrap_or_else(|e| panic!("node {} should exist: {e}", dst.0))
            .in_edges()[port];
        assert_eq!(
            in_edge.src_node_id, src,
            "port {port} of node {} should be fed by node {}",
            dst.0, src.0
        );
        let out_edge = tile_graph
            .node(src)
            .unwrap_or_else(|e| panic!("node {} should exist: {e}", src.0))
            .out_edges()
            .iter()
            .find(|edge| edge.dst_node_id == dst && edge.dst_edge_id == EdgeId(port))
            .copied()
            .unwrap_or_else(|| panic!("node {} should have an out edge to node {}", src.0, dst.0));

        let level = in_edge.memory_level.map(|level| level.kind);
        assert_eq!(
            out_edge.memory_level.map(|level| level.kind),
            level,
            "edge {} -> {} (port {port}) has different levels in src.out_edges and dst.in_edges",
            src.0,
            dst.0
        );
        level
    }

    #[test]
    fn test_softmax_from_primitives_schedules_as_one_group() {
        // x -> max -> sub(x, .) -> exp -> sum -> div(exp, .)
        // |__________^              |___________^
        //
        // ANALYSIS.md §12.7 D1. `x` has two consumers, both reduced tensors
        // are broadcast back over the column axis, and `max` and `sum` each
        // need an rstep. Checks the parts of D1 visible in a schedule's
        // connect levels. Only DeviceMemory and SharedMemory are scheduled:
        // Triton, not the kernel writer, decides what lives in registers.
        //
        // - (b) a full-row tile is a legal tiling of the whole group, and a
        //   row is small enough to fit in shared memory, so the five compute
        //   nodes should fuse: every edge between them is at SharedMemory.
        // - `x` is a placeholder, produced outside the group, so its two
        //   out edges stay at DeviceMemory.
        const ROWS: usize = 64;
        const COLS: usize = 1024;

        let softmax = DecomposedSoftmaxBuilder::default()
            .rows(ROWS)
            .cols(COLS)
            .build()
            .expect("every required field is set")
            .graph();

        // A row is 4 KiB, so a full-row tile fits in Orin Nano's 96 KiB of
        // shared memory. The whole 256 KiB tensor doesn't, so the schedule
        // has to tile it.
        let hardware = orin_nano();

        let lowering = TritonLowering::default();
        let (dag, graph_to_dag, _) = lowering
            .lower_with_mapping(&softmax.graph, LoweringMode::Inference)
            .expect("lowering should not fail here");
        let mut tile_graph = TileGraph::from_dag(&dag);

        // `from_dag` numbers tile nodes in topological order, which need not
        // match graph order, so look each one up by its DAG index. The
        // trailing `Output` shares `div`'s DAG index, so skip it.
        let tile_node = |graph_idx: usize| {
            let dag_idx = graph_to_dag[graph_idx];
            tile_graph
                .nodes
                .iter()
                .find(|node| node.dag_node_idx == dag_idx && node.kind != NodeKind::Output)
                .map(|node| node.id)
                .unwrap_or_else(|| panic!("graph node {graph_idx} should have a tile node"))
        };
        let [x, max, sub, exp, sum, div] = [
            softmax.input,
            softmax.max,
            softmax.sub,
            softmax.exp,
            softmax.sum,
            softmax.div,
        ]
        .map(tile_node);

        schedule_graph(&mut tile_graph, &hardware, &SimpleProfiler, 5)
            .expect("scheduling softmax should not fail here");

        for (src, dst, port) in [(x, max, 0), (x, sub, 0)] {
            assert_eq!(
                edge_level(&tile_graph, src, dst, port),
                Some(MemoryLevelKind::DeviceMemory),
                "x is produced outside the group, so edge {} -> {} should stay at DeviceMemory",
                src.0,
                dst.0
            );
        }

        for (name, src, dst, port) in [
            ("max -> sub", max, sub, 1),
            ("sub -> exp", sub, exp, 0),
            ("exp -> sum", exp, sum, 0),
            ("exp -> div", exp, div, 0),
            ("sum -> div", sum, div, 1),
        ] {
            assert_eq!(
                edge_level(&tile_graph, src, dst, port),
                Some(MemoryLevelKind::SharedMemory),
                "{name} should be fused at SharedMemory"
            );
        }
    }
}
