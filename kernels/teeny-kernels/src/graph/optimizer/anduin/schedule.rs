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
    use teeny_core::device::hardware::{MemoryLevel, MemoryLevelKind};
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::*;
    use crate::graph::optimizer::anduin::profile::SimpleProfiler;

    struct TestOp {
        name: &'static str,
        dtype: DtypeRepr,
        shape: Shape,
        is_input: bool,
    }

    impl ExecutableOp for TestOp {
        fn name(&self) -> &str {
            self.name
        }

        fn is_input(&self) -> bool {
            self.is_input
        }

        fn forward_kernel_source(&self) -> &str {
            ""
        }

        fn forward_kernel_entry_point(&self) -> &str {
            ""
        }

        fn output_shape(&self) -> &Shape {
            &self.shape
        }

        fn output_dtype(&self) -> DtypeRepr {
            self.dtype
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn op(name: &'static str, shape: Shape, is_input: bool) -> Box<dyn ExecutableOp> {
        Box::new(TestOp {
            name,
            dtype: DtypeRepr::F32,
            shape,
            is_input,
        })
    }

    fn two_level_hardware(fast_bandwidth: f64, slow_bandwidth: f64) -> HardwareProfile {
        HardwareProfile {
            name: "test-device".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::Register,
                    capacity: u64::MAX,
                    bandwidth: Some(fast_bandwidth),
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: u64::MAX,
                    bandwidth: Some(slow_bandwidth),
                    latency: None,
                },
            ],
        }
    }

    #[test]
    fn schedule_graph_leaves_every_edge_at_a_declared_memory_level() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(1e12, 1e9);

        schedule_graph(&mut tile_graph, &hardware, &SimpleProfiler).unwrap();

        let ab_edge = tile_graph.children(a)[0].1;
        let level = tile_graph.connect_level(ab_edge);
        assert!(
            hardware.memory_levels.iter().any(|m| m.kind == level),
            "connect_level {level:?} was never one of the tried candidates"
        );
    }

    #[test]
    fn schedule_graph_prefers_the_faster_level_when_it_scores_better() {
        // Register is vastly higher-bandwidth than DeviceMemory here, so
        // SimpleProfiler's boundary-traffic-over-bandwidth estimate must
        // prefer Register for every edge.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(1e15, 1.0);

        schedule_graph(&mut tile_graph, &hardware, &SimpleProfiler).unwrap();

        let ab_edge = tile_graph.children(a)[0].1;
        assert_eq!(tile_graph.connect_level(ab_edge), MemoryLevelKind::Register);
    }

    #[test]
    fn schedule_graph_on_an_empty_hardware_profile_leaves_levels_unchanged() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let before = tile_graph.connect_level(ab_edge);

        let hardware = HardwareProfile {
            name: "empty".to_string(),
            compute_units: 1,
            memory_levels: vec![],
        };
        schedule_graph(&mut tile_graph, &hardware, &SimpleProfiler).unwrap();

        assert_eq!(tile_graph.connect_level(ab_edge), before);
    }

    #[test]
    fn schedule_graph_records_the_winning_tiling_result_per_edge() {
        // Before this fix, schedule_graph computed a SubGraphTilingResult
        // per candidate level and threw all of them away, keeping only
        // connect_level -- confirm it's now retained and covers the right
        // node set.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(1e12, 1e9);

        let ab_edge = tile_graph.children(a)[0].1;
        assert!(tile_graph.resolved_tiling(ab_edge).is_none());

        schedule_graph(&mut tile_graph, &hardware, &SimpleProfiler).unwrap();

        let resolved = tile_graph
            .resolved_tiling(ab_edge)
            .expect("schedule_graph should have recorded a winning result for this edge");
        assert!(resolved.nodes.contains(&a));
    }
}
