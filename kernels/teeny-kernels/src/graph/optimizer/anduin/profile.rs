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

//! [`Profiler`] — Welder §3.2's `Profile` device interface.
//!
//! `GraphConnecting` (Fig. 7) calls `Min(d.Profile(configs))` to score each
//! candidate memory-level connection and keep the cheapest. Table 1's
//! abstracted hardware interfaces (`Allocate`, `LoadTiles`, `ComputeTile`,
//! `StoreTiles`, `MemLevels`) are what a *real* `Profile` would be built
//! from — actually allocating, loading, computing, and storing tiles on (or
//! in simulation of) a device and timing the result. That real profiler
//! doesn't exist yet: it needs `TileConfig`/`Propagate`, which are still
//! blocked on the open `ExecutableOp` fidelity decision (see
//! `TILE_GRAPH_SCHEDULING_PLAN.md`).
//!
//! [`SimpleProfiler`] is a stand-in that only needs what [`TileGraph`]
//! already has today: it estimates cost from [`TileGraph::mem_traffic`]'s
//! boundary bytes, divided per-edge by that edge's own connect-level
//! bandwidth. It is good enough to rank candidate connection levels
//! relative to each other structurally; it is not a substitute for
//! validating the winner on real hardware.

use teeny_core::device::hardware::HardwareProfile;
use teeny_core::graph::DtypeRepr;

use super::{NodeId, TileEdge, TileGraph};

/// Welder §3.2's `Profile` device interface: an estimated cost (lower is
/// better) of executing `nodes` — typically a
/// [`TileGraph::extract_subgraph`] result — as one fused unit on `hardware`.
/// `GraphConnecting` minimizes this over candidate connection levels to
/// decide `SetConnect`'s target.
pub trait Profiler {
    /// Estimated cost of executing `nodes` as a fused unit on `hardware`.
    fn profile(&self, tile_graph: &TileGraph, nodes: &[NodeId], hardware: &HardwareProfile) -> f64;
}

/// A structural [`Profiler`]: estimated latency in seconds, computed
/// per boundary edge (see [`TileGraph::boundary_edges`], the same set
/// [`TileGraph::mem_traffic`] sums) as that edge's byte size divided by its
/// own connect level's bandwidth in `hardware`. A boundary edge whose level
/// has no known bandwidth (`MemoryLevel::bandwidth` is `None`) contributes
/// zero rather than an estimate — optimistic, never pessimistic, about a
/// candidate's cost, same as [`TileEdge::byte_size`]'s dynamic-axis
/// handling.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleProfiler;

impl Profiler for SimpleProfiler {
    fn profile(&self, tile_graph: &TileGraph, nodes: &[NodeId], hardware: &HardwareProfile) -> f64 {
        tile_graph
            .boundary_edges(nodes)
            .into_iter()
            .map(|(edge_id, dtype)| edge_latency(tile_graph.edge(edge_id), dtype, hardware))
            .sum()
    }
}

/// Estimated seconds to move `edge`'s data through its own connect level,
/// using `hardware`'s bandwidth for that level. Zero if that level's
/// bandwidth isn't known.
fn edge_latency(edge: &TileEdge, dtype: DtypeRepr, hardware: &HardwareProfile) -> f64 {
    let bytes = edge.byte_size(dtype);
    let bandwidth = hardware
        .level(edge.memory_level)
        .and_then(|level| level.bandwidth);

    match bandwidth {
        Some(bandwidth) if bandwidth > 0.0 => bytes as f64 / bandwidth,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::device::hardware::{MemoryLevel, MemoryLevelKind};
    use teeny_core::graph::Shape;
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::*;

    /// Minimal [`ExecutableOp`] test double: just enough surface
    /// (name/shape/dtype) for [`TileGraph::from_dag`] to convert on.
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

    fn hardware_with_bandwidth(kind: MemoryLevelKind, bandwidth: f64) -> HardwareProfile {
        HardwareProfile {
            name: "test-device".to_string(),
            compute_units: 1,
            memory_levels: vec![MemoryLevel {
                kind,
                capacity: u64::MAX,
                bandwidth: Some(bandwidth),
                latency: None,
            }],
            execution: None,
        }
    }

    #[test]
    fn profile_of_a_single_boundary_node_sums_its_input_and_output_edges() {
        // A lone F32 [4] node: an input boundary edge in, an output
        // boundary edge out, both at DeviceMemory. 4 elements * 4 bytes =
        // 16 bytes per edge, twice, over a 16 bytes/sec device -> 2.0s.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape, true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = hardware_with_bandwidth(MemoryLevelKind::DeviceMemory, 16.0);

        let latency = SimpleProfiler.profile(&tile_graph, &[a], &hardware);

        assert_eq!(latency, 2.0);
    }

    #[test]
    fn profile_ignores_edges_internal_to_the_node_set() {
        // a -> b, both included: the internal edge contributes nothing:
        // only a's input boundary and b's output boundary count.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = hardware_with_bandwidth(MemoryLevelKind::DeviceMemory, 16.0);

        let latency = SimpleProfiler.profile(&tile_graph, &[a, b], &hardware);

        // a's input edge (16 bytes) + b's output edge (16 bytes), not the
        // a -> b edge in between.
        assert_eq!(latency, 2.0);
    }

    #[test]
    fn profile_counts_an_excluded_neighbors_edge_as_boundary_traffic() {
        // a -> b -> c, but only {a, b} are in the extracted set: b's edge
        // to the excluded c must still count, from b's outgoing side.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = hardware_with_bandwidth(MemoryLevelKind::DeviceMemory, 16.0);

        let latency = SimpleProfiler.profile(&tile_graph, &[a, b], &hardware);

        // a's input edge (16B) + b -> c edge (16B, excluded consumer) = 2.0s.
        // b -> c is internal-looking but c isn't in the set, so it counts;
        // a -> b is fully internal and does not.
        assert_eq!(latency, 2.0);
    }

    #[test]
    fn profile_counts_an_excluded_producers_edge_from_the_consumer_side() {
        // a -> b -> c, extracting {b, c}: b's edge from the excluded a must
        // still count, read from b's parent_edges.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = hardware_with_bandwidth(MemoryLevelKind::DeviceMemory, 16.0);

        let latency = SimpleProfiler.profile(&tile_graph, &[b, c], &hardware);

        // a -> b edge (16B, excluded producer) + c's output edge (16B) = 2.0s.
        // b -> c is fully internal and does not count.
        assert_eq!(latency, 2.0);
    }

    #[test]
    fn profile_treats_a_dynamic_axis_as_a_single_element() {
        let shape = vec![None, Some(4)]; // dynamic batch axis
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape, true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = hardware_with_bandwidth(MemoryLevelKind::DeviceMemory, 16.0);

        let latency = SimpleProfiler.profile(&tile_graph, &[a], &hardware);

        // a is isolated (no dag edges), so from_dag gives it both an input
        // and an output boundary edge: (1 * 4) elements * 4 bytes = 16
        // bytes / 16 = 1.0s, twice.
        assert_eq!(latency, 2.0);
    }

    #[test]
    fn profile_treats_unknown_bandwidth_as_zero_cost() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape, true));

        let tile_graph = TileGraph::from_dag(&dag);
        // No DeviceMemory level declared at all.
        let hardware = HardwareProfile {
            name: "test-device".to_string(),
            compute_units: 1,
            memory_levels: vec![],
            execution: None,
        };

        let latency = SimpleProfiler.profile(&tile_graph, &[a], &hardware);

        assert_eq!(latency, 0.0);
    }

    #[test]
    fn profile_of_an_empty_node_set_is_zero() {
        let tile_graph = TileGraph::default();
        let hardware = hardware_with_bandwidth(MemoryLevelKind::DeviceMemory, 16.0);

        assert_eq!(SimpleProfiler.profile(&tile_graph, &[], &hardware), 0.0);
    }
}
