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

use super::{NodeId, TileGraph};

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
        todo!("teenygrad-1nr: implement SimpleProfiler::profile")
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
}
