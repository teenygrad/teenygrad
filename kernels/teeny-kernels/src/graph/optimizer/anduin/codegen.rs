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

//! The code generator built on Welder §3.3's `ExecuteGraph`/`Profile`
//! interfaces (teenygrad-1nr.6) — turning a scheduled tile-graph into a
//! `Dag` of (possibly fused) custom ops, the way the original,
//! hand-coded Anduin fusion strategies did before they were removed for
//! not being Welder (see this crate's `anduin` module doc comment).
//!
//! [`ExecuteDevice`] is Table 1's four abstracted device interfaces
//! (`Allocate`/`LoadTiles`/`ComputeTile`/`StoreTiles`), the same pluggable
//! pattern [`super::Profiler`] already uses, plus our own addition
//! [`ExecuteDevice::virtual_node`] (see its doc comment for why). Two
//! directions drive it:
//!
//! - [`super::trace::Trace::trace_graph`] (its own module) walks a
//!   [`SubGraphTilingResult`] (teenygrad-1nr.4) *live*, driving an
//!   `ExecuteDevice` as it recurses through the memory hierarchy.
//!   [`super::trace::Trace`] is the `ExecuteDevice` shipped for this
//!   direction: it just records a [`super::trace::TraceEvent`] trace
//!   rather than doing anything real.
//! - [`codegen`] runs the other way: given an *already-recorded* trace
//!   (typically `Trace::events`, from a completed `Trace::trace_graph` run),
//!   it replays each event through an `ExecuteDevice` again — the same
//!   interface, just driven from a static event list instead of a live
//!   recursive walk. [`DagCodegen`] is the intended `ExecuteDevice` for
//!   *this* direction: one that builds a real `Dag<Box<dyn ExecutableOp>>`
//!   of custom ops as it goes — each `virtual_node` call marking the
//!   start of one such op (matching `GraphOptimizer::optimize`'s own
//!   `(Dag, Vec<usize>)` contract, the way the original hand-coded Anduin
//!   fusion strategies did before removal). Its methods are dummy stubs
//!   (`todo!()`) for now — real (non-tracing) codegen is §4.2's scope
//!   (register-level `compute_inline`-style fusion, shared-memory
//!   load/store rewriting, block/thread index remapping, a best-fit
//!   shared-memory allocator) and overlaps heavily with teenygrad-1nr.1's
//!   still-open `Tile<D>` composition rework.

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

use super::trace::TraceEvent;

use super::tile_graph::NodeId;

/// Welder Table 1's four abstracted device interfaces, as a pluggable
/// trait — mirrors [`super::Profiler`]'s existing pattern — plus one
/// addition of our own, [`virtual_node`](Self::virtual_node), not in
/// Table 1. `level` on `virtual_node`/`allocate`/`load_tiles`/`store_tiles`
/// is the memory level [`Trace::trace_graph`](super::trace::Trace::trace_graph) is
/// currently executing at (Fig. 8's `mem`'s level); `nodes` on
/// `virtual_node`/`load_tiles`/`store_tiles` is the node set whose
/// boundary tiles are moving through that level in this call.
pub trait ExecuteDevice {
    /// Announces Welder §3.1/Fig. 5's *virtual node*: `nodes`, the set of
    /// original DAG nodes consolidated into one fused unit as viewed from
    /// `level` — e.g. Fig. 5's `Conv+ReLU` virtual node at L0. Always
    /// called first, before `allocate`, for every
    /// [`Trace::trace_graph`](super::trace::Trace::trace_graph) invocation (i.e. once
    /// per recursion frame) — including a singleton `nodes` (a node not
    /// actually fused with anything is still "viewed from `level`" as
    /// itself).
    ///
    /// Not part of Welder's own Table 1 (which has no notion of naming a
    /// virtual node explicitly — Fig. 8's `ExecuteGraph` just recurses).
    /// Added because this codebase's real `HardwareProfile`s only ever
    /// declare `SharedMemory`/`DeviceMemory` as levels a scheduling
    /// decision can target (Triton gives no explicit control over
    /// registers, L1, or L2 — those stay hardware-managed within a single
    /// kernel body). So unlike the paper, where a virtual node can exist
    /// at *any* level including deep register-level sub-fusion, in our
    /// system every virtual node this method reports is a genuine
    /// candidate kernel boundary — this is the exact grouping
    /// [`DagCodegen`] needs to decide "these nodes become one compiled
    /// kernel."
    fn virtual_node(&mut self, nodes: &[NodeId], level: MemoryLevelKind);
    /// Allocate a `footprint`-byte workspace in `level`
    /// (`TileGraph::mem_footprint_with_config`'s result for the current
    /// node set and config).
    fn allocate(&mut self, footprint: u64, level: MemoryLevelKind);
    /// Load `nodes`' input tiles into the workspace just allocated at
    /// `level`.
    fn load_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind);
    /// Compute `node`'s operator-tile directly — only called once
    /// `Trace::trace_graph` has recursed to the top of the memory hierarchy.
    fn compute_tile(&mut self, node: NodeId);
    /// Store `nodes`' result tiles from `level`'s workspace back down.
    fn store_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind);
}

/// Replays an already-recorded `trace` (typically
/// [`Trace::events`](super::trace::Trace::events), from a
/// completed [`Trace::trace_graph`](super::trace::Trace::trace_graph) run) through
/// `device` — the same [`ExecuteDevice`] interface `Trace::trace_graph` drives
/// live, just fed from a static event list instead of a recursive walk.
/// Dispatches each [`TraceEvent`] variant to `device`'s corresponding
/// method, in order.
pub fn codegen(trace: &[TraceEvent], device: &mut dyn ExecuteDevice) {
    for event in trace {
        match event {
            TraceEvent::VirtualNode { nodes, level } => device.virtual_node(nodes, *level),
            TraceEvent::Allocate { footprint, level } => device.allocate(*footprint, *level),
            TraceEvent::LoadTiles { nodes, level } => device.load_tiles(nodes, *level),
            TraceEvent::ComputeTile { node } => device.compute_tile(*node),
            TraceEvent::StoreTiles { nodes, level } => device.store_tiles(nodes, *level),
        }
    }
}

/// The genuine code generator [`codegen`] is meant to drive: an
/// [`ExecuteDevice`] that builds a real `Dag<Box<dyn ExecutableOp>>` of
/// custom (possibly fused) ops as it replays a trace, matching
/// `GraphOptimizer::optimize`'s own `(Dag, Vec<usize>)` contract — the way
/// the original, hand-coded Anduin fusion strategies did before removal.
///
/// Not yet implemented: every method is a dummy stub. Building this for
/// real needs the same composition machinery `#[tiled_kernel]`'s rework
/// (teenygrad-1nr.1) is blocked on — see this module's doc comment.
#[derive(Default)]
pub struct DagCodegen {
    // Not read yet -- every method below is a stub. Will back
    // `ExecuteDevice`'s methods and `into_dag` once implemented.
    #[allow(dead_code)]
    dag: Option<Dag<Box<dyn ExecutableOp>>>,
}

impl DagCodegen {
    /// A fresh codegen pass over `dag` (the original, un-fused `Dag` — the
    /// same one [`TraceEvent::ComputeTile`]'s `NodeId`s index into).
    pub fn new(dag: Dag<Box<dyn ExecutableOp>>) -> Self {
        Self { dag: Some(dag) }
    }

    /// Consumes this pass and returns the generated `Dag` of custom ops
    /// plus the original-node-index -> generated-node-index mapping,
    /// mirroring `GraphOptimizer::optimize`'s return shape.
    pub fn into_dag(self) -> (Dag<Box<dyn ExecutableOp>>, Vec<usize>) {
        todo!("teenygrad-1nr: finalize the generated Dag of custom ops from the replayed trace")
    }
}

impl ExecuteDevice for DagCodegen {
    fn virtual_node(&mut self, _nodes: &[NodeId], _level: MemoryLevelKind) {
        todo!("teenygrad-1nr: begin generating a custom op for this virtual node's group")
    }

    fn allocate(&mut self, _footprint: u64, _level: MemoryLevelKind) {
        todo!("teenygrad-1nr: allocate a workspace for the fused group being generated")
    }

    fn load_tiles(&mut self, _nodes: &[NodeId], _level: MemoryLevelKind) {
        todo!("teenygrad-1nr: wire up this fused group's input tiles")
    }

    fn compute_tile(&mut self, _node: NodeId) {
        todo!("teenygrad-1nr: fold this node's op into the fused group being generated")
    }

    fn store_tiles(&mut self, _nodes: &[NodeId], _level: MemoryLevelKind) {
        todo!("teenygrad-1nr: finalize this fused group into a custom-op Dag node")
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::device::hardware::{HardwareProfile, MemoryLevel};
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::*;
    use crate::graph::optimizer::anduin::trace::Trace;
    use crate::graph::optimizer::anduin::{SubGraphTilingResult, TileConfig, TileGraph};

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

    fn two_level_hardware() -> HardwareProfile {
        HardwareProfile {
            name: "two-level".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::Register,
                    capacity: u64::MAX,
                    bandwidth: None,
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: u64::MAX,
                    bandwidth: None,
                    latency: None,
                },
            ],
        }
    }

    #[test]
    fn codegen_replays_a_trace_through_a_device_in_order() {
        // Record a real trace via trace_graph, then replay it through a
        // fresh Trace via codegen -- the replayed trace must match
        // the original exactly.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware();
        let result = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: Vec::new(),
        };

        let recorder = Trace::trace_graph(
            &tile_graph,
            &result,
            MemoryLevelKind::DeviceMemory,
            &hardware,
        );

        let mut replayed = Trace::default();
        codegen(&recorder.events, &mut replayed);

        assert_eq!(replayed.events, recorder.events);
    }
}
