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

//! Welder §3.3's `ExecuteGraph` (Fig. 8) and the code generator built on
//! top of it (teenygrad-1nr.6) — turning a scheduled tile-graph into a
//! `Dag` of (possibly fused) custom ops, the way the original,
//! hand-coded Anduin fusion strategies did before they were removed for
//! not being Welder (see this crate's `anduin` module doc comment).
//!
//! Two directions through the same [`ExecuteDevice`] interface
//! (Table 1's `Allocate`/`LoadTiles`/`ComputeTile`/`StoreTiles`, the same
//! pattern [`super::Profiler`] already uses):
//!
//! - [`execute_graph`] walks a [`SubGraphTilingResult`] (teenygrad-1nr.4)
//!   *live*, driving an `ExecuteDevice` as it recurses through the memory
//!   hierarchy. [`super::trace::TraceDevice`] is the `ExecuteDevice`
//!   shipped for this direction: it just records a
//!   [`super::trace::TraceEvent`] trace rather than doing anything real.
//! - [`codegen`] runs the other way: given an *already-recorded* trace
//!   (typically `TraceDevice::events`, from a completed `execute_graph`
//!   run), it replays each event through an `ExecuteDevice` again — the
//!   same interface, just driven from a static event list instead of a
//!   live recursive walk. [`DagCodegen`] is the intended `ExecuteDevice`
//!   for *this* direction: one that builds a real `Dag<Box<dyn ExecutableOp>>`
//!   of custom ops as it goes (each maximal fused group becoming one op),
//!   matching `GraphOptimizer::optimize`'s own `(Dag, Vec<usize>)`
//!   contract. Its four methods are dummy stubs (`todo!()`) for now —
//!   real (non-tracing) codegen is §4.2's scope (register-level
//!   `compute_inline`-style fusion, shared-memory load/store rewriting,
//!   block/thread index remapping, a best-fit shared-memory allocator) and
//!   overlaps heavily with teenygrad-1nr.1's still-open `Tile<D>`
//!   composition rework.

use teeny_core::device::hardware::{HardwareProfile, MemoryLevelKind};
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

use super::trace::TraceEvent;

use super::tile_graph::{NodeId, SubGraphTilingResult, TileGraph, next_memory_level};

/// Welder Table 1's four abstracted device interfaces, as a pluggable
/// trait — mirrors [`super::Profiler`]'s existing pattern. `level` on
/// `allocate`/`load_tiles`/`store_tiles` is the memory level
/// [`execute_graph`] is currently executing at (Fig. 8's `mem`'s level);
/// `nodes` on `load_tiles`/`store_tiles` is the node set whose boundary
/// tiles are moving through that level in this call.
pub trait ExecuteDevice {
    /// Allocate a `footprint`-byte workspace in `level`
    /// (`TileGraph::mem_footprint_with_config`'s result for the current
    /// node set and config).
    fn allocate(&mut self, footprint: u64, level: MemoryLevelKind);
    /// Load `nodes`' input tiles into the workspace just allocated at
    /// `level`.
    fn load_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind);
    /// Compute `node`'s operator-tile directly — only called once
    /// `execute_graph` has recursed to the top of the memory hierarchy.
    fn compute_tile(&mut self, node: NodeId);
    /// Store `nodes`' result tiles from `level`'s workspace back down.
    fn store_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind);
}

/// Welder §3.3's `ExecuteGraph` (Fig. 8): recursively executes `result`
/// (a [`TileGraph::sub_graph_tiling`]/[`TileGraph::resolved_tiling`]
/// result) at `level`, through `device`.
///
/// Deviates from the paper's literal `for n : g.nodes()` loop in one
/// respect, not specified precisely enough by the pseudocode to port
/// verbatim: `SubGraphTilingResult::children` can (via
/// `sub_graph_tiling`'s own deduplication) cover *several* of `result`'s
/// nodes with a single child, when their subgraphs one level up turn out
/// identical (a fused group becomes one virtual node at the next level —
/// see §3.1/Fig. 5). So rather than dispatching once per node
/// unconditionally, this dispatches once per *child* (covering every node
/// that child's `nodes` contains) and only falls through to
/// `compute_tile` directly for a node no child covers (e.g. one with no
/// declared `tile_spec`, a hard boundary `propagate` never resolved past).
pub fn execute_graph(
    tile_graph: &TileGraph,
    result: &SubGraphTilingResult,
    level: MemoryLevelKind,
    hardware: &HardwareProfile,
    device: &mut dyn ExecuteDevice,
) {
    let footprint = tile_graph.mem_footprint_with_config(&result.nodes, &result.config);
    device.allocate(footprint, level);
    device.load_tiles(&result.nodes, level);

    let next_level = next_memory_level(hardware, level);
    let mut executed = std::collections::HashSet::new();

    for &node in &result.nodes {
        if executed.contains(&node) {
            continue;
        }
        match next_level {
            None => {
                device.compute_tile(node);
                executed.insert(node);
            }
            Some(next_level) => {
                if let Some(child) = result.children.iter().find(|c| c.nodes.contains(&node)) {
                    execute_graph(tile_graph, child, next_level, hardware, device);
                    executed.extend(child.nodes.iter().copied());
                } else {
                    device.compute_tile(node);
                    executed.insert(node);
                }
            }
        }
    }

    device.store_tiles(&result.nodes, level);
}

/// Replays an already-recorded `trace` (typically
/// [`TraceDevice::events`](super::trace::TraceDevice::events), from a
/// completed [`execute_graph`] run) through `device` — the same
/// [`ExecuteDevice`] interface `execute_graph` drives live, just fed from
/// a static event list instead of a recursive walk. Dispatches each
/// [`TraceEvent`] variant to `device`'s corresponding method, in order.
pub fn codegen(trace: &[TraceEvent], device: &mut dyn ExecuteDevice) {
    for event in trace {
        match event {
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
    use teeny_core::device::hardware::MemoryLevel;
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::*;
    use crate::graph::optimizer::anduin::TileConfig;
    use crate::graph::optimizer::anduin::trace::{TraceDevice, TraceEvent};

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

    fn single_level_hardware(kind: MemoryLevelKind) -> HardwareProfile {
        HardwareProfile {
            name: "single-level".to_string(),
            compute_units: 1,
            memory_levels: vec![MemoryLevel {
                kind,
                capacity: u64::MAX,
                bandwidth: None,
                latency: None,
            }],
        }
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
    fn execute_graph_computes_directly_at_the_top_level() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(4)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = single_level_hardware(MemoryLevelKind::DeviceMemory);
        let footprint = tile_graph.mem_footprint(&[a]);

        let result = SubGraphTilingResult {
            nodes: vec![a],
            config: TileConfig::default(),
            children: Vec::new(),
        };

        let mut device = TraceDevice::default();
        execute_graph(
            &tile_graph,
            &result,
            MemoryLevelKind::DeviceMemory,
            &hardware,
            &mut device,
        );

        assert_eq!(
            device.events,
            vec![
                TraceEvent::Allocate {
                    footprint,
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::LoadTiles {
                    nodes: vec![a],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::ComputeTile { node: a },
                TraceEvent::StoreTiles {
                    nodes: vec![a],
                    level: MemoryLevelKind::DeviceMemory,
                },
            ]
        );
    }

    #[test]
    fn execute_graph_recurses_into_a_child_and_does_not_double_execute_its_nodes() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware();

        let child_footprint = tile_graph.mem_footprint(&[a, b]);
        let child = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: Vec::new(),
        };
        let top_footprint = tile_graph.mem_footprint(&[a, b]);
        let top = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: vec![child],
        };

        let mut device = TraceDevice::default();
        execute_graph(
            &tile_graph,
            &top,
            MemoryLevelKind::Register,
            &hardware,
            &mut device,
        );

        // b is covered by the same child as a (child.nodes contains both),
        // so the outer loop must dispatch the recursive call once (when it
        // reaches a) and skip b afterwards -- not compute or recurse twice.
        assert_eq!(
            device.events,
            vec![
                TraceEvent::Allocate {
                    footprint: top_footprint,
                    level: MemoryLevelKind::Register,
                },
                TraceEvent::LoadTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::Register,
                },
                TraceEvent::Allocate {
                    footprint: child_footprint,
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::LoadTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::ComputeTile { node: a },
                TraceEvent::ComputeTile { node: b },
                TraceEvent::StoreTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::DeviceMemory,
                },
                TraceEvent::StoreTiles {
                    nodes: vec![a, b],
                    level: MemoryLevelKind::Register,
                },
            ]
        );
    }

    #[test]
    fn execute_graph_falls_back_to_compute_tile_when_no_child_covers_a_node() {
        // Two independent nodes; only a is covered by a child (simulating
        // e.g. a hard boundary that kept b from being fused further) -- b
        // must still be computed, directly at the current level, not
        // silently dropped.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(4)], true));
        let b = dag.add_node(op("b", vec![Some(4)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware();

        let child_a = SubGraphTilingResult {
            nodes: vec![a],
            config: TileConfig::default(),
            children: Vec::new(),
        };
        let top = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: vec![child_a],
        };

        let mut device = TraceDevice::default();
        execute_graph(
            &tile_graph,
            &top,
            MemoryLevelKind::Register,
            &hardware,
            &mut device,
        );

        // b's ComputeTile must appear directly in the trace (not nested
        // inside a second Allocate/LoadTiles/StoreTiles block), and exactly
        // once.
        let compute_b_count = device
            .events
            .iter()
            .filter(|event| matches!(event, TraceEvent::ComputeTile { node } if *node == b))
            .count();
        assert_eq!(compute_b_count, 1);
        assert!(device.events.contains(&TraceEvent::ComputeTile { node: a }));
    }

    #[test]
    fn codegen_replays_a_trace_through_a_device_in_order() {
        // Record a real trace via execute_graph, then replay it through a
        // fresh TraceDevice via codegen -- the replayed trace must match
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

        let mut recorder = TraceDevice::default();
        execute_graph(
            &tile_graph,
            &result,
            MemoryLevelKind::DeviceMemory,
            &hardware,
            &mut recorder,
        );

        let mut replayed = TraceDevice::default();
        codegen(&recorder.events, &mut replayed);

        assert_eq!(replayed.events, recorder.events);
    }
}
