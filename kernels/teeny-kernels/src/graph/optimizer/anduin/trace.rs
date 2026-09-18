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

//! [`Trace::trace_graph`] — Welder §3.3's `ExecuteGraph` (Fig. 8), renamed
//! and made an associated function of [`Trace`]: what this walk actually
//! does, in this codebase, is build a *trace* — the one and only way a
//! [`Trace`] gets created — not execute anything for real, so a free
//! `execute_graph`/`trace_graph` taking a generic
//! [`ExecuteDevice`](super::codegen::ExecuteDevice) claimed more genericity
//! than the walk ever used: nothing has ever driven it with anything but a
//! [`Trace`].
//!
//! `Trace::trace_graph` recurses through the memory hierarchy: allocate a
//! workspace *at this level*, load input tiles into it, then for each node
//! either compute it directly (if this is the top memory level) or recurse
//! into that node's own subgraph one level up — finally store results back
//! down. [`SubGraphTilingResult`] (teenygrad-1nr.4) is already exactly this
//! recursive shape (a config at this level plus recursively-tiled
//! children), so `trace_graph` just walks it; there's no separate
//! "execution plan" type to build.
//!
//! The resulting trace is deliberately the intended output — not just a
//! test double. It's the input [`codegen`](super::codegen::codegen)
//! replays through a *different* [`ExecuteDevice`](super::codegen::ExecuteDevice)
//! — [`DagCodegen`](super::codegen::DagCodegen), still a stub — to
//! actually build a `Dag` of custom ops. `codegen`'s replay direction is
//! where genericity over `ExecuteDevice` actually matters (any device can
//! consume a trace); building one only ever produces a `Trace`. See
//! `codegen`'s module doc comment for the full picture.

use teeny_core::device::hardware::{HardwareProfile, MemoryLevelKind};

use super::codegen::ExecuteDevice;
use super::tile_graph::{NodeId, TileGraph};

/// One call [`Trace`] recorded.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEvent {
    /// Welder §3.1/Fig. 5's *virtual node*: the original `NodeId`s
    /// consolidated into one fused unit as viewed from `level` — see
    /// [`ExecuteDevice::virtual_node`](super::codegen::ExecuteDevice::virtual_node).
    VirtualNode {
        nodes: Vec<NodeId>,
        level: MemoryLevelKind,
    },
    Allocate {
        footprint: u64,
        level: MemoryLevelKind,
    },
    LoadTiles {
        nodes: Vec<NodeId>,
        level: MemoryLevelKind,
    },
    ComputeTile {
        node: NodeId,
    },
    StoreTiles {
        nodes: Vec<NodeId>,
        level: MemoryLevelKind,
    },
}

/// A recorded trace of [`ExecuteDevice`] calls — a structural stand-in for
/// actually executing a scheduled tile-graph. The only way to build one is
/// [`Trace::trace_graph`]; until a real device exists, a `Trace` is also
/// the handoff point to a future codegen pass (`codegen`'s module doc
/// comment).
#[derive(Debug, Default)]
pub struct Trace {
    pub events: Vec<TraceEvent>,
}

impl Trace {
    /// Welder §3.3's `ExecuteGraph` (Fig. 8), renamed and made an
    /// associated function — see the module doc comment for why. Builds a
    /// fresh `Trace` by recursively walking `result` (a
    /// [`TileGraph::sub_graph_tiling`]/[`TileGraph::resolved_tiling`]
    /// result) at `level`.
    ///
    /// Deviates from the paper's literal `for n : g.nodes()` loop in one
    /// respect, not specified precisely enough by the pseudocode to port
    /// verbatim: `SubGraphTilingResult::children` can (via
    /// `sub_graph_tiling`'s own deduplication) cover *several* of
    /// `result`'s nodes with a single child, when their subgraphs one
    /// level up turn out identical (a fused group becomes one virtual
    /// node at the next level — see §3.1/Fig. 5). So rather than
    /// dispatching once per node unconditionally, this dispatches once
    /// per *child* (covering every node that child's `nodes` contains)
    /// and only falls through to `compute_tile` directly for a node no
    /// child covers (e.g. one with no declared `tile_spec`, a hard
    /// boundary `propagate` never resolved past).
    pub fn trace_graph(
        tile_graph: &TileGraph,
        level: MemoryLevelKind,
        hardware: &HardwareProfile,
    ) -> Self {
        todo!("teenygrad-1nr: implement Trace::trace_graph")
    }
}

impl ExecuteDevice for Trace {
    fn virtual_node(&mut self, nodes: &[NodeId], level: MemoryLevelKind) {
        self.events.push(TraceEvent::VirtualNode {
            nodes: nodes.to_vec(),
            level,
        });
    }

    fn allocate(&mut self, footprint: u64, level: MemoryLevelKind) {
        self.events.push(TraceEvent::Allocate { footprint, level });
    }

    fn load_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind) {
        self.events.push(TraceEvent::LoadTiles {
            nodes: nodes.to_vec(),
            level,
        });
    }

    fn compute_tile(&mut self, node: NodeId) {
        self.events.push(TraceEvent::ComputeTile { node });
    }

    fn store_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind) {
        self.events.push(TraceEvent::StoreTiles {
            nodes: nodes.to_vec(),
            level,
        });
    }
}
