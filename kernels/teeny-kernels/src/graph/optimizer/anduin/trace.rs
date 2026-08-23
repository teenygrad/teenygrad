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

//! [`TraceDevice`] — a structural [`ExecuteDevice`](super::codegen::ExecuteDevice)
//! for [`execute_graph`](super::codegen::execute_graph) that records a
//! [`TraceEvent`] trace instead of doing anything real.
//!
//! The trace is deliberately the intended output of running `execute_graph`
//! with this device — not just a test double. It's the input
//! [`codegen`](super::codegen::codegen) replays through a *second*
//! `ExecuteDevice` — [`DagCodegen`](super::codegen::DagCodegen), still a
//! stub — to actually build a `Dag` of custom ops. See `codegen`'s module
//! doc comment for the full picture.

use teeny_core::device::hardware::MemoryLevelKind;

use super::codegen::ExecuteDevice;
use super::tile_graph::NodeId;

/// One call [`TraceDevice`] recorded.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEvent {
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

/// A structural [`ExecuteDevice`]: records every call as a [`TraceEvent`]
/// instead of doing anything real. Lets tests assert
/// [`execute_graph`](super::execute::execute_graph)'s recursive walk
/// visits the right nodes, in the right order, with the right footprints
/// — and, until a real device exists, is the handoff point to a future
/// codegen pass. See the module doc comment.
#[derive(Debug, Default)]
pub struct TraceDevice {
    pub events: Vec<TraceEvent>,
}

impl ExecuteDevice for TraceDevice {
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
