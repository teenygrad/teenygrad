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

//! Pluggable graph optimizers selected on [`super::TritonLowering`].
//!
//! Named after rivers of Middle-earth (Anduin, …). Strategies lower a
//! [`teeny_core::graph::Graph`] straight to an executable pipeline, in place
//! of [`super::TritonLowering`]'s ordinary per-`Op` dispatch table.

mod anduin;

pub use anduin::{Anduin, TileDim, TileEdge, TileEdgeShape, TileGraph, TileOp};

use teeny_core::device::hardware::HardwareProfile;
use teeny_core::graph::Graph;
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

use crate::errors::Result;

/// Backend-specific strategy that lowers a graph straight to an executable
/// pipeline, bypassing [`super::TritonLowering`]'s ordinary per-`Op`
/// dispatch table.
///
/// Attach with [`super::TritonLowering::with_optimizer`]. Multiple strategies
/// (Anduin, later peers) implement this trait; the lowering chooses which one runs.
pub trait GraphOptimizer: Send + Sync {
    /// Short stable name (e.g. `"anduin"`).
    fn name(&self) -> &str;

    /// Lowers `graph` for this strategy, using `hardware` for any
    /// scheduling/cost-model decisions (e.g. Anduin's memory-level search).
    /// Must be pure w.r.t. the input graph.
    ///
    /// A strategy like Anduin can fuse several graph nodes into one DAG node
    /// (a single fused kernel), so the result isn't just a rewritten
    /// [`Graph`] fed back through the ordinary per-`Op` table — it's the
    /// final pipeline, exactly as [`teeny_core::model::Lowering::lower`]
    /// would produce it. The returned `Vec<usize>` is the graph-node-index →
    /// DAG-node-index mapping (indexed against the *input* `graph`, fused
    /// nodes included, many-to-one where fusion occurred) — callers need
    /// this to place pretrained weights for named graph nodes onto the
    /// right DAG node, the same way [`super::TritonLowering::lower_with_mapping`]'s
    /// mapping is used today.
    fn optimize(
        &self,
        graph: &Graph,
        hardware: &HardwareProfile,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)>;
}
