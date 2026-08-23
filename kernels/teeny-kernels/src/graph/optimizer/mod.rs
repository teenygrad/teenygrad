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

pub use anduin::{
    Anduin, DagCodegen, EdgeId, ExecuteDevice, NodeId, Profiler, SimpleProfiler,
    SubGraphTilingResult, TileConfig, TileDim, TileEdge, TileEdgeShape, TileGraph, TileOp,
    TraceDevice, TraceEvent, codegen, execute_graph, schedule_graph,
};

use teeny_core::device::hardware::HardwareProfile;
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

use crate::errors::Result;

/// Backend-specific strategy that rewrites an already-lowered pipeline,
/// e.g. by replacing a run of ops with a single fused [`ExecutableOp`].
///
/// Run after [`super::TritonLowering::lower_with_mapping`]: lowering has no
/// knowledge of optimization, so callers that want fusion take its
/// `(Dag, Vec<usize>)` output and feed it through a chosen strategy's
/// [`optimize`](GraphOptimizer::optimize) themselves. Multiple strategies
/// (Anduin, later peers) implement this trait.
pub trait GraphOptimizer: Send + Sync {
    /// Short stable name (e.g. `"anduin"`).
    fn name(&self) -> &str;

    /// Rewrites `dag` (already lowered, one node per graph op) for this
    /// strategy, using `hardware` for any scheduling/cost-model decisions
    /// (e.g. Anduin's memory-level search). Must be pure w.r.t. the input DAG.
    ///
    /// `mapping` is the graph-node-index → DAG-node-index mapping that
    /// produced `dag` (see [`super::TritonLowering::lower_with_mapping`]).
    /// A strategy like Anduin can fuse several DAG nodes into one (a single
    /// fused kernel) — any per-op information the strategy needs to decide
    /// that (shape, dtype, op identity, ...) must already be recoverable
    /// from the `ExecutableOp`s themselves, since the source [`teeny_core::graph::Graph`]
    /// is not available here. The returned `Vec<usize>` is `mapping`
    /// reindexed onto the *output* DAG's node indices (many-to-one where
    /// fusion occurred) — callers need this to place pretrained weights for
    /// named graph nodes onto the right DAG node.
    fn optimize(
        &self,
        dag: Dag<Box<dyn ExecutableOp>>,
        mapping: Vec<usize>,
        hardware: &HardwareProfile,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)>;
}
