/*
 * Copyright (c) 2026 Teenygrad.
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
//! Named after rivers of Middle-earth (Anduin, …). Strategies rewrite a
//! [`teeny_core::graph::Graph`] into [`teeny_core::graph::Op::Custom`] nodes
//! (see [`ops`]) before Triton lowering.

mod anduin;
pub mod ops;

pub use anduin::Anduin;
pub use ops::PointwiseFuse;

use teeny_core::graph::Graph;

use crate::errors::Result;

/// Backend-specific graph rewrite applied before Triton lowering.
///
/// Attach with [`super::TritonLowering::with_optimizer`]. Multiple strategies
/// (Anduin, later peers) implement this trait; the lowering chooses which one runs.
pub trait GraphOptimizer: Send + Sync {
    /// Short stable name (e.g. `"anduin"`).
    fn name(&self) -> &str;

    /// Rewrite `graph` for this strategy. Must be pure w.r.t. the input graph.
    fn optimize(&self, graph: &Graph) -> Result<Graph>;
}
