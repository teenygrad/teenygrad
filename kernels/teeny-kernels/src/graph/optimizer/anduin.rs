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

//! Anduin — Triton graph optimizer.
//!
//! The previous implementation was a set of hand-coded pattern detectors
//! (linear pointwise chains, fan-in binary tails, chain-into-reduction,
//! chain-into-transpose), each matching one `Op` shape and each lowering via
//! `FusionCore` source-text splicing. That is not Welder (OSDI'23): Welder
//! never special-cases op shapes. It converts the graph to a TileGraph,
//! propagates tile shapes backward from the output (as expressions in
//! shared free variables, not fixed numbers), then searches -- per node --
//! which memory-hierarchy level its output tile should live at, pruning
//! candidates with a cheap cost model down to a top-k before validating the
//! winner on real hardware. See `teenygrad-1nr`.
//!
//! `optimize` is a stub until that TileGraph search replaces the pattern
//! detectors it used to run. See [`super::propagate::propagate_graph`] for
//! the one piece of the real algorithm (backward shape propagation) already
//! built.

use teeny_core::graph::Graph;

use crate::errors::Result;
use crate::graph::optimizer::GraphOptimizer;

/// Anduin: Triton-side graph rewrite before lowering.
#[derive(Debug, Default, Clone, Copy)]
pub struct Anduin;

impl GraphOptimizer for Anduin {
    fn name(&self) -> &str {
        "anduin"
    }

    fn optimize(&self, _graph: &Graph) -> Result<Graph> {
        todo!("teenygrad-1nr: Welder-style TileGraph scheduler — see this module's doc comment")
    }
}
