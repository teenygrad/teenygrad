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

//! Anduin — first Triton graph optimizer (LotR river names).

use teeny_core::graph::Graph;

use crate::errors::Result;
use crate::graph::optimizer::GraphOptimizer;

/// Anduin: Triton-side graph rewrites before lowering.
///
/// Currently a no-op identity. Pattern-special kernels (e.g. the orphaned
/// `nn/fused/conv2d_bn_silu*`) are not emitted here — fusion should compose
/// member kernels natively instead of rewriting to a hand-written fused op.
#[derive(Debug, Default, Clone, Copy)]
pub struct Anduin;

impl GraphOptimizer for Anduin {
    fn name(&self) -> &str {
        "anduin"
    }

    fn optimize(&self, graph: &Graph) -> Result<Graph> {
        Ok(graph.clone())
    }
}
