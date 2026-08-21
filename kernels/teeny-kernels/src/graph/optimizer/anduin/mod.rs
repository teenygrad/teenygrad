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
//! detectors it used to run. The `#[tile(...)]`-declared tile-shape metadata
//! (`KernelTileSpec`, `propagate_within_kernel`/`propagate_graph`) that a
//! prior pass of this work built for a future scheduler was never consumed
//! by this optimizer or anything else and has been removed — see
//! teenygrad-1nr. Real tile shapes should come from the graph nodes
//! themselves once the TileGraph search exists, not from per-kernel
//! declarations.

mod tile_graph;

pub use tile_graph::{TileDim, TileEdge, TileEdgeShape, TileGraph, TileOp};

use teeny_core::device::hardware::HardwareProfile;
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

use crate::errors::Result;
use crate::graph::optimizer::GraphOptimizer;

/// Anduin: Triton-side rewrite of an already-lowered pipeline.
#[derive(Debug, Default, Clone, Copy)]
pub struct Anduin;

impl GraphOptimizer for Anduin {
    fn name(&self) -> &str {
        "anduin"
    }

    fn optimize(
        &self,
        dag: Dag<Box<dyn ExecutableOp>>,
        _mapping: Vec<usize>,
        _hardware: &HardwareProfile,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)> {
        // Step 1: structural Dag -> TileGraph conversion (see `TileGraph::from_dag`).
        // Steps 2-3 (backward tile-shape propagation, then the per-node
        // memory-level search) are the rest of `teenygrad-1nr` — see this
        // module's doc comment — and still need to run before this can
        // materialize a rewritten Dag to return.
        let _tile_graph = TileGraph::from_dag(&dag);

        todo!("teenygrad-1nr: Welder-style TileGraph scheduler — see this module's doc comment")
    }
}

#[cfg(test)]
mod tests {
    use super::Anduin;
    use crate::graph::TritonLowering;
    use crate::graph::optimizer::GraphOptimizer;
    use teeny_core::graph::{DtypeRepr, Graph, Op};
    use teeny_core::model::LoweringMode;

    use crate::testing::orin_nano_hardware_profile;

    #[test]
    fn two_pointwise_then_reduction_then_pointwise_preserves_chain() {
        // input -> relu -> sigmoid -> reduce_sum -> relu
        //          \_____________/    \________/    \__/
        //          2 pointwise ops     reduction   pointwise
        //
        // Output:
        // input -> (relu -> sigmoid -> reduce_sum) -> relu
        //          \____________________________/     \__/
        //                  fused node               pointwise
        //
        let mut graph = Graph::new();
        let shape = vec![Some(2048), Some(4096)];
        let reduced_shape = vec![Some(2048)];

        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
        let sigmoid = graph.add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape.clone());
        let reduce_sum = graph.add_node(
            Op::ReduceSum {
                keepdims: false,
                noop_with_empty_axes: false,
            },
            vec![sigmoid],
            DtypeRepr::F32,
            reduced_shape.clone(),
        );
        graph.add_node(Op::Relu, vec![reduce_sum], DtypeRepr::F32, reduced_shape);

        // Anchor the "won't fit on a single SM" claim above against a real
        // two-level hardware profile: the full [2048, 4096] F32 tile is
        // bigger than shared memory but comfortably smaller than device
        // memory.
        let profile = orin_nano_hardware_profile();

        let lowering = TritonLowering::default();
        let (dag, mapping, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .expect("lowering should not fail here");

        let anduin = Anduin;
        let result = anduin.optimize(dag, mapping, &profile);
        let (_, elements) = result.expect("Anduin optimizer should not fail here");

        // The pointwise and reduction nodes should be fused into a single node.
        // the second Relu should be a separate node, as it cannot be fused with the reduction.
        assert_eq!(
            elements.len(),
            2,
            "Expected 2 elements in the optimizer output, got {}",
            elements.len()
        );
    }
}
