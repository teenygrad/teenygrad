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
//! `optimize` now runs the full pipeline every piece Fig. 7/8 needs:
//! `SetConnect`/`ExtractSubgraph`/`MemFootprint`/`MemTraffic`/`Profiler`,
//! `Propagate` (teenygrad-1nr.2, via a revived, metadata-only
//! `KernelTileSpec` consumed by name-matching, not the removed
//! `#[tile(...)]` attribute's codegen), `EnumerateSubtiles` (teenygrad-1nr.3,
//! a Roller-style power-of-two tile search), `sub_graph_tiling`
//! (teenygrad-1nr.4, Welder's `SubGraphTiling`), [`schedule_graph`]
//! (teenygrad-1nr.5, Welder's `GraphConnecting` under a better name — see
//! `schedule`'s module doc comment), and §3.3's `Trace::trace_graph` +
//! [`codegen`] (teenygrad-1nr.6). What it can't do yet is materialize real
//! fused kernels: [`DagCodegen`], the `ExecuteDevice` `codegen` replays a
//! trace through to build the rewritten `Dag`, still has every method as a
//! `todo!()` stub — that needs reworking `#[tiled_kernel]` to compose
//! `Tile<D>` functions (teenygrad-1nr.1) first. So `optimize` reaches
//! `DagCodegen` and panics there instead of at a blanket `todo!()` up
//! front — the pipeline is wired end-to-end, but the last stage is a
//! deliberate stub. See `TILE_GRAPH_SCHEDULING_PLAN.md` and teenygrad-1nr.

mod codegen;
mod profile;
mod schedule;
mod tile_graph;
mod trace;

pub use codegen::{DagCodegen, ExecuteDevice, codegen};
pub use profile::{Profiler, SimpleProfiler};
pub use schedule::schedule_graph;
pub use tile_graph::{
    EdgeId, NodeId, SubGraphTilingResult, TileConfig, TileDim, TileEdge, TileEdgeShape, TileGraph,
    TileOp,
};
pub use trace::{Trace, TraceEvent};

use teeny_core::device::hardware::{HardwareProfile, MemoryLevelKind};
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
        // TODO(teenygrad-1nr.1): once `DagCodegen::into_dag` returns a real
        // node mapping, compose it with `_mapping` (graph-node-idx ->
        // input-dag-node-idx) to get the graph-node-idx -> output-dag-
        // node-idx mapping `GraphOptimizer::optimize` promises. Left
        // unused until that mapping exists to compose against.
        _mapping: Vec<usize>,
        hardware: &HardwareProfile,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)> {
        let (_tile_graph, traces) = Self::schedule(&dag, hardware)?;
        Ok(Self::codegen(dag, &traces))
    }
}

impl Anduin {
    /// §3.1-§3.3 up to (not including) codegen: builds a [`TileGraph`] from
    /// `dag`, schedules every edge's memory level and caches the winning
    /// tile shapes ([`schedule_graph`] — see `schedule`'s module doc
    /// comment), then replays each edge's resolved tiling into a [`Trace`]
    /// via [`Trace::trace_graph`]. Returns the scheduled `TileGraph` (for
    /// inspecting `connect_level`/`resolved_tiling`) alongside one `Trace`
    /// per root edge -- each trace's outermost `TraceEvent::VirtualNode`
    /// names the node set that edge's schedule would fuse into one kernel.
    ///
    /// Split out from `optimize` so scheduling/fusion decisions can be
    /// inspected and tested without going through [`Self::codegen`]'s
    /// still-`todo!()` [`DagCodegen`].
    ///
    /// `already_traced` skips an edge whose whole node set a
    /// previously-traced edge already covered -- mirrors
    /// `Trace::trace_graph`'s own `executed` bookkeeping, since a node can
    /// be the source of more than one outgoing edge whose resolved
    /// subgraphs overlap.
    fn schedule(
        dag: &Dag<Box<dyn ExecutableOp>>,
        hardware: &HardwareProfile,
    ) -> Result<(TileGraph, Vec<Trace>)> {
        let mut tile_graph = TileGraph::from_dag(dag);
        schedule_graph(&mut tile_graph, hardware, &SimpleProfiler)?;

        let mut already_traced = std::collections::HashSet::new();
        let mut traces = Vec::new();

        for node in tile_graph.topological_sort() {
            for (_, edge_id) in tile_graph.children(node) {
                let Some(result) = tile_graph.resolved_tiling(edge_id) else {
                    continue;
                };
                if result.nodes.iter().all(|n| already_traced.contains(n)) {
                    continue;
                }

                let trace =
                    Trace::trace_graph(&tile_graph, result, MemoryLevelKind::Register, hardware);
                already_traced.extend(result.nodes.iter().copied());
                traces.push(trace);
            }
        }

        Ok((tile_graph, traces))
    }

    /// §3.3's codegen finalization: replays every trace [`Self::schedule`]
    /// collected through a single [`DagCodegen`] pass so it accumulates
    /// every fused group into one output `Dag`, mirroring
    /// `GraphOptimizer::optimize`'s own `(Dag, Vec<usize>)` return shape.
    /// Hits `DagCodegen`'s `todo!()` stubs as soon as any `trace` has
    /// events -- see this module's doc comment.
    fn codegen(
        dag: Dag<Box<dyn ExecutableOp>>,
        traces: &[Trace],
    ) -> (Dag<Box<dyn ExecutableOp>>, Vec<usize>) {
        let mut codegen_device = DagCodegen::new(dag);
        for trace in traces {
            codegen::codegen(&trace.events, &mut codegen_device);
        }
        codegen_device.into_dag()
    }
}

#[cfg(test)]
mod tests {
    use super::{Anduin, NodeId, TraceEvent};
    use crate::graph::TritonLowering;
    use crate::graph::optimizer::GraphOptimizer;
    use crate::nn::fused::conv2d_bn_silu::Conv2dBnSiluForward;
    use teeny_core::device::hardware::MemoryLevelKind;
    use teeny_core::graph::{DtypeRepr, Graph, Op};
    use teeny_core::model::{LoweringMode, RuntimeOp};

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

    /// The [`TraceEvent::VirtualNode`] events in `events`, in order --
    /// pulled out for assertions instead of matching the whole trace, since
    /// `Allocate`/`LoadTiles`/`ComputeTile`/`StoreTiles` around them carry
    /// footprint numbers not relevant to "did this group of nodes fuse."
    fn virtual_nodes(events: &[TraceEvent]) -> Vec<(&[NodeId], MemoryLevelKind)> {
        events
            .iter()
            .filter_map(|event| match event {
                TraceEvent::VirtualNode { nodes, level } => Some((nodes.as_slice(), *level)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn conv2d_batchnorm_silu_schedule_fuses_the_three_compute_nodes_apart_from_input() {
        // input -> conv2d -> batchnorm2d -> silu
        //          \___________________________/
        //             Conv2d/Silu still have no declared KernelTileSpec
        //             (only Relu/MatMul/BatchNorm2d do -- see
        //             `graph::mod`'s "Tile-shape metadata" section);
        //             BatchNorm2d gained BATCHNORM2D_TILE_SPEC in
        //             teenygrad-1nr.8, after this test was first written.
        //             Asserts the group still fuses the same way with a
        //             mixed spec/no-spec node set -- `schedule_graph`/
        //             `sub_graph_tiling` don't actually require every
        //             node to have a tile_spec to make a real fusion
        //             decision, since mem_footprint/mem_traffic fall back
        //             to full-shape estimates for any node propagate
        //             can't refine.
        //
        //             B=2 (not 1): also asserts, on this same fused
        //             group, that the three real ops' own RuntimeOp::grid()
        //             implementations don't agree -- BatchNorm2d's grid is
        //             genuinely 2-D ([C, B]), Conv2d's/Silu's are flat 1-D
        //             -- teenygrad-1nr.17. B=2 makes this visible; at B=1
        //             BatchNorm2d's grid.y would trivially collapse to 1,
        //             indistinguishable from "no second dimension."
        let mut graph = Graph::new();
        let shape = vec![Some(2), Some(3), Some(32), Some(32)];
        let out_shape = vec![Some(2), Some(8), Some(32), Some(32)];

        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape);
        let conv = graph.add_node(
            Op::Conv2d {
                in_channels: 3,
                out_channels: 8,
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                padding_h: 1,
                padding_w: 1,
                groups: 1,
                has_bias: false,
            },
            vec![input],
            DtypeRepr::F32,
            out_shape.clone(),
        );
        let bn = graph.add_node(
            Op::BatchNorm2d {
                num_features: 8,
                eps: 1e-5,
                momentum: 0.1,
                affine: true,
                track_running_stats: true,
            },
            vec![conv],
            DtypeRepr::F32,
            out_shape.clone(),
        );
        graph.add_node(Op::Silu, vec![bn], DtypeRepr::F32, out_shape);

        let profile = orin_nano_hardware_profile();
        let lowering = TritonLowering::default();
        let (dag, _mapping, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .expect("lowering should not fail here");

        let (_tile_graph, traces) = Anduin::schedule(&dag, &profile).unwrap();

        // One root trace covers the whole 4-node graph: input(0)'s single
        // outgoing edge already resolves nodes [0,1,2,3], so the conv->bn
        // and bn->silu edges are skipped as already-traced.
        assert_eq!(traces.len(), 1);

        let virtual_nodes = virtual_nodes(&traces[0].events);

        // Real fusion decision made purely from mem_footprint/mem_traffic,
        // with no tile_spec on any of the three ops: at SharedMemory, conv
        // + batchnorm + silu land in one virtual node (nodes [1, 2, 3]),
        // separate from input (node [0]) -- i.e. this would become one
        // fused kernel plus a separate load of the input.
        assert!(
            virtual_nodes
                .iter()
                .any(|&(nodes, level)| nodes == [1, 2, 3] && level == MemoryLevelKind::SharedMemory),
            "expected conv+batchnorm+silu (nodes [1, 2, 3]) to be grouped into one \
             SharedMemory-level virtual node, got: {virtual_nodes:?}"
        );
        assert!(
            virtual_nodes
                .iter()
                .any(|&(nodes, level)| nodes == [0] && level == MemoryLevelKind::SharedMemory),
            "expected input (node [0]) to stay its own SharedMemory-level virtual node, \
             separate from the fused group, got: {virtual_nodes:?}"
        );

        // teenygrad-1nr.17: the fused group [1, 2, 3] above has no single
        // compatible grid. Pull each node's real RuntimeOp straight off
        // the lowered Dag (not a stub) and compare their actual
        // production grid() implementations: BatchNorm2dNchwInferenceRuntimeOp
        // (nn/norm/batchnorm.rs) is genuinely 2-D ([C, B, 1] -- channel on
        // Axis::X, batch on Axis::Y); Conv2dForward (nn/conv/conv2d.rs)
        // and SiluForward (nn/activation/sigmoid.rs) are both flat 1-D.
        // There is no (grid_x, grid_y, grid_z) a kernel covering all
        // three nodes could actually be launched with.
        let concrete_shape = [2usize, 8, 32, 32];
        let conv_grid = dag
            .node(1)
            .value
            .runtime_op()
            .expect("conv2d should have a real RuntimeOp")
            .grid(&concrete_shape);
        let bn_grid = dag
            .node(2)
            .value
            .runtime_op()
            .expect("batchnorm2d should have a real RuntimeOp")
            .grid(&concrete_shape);
        let silu_grid = dag
            .node(3)
            .value
            .runtime_op()
            .expect("silu should have a real RuntimeOp")
            .grid(&concrete_shape);

        assert_eq!(
            conv_grid[1], 1,
            "conv2d's own grid is flat 1-D: {conv_grid:?}"
        );
        assert_eq!(
            silu_grid[1], 1,
            "silu's own grid is flat 1-D: {silu_grid:?}"
        );
        assert!(
            bn_grid[1] > 1,
            "batchnorm2d's own grid is genuinely 2-D ([C, B, 1]): {bn_grid:?}"
        );

        // The correct grid for the fused group is knowable, even though
        // nothing in Anduin computes it yet: BatchNorm2d here runs in
        // inference mode (precomputed running mean/var, no data-dependent
        // reduction) and SiLU is pointwise, so both are pure per-element
        // functions of Conv2d's own output value at Conv2d's own thread
        // -- neither needs a grid dimension of its own. Conv2d is the
        // only op with a real compute-structural constraint (each thread
        // must reduce over its own KHxKW input window), so its own grid
        // is the correct one for the whole fused group; BatchNorm2d's
        // affine and SiLU's activation get folded into that same thread's
        // work with zero extra grid structure, exactly matching the
        // conv+BN+SiLU epilogue fusion already hand-written in this
        // codebase's own `Conv2dBnSiluForward` (nn/fused/conv2d_bn_silu.rs
        // -- "Grid: pid = ((b*C_OUT+c_out)*OH+oh)*num_ow_tiles+ow_tile").
        // block_ow=16 matches the same constant the real Conv2d node
        // above was constructed with (graph/mod.rs's `Op::Conv2d` arm).
        let correct_fused_grid =
            Conv2dBnSiluForward::new(3, 3, 1, 1, 1, 1, 1, 16).grid(&concrete_shape);
        assert_eq!(
            correct_fused_grid, conv_grid,
            "the correct fused grid should be conv2d's own native grid \
             (BatchNorm2d/SiLU are pure per-element epilogue work, matching \
             Conv2dBnSiluForward's own real grid), got: {correct_fused_grid:?}"
        );
        assert_ne!(
            correct_fused_grid, bn_grid,
            "the correct fused grid must NOT be batchnorm2d's own standalone \
             grid -- that 2-D [C, B] decomposition is just an artifact of how \
             BatchNorm2d happens to be scheduled when run standalone, not a \
             real requirement of the fused computation: {correct_fused_grid:?}"
        );
    }

    #[test]
    fn relu_reduce_sum_relu_schedule_fuses_relu_reduce_sum_relu_apart_from_input() {
        // input -> relu -> reduce_sum -> relu
        //
        // Both `Relu`s now carry a real, working `KernelTileSpec`
        // (`flat_elementwise_tile_spec`, fixed to match each node's real
        // rank -- previously `RELU_TILE_SPEC` only ever matched an
        // already-1-D node, never a realistic ND tensor like this
        // [2048, 4096] shape, so it was silently inert here). With a
        // real spec on both ends of the chain, `sub_graph_tiling` now
        // finds real SharedMemory-level structure instead of falling
        // straight through to one flat Register-level group the way it
        // used to: relu -> reduce_sum -> relu (nodes [1, 2, 3]) fuse
        // together, separate from input (node 0, which has no child
        // covering it and computes directly at the top Register frame).
        let mut graph = Graph::new();
        let shape = vec![Some(2048), Some(4096)];
        let reduced_shape = vec![Some(2048)];

        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape);
        let reduce_sum = graph.add_node(
            Op::ReduceSum {
                keepdims: false,
                noop_with_empty_axes: false,
            },
            vec![relu],
            DtypeRepr::F32,
            reduced_shape.clone(),
        );
        graph.add_node(Op::Relu, vec![reduce_sum], DtypeRepr::F32, reduced_shape);

        let profile = orin_nano_hardware_profile();
        let lowering = TritonLowering::default();
        let (dag, _mapping, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .expect("lowering should not fail here");

        let (_tile_graph, traces) = Anduin::schedule(&dag, &profile).unwrap();

        assert_eq!(traces.len(), 1);

        let virtual_nodes = virtual_nodes(&traces[0].events);
        assert!(
            virtual_nodes
                .iter()
                .any(|&(nodes, level)| nodes == [1, 2, 3] && level == MemoryLevelKind::SharedMemory),
            "expected relu+reduce_sum+relu (nodes [1, 2, 3]) to be grouped into one \
             SharedMemory-level virtual node, got: {virtual_nodes:?}"
        );
        assert!(
            virtual_nodes.iter().all(|&(nodes, _)| nodes != [0]),
            "expected input (node 0) to have no child covering it and compute \
             directly at the top frame, not get its own virtual node, got: \
             {virtual_nodes:?}"
        );

        let compute_tiles: Vec<NodeId> = traces[0]
            .events
            .iter()
            .filter_map(|event| match event {
                TraceEvent::ComputeTile { node } => Some(*node),
                _ => None,
            })
            .collect();
        assert!(
            compute_tiles.contains(&0),
            "expected input (node 0) to still be computed somewhere in the trace, \
             got: {compute_tiles:?}"
        );
    }
}
