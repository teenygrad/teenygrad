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

//! The code generator built on Welder §3.3's `ExecuteGraph`/`Profile`
//! interfaces (teenygrad-1nr.6) — turning a scheduled tile-graph into a
//! `Dag` of (possibly fused) custom ops.
//!
//! [`ExecuteDevice`] is Table 1's four abstracted device interfaces
//! (`Allocate`/`LoadTiles`/`ComputeTile`/`StoreTiles`), the same pluggable
//! pattern [`super::Profiler`] already uses, plus our own addition
//! [`ExecuteDevice::virtual_node`] (see its doc comment for why). Two
//! directions drive it:
//!
//! - [`super::trace::Trace::trace_graph`] (its own module) walks a
//!   [`SubGraphTilingResult`] (teenygrad-1nr.4) *live*, driving an
//!   `ExecuteDevice` as it recurses through the memory hierarchy.
//!   [`super::trace::Trace`] is the `ExecuteDevice` shipped for this
//!   direction: it just records a [`super::trace::TraceEvent`] trace
//!   rather than doing anything real.
//! - [`codegen`] runs the other way: given an *already-recorded* trace
//!   (typically `Trace::events`, from a completed `Trace::trace_graph` run),
//!   it replays each event through an `ExecuteDevice` again — the same
//!   interface, just driven from a static event list instead of a live
//!   recursive walk. [`DagCodegen`] is the intended `ExecuteDevice` for
//!   *this* direction: one that builds a real `Dag<Box<dyn ExecutableOp>>`
//!   of custom ops as it goes — each `virtual_node` call marking the
//!   start of one such op (matching `GraphOptimizer::optimize`'s own
//!   `(Dag, Vec<usize>)` contract, the way the original hand-coded Anduin
//!   fusion strategies did before removal). Its methods are dummy stubs
//!   (`todo!()`) for now — real (non-tracing) codegen is §4.2's scope
//!   (register-level `compute_inline`-style fusion, shared-memory
//!   load/store rewriting, block/thread index remapping, a best-fit
//!   shared-memory allocator) and overlaps heavily with teenygrad-1nr.1's
//!   still-open `Tile<D>` composition rework.
//!
//! ## Flat elementwise virtual nodes (the simple case)
//!
//! For a fused group whose every member is a unary `In<Tile<..>>` /
//! `Out<Tile<..>>` kernel with the usual `#[tile(block = BLOCK_SIZE,
//! extent = n_elements)]` + `n_elements: i32` shape (relu, silu, …),
//! [`DagCodegen`] drives codegen from the members' [`KernelTileSpec`] /
//! [`GridSpec`] (teenygrad-1nr.18/19) and the schedule's resolved
//! [`TileConfig`], not from each kernel's hand-picked constructor default:
//!
//! - **`BLOCK_SIZE`** — the concrete tile extent the scheduler chose
//!   (`TileDim::Fixed` on the virtual node's output edge in
//!   `resolved_tiling`), emitted as the fused kernel's `const BLOCK_SIZE`
//!   generic and as `cdiv(n_elements, BLOCK_SIZE)` CTAs on the grid axis
//!   named by `grid_spec()` (today always `GridDim::X` for this shape).
//! - **`n_elements`** — the runtime `i32` extent param named by
//!   `tile_spec()` (`extent_param`, usually `"n_elements"`): product of
//!   the graph node's output shape, passed through unchanged from the
//!   unfused `RuntimeOp::pack_args` ABI.
//!
//! The **wrapper** (owned by `virtual_node` + `load_tiles` +
//! `store_tiles`, emitted once per fused kernel) performs the pid decode
//! that standalone `#[tiled_kernel]` preludes inject today; composed
//! tile-op bodies must not repeat it. [`TraceEvent::Allocate`] is still
//! recorded by [`super::trace::Trace::trace_graph`] for scheduling cost
//! (`mem_footprint_with_config`), but [`DagCodegen`] does **not** emit a
//! separate workspace allocation for this shape: `load_tiles` materialises
//! boundary tiles directly via `T::load`, and intermediate `Tile` values
//! between chained `compute_tile` calls live in SSA registers, not an
//! explicitly allocated buffer.
//!
//! ```text
//! pid         = program_id(X)
//! block_start = pid * BLOCK_SIZE
//! offsets     = arange(0, BLOCK_SIZE) + block_start
//! in_bounds   = offsets.lt(n_elements)
//!
//! x  = Tile { load(x_ptr.add_offsets(offsets), mask=in_bounds), mask }
//! y  = Tile { y_ptr.add_offsets(offsets), mask }   // addressed, not loaded
//! ```
//!
//! `load_tiles` materialises boundary `In<Tile<..>>` params (`x` above).
//! Each `compute_tile` splices one member's tile-op *body* only (e.g.
//! `maximum(x.tensor, …)` for relu), threading the resulting `Tile`
//! through the chain as SSA values — no explicit staging buffer between
//! ops. `store_tiles` finishes with `store(y.tensor, result, y.mask, …)`.
//!
//! **Acceptance test:** `tests/test_fused_pointwise.rs` — `input -> relu ->
//! silu`, lowered and scheduled on real CUDA hardware.

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

use super::tile_graph::NodeId;
use super::trace::TraceEvent;

use crate::errors::Result;

/// Welder Table 1's four abstracted device interfaces, as a pluggable
/// trait — mirrors [`super::Profiler`]'s existing pattern — plus one
/// addition of our own, [`virtual_node`](Self::virtual_node), not in
/// Table 1. `level` on `virtual_node`/`allocate`/`load_tiles`/`store_tiles`
/// is the memory level [`Trace::trace_graph`](super::trace::Trace::trace_graph) is
/// currently executing at (Fig. 8's `mem`'s level); `nodes` on
/// `virtual_node`/`load_tiles`/`store_tiles` is the node set whose
/// boundary tiles are moving through that level in this call.
pub trait ExecuteDevice {
    /// Announces Welder §3.1/Fig. 5's *virtual node*: `nodes`, the set of
    /// original DAG nodes consolidated into one fused unit as viewed from
    /// `level` — e.g. Fig. 5's `Conv+ReLU` virtual node at L0. Always
    /// called first, before `allocate`, for every
    /// [`Trace::trace_graph`](super::trace::Trace::trace_graph) invocation (i.e. once
    /// per recursion frame) — including a singleton `nodes` (a node not
    /// actually fused with anything is still "viewed from `level`" as
    /// itself).
    ///
    /// Not part of Welder's own Table 1 (which has no notion of naming a
    /// virtual node explicitly — Fig. 8's `ExecuteGraph` just recurses).
    /// Added because this codebase's real `HardwareProfile`s only ever
    /// declare `SharedMemory`/`DeviceMemory` as levels a scheduling
    /// decision can target (Triton gives no explicit control over
    /// registers, L1, or L2 — those stay hardware-managed within a single
    /// kernel body). So unlike the paper, where a virtual node can exist
    /// at *any* level including deep register-level sub-fusion, in our
    /// system every virtual node this method reports is a genuine
    /// candidate kernel boundary — this is the exact grouping
    /// [`DagCodegen`] needs to decide "these nodes become one compiled
    /// kernel."
    fn virtual_node(&mut self, nodes: &[NodeId], level: MemoryLevelKind);

    /// Record a `footprint`-byte workspace at `level`
    /// (`TileGraph::mem_footprint_with_config`'s result for the current
    /// node set and config). [`Trace`] records this for scheduling; for
    /// flat elementwise fusion [`DagCodegen`] is a no-op here — boundary
    /// tiles are materialised by `T::load` in [`load_tiles`](Self::load_tiles),
    /// and intermediate tiles between chained [`compute_tile`](Self::compute_tile)
    /// calls are SSA values, not an explicitly allocated buffer.
    fn allocate(&mut self, footprint: u64, level: MemoryLevelKind);

    /// Load `nodes`' boundary input tiles at `level` — for flat elementwise
    /// fusion, emits `T::load` into a `Tile` (see the module doc's wrapper
    /// prelude).
    fn load_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind);

    /// Compute `node`'s operator-tile directly — only called once
    /// `Trace::trace_graph` has recursed to the top of the memory hierarchy.
    fn compute_tile(&mut self, node: NodeId);

    /// Store `nodes`' result tiles from `level`'s workspace back down.
    fn store_tiles(&mut self, nodes: &[NodeId], level: MemoryLevelKind);
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
pub struct AnduinCodegen {
    dag: Option<Dag<Box<dyn ExecutableOp>>>,

    /// The source code of the generated kernel.
    source: String,
}

impl AnduinCodegen {
    pub fn new() -> Self {
        Self {
            dag: None,
            source: String::new(),
        }
    }

    pub fn into_dag(self) -> Dag<Box<dyn ExecutableOp>> {
        self.dag.unwrap()
    }

    /// Replays an already-recorded `trace` (typically
    /// [`Trace::events`](super::trace::Trace::events), from a
    /// completed [`Trace::trace_graph`](super::trace::Trace::trace_graph) run) through
    /// `device` — the same [`ExecuteDevice`] interface `Trace::trace_graph` drives
    /// live, just fed from a static event list instead of a recursive walk.
    /// Dispatches each [`TraceEvent`] variant to `device`'s corresponding
    /// method, in order.
    pub fn codegen(&mut self, trace: &[TraceEvent]) -> Result<&str> {
        for event in trace {
            match event {
                TraceEvent::VirtualNode { nodes, level } => self.virtual_node(nodes, *level),
                TraceEvent::Allocate { footprint, level } => self.allocate(*footprint, *level),
                TraceEvent::LoadTiles { nodes, level } => self.load_tiles(nodes, *level),
                TraceEvent::ComputeTile { node } => self.compute_tile(*node),
                TraceEvent::StoreTiles { nodes, level } => self.store_tiles(nodes, *level),
            }
        }

        Ok(&self.source)
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}

impl ExecuteDevice for AnduinCodegen {
    fn virtual_node(&mut self, _nodes: &[NodeId], _level: MemoryLevelKind) {
        todo!("teenygrad-1nr: begin generating a custom op for this virtual node's group")
    }

    fn allocate(&mut self, _footprint: u64, _level: MemoryLevelKind) {
        // Scheduling-only for flat elementwise fusion: `load_tiles` emits
        // `T::load` directly; intermediate tiles are SSA between
        // `compute_tile` calls. Explicit workspace allocation is for
        // future multi-level, not this path.
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
    use super::*;

    use teeny_core::graph::{DtypeRepr, Graph, Op};
    use teeny_core::model::LoweringMode;

    use crate::graph::TritonLowering;
    use crate::graph::optimizer::anduin::Anduin;
    use crate::testing::hardware_profile::orin_nano;

    #[test]
    fn test_codegen_pointwise_virtual_node() {
        let mut graph = Graph::new();
        let shape = vec![Some(2048), Some(4096)];

        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
        let _silu = graph.add_node(Op::Silu, vec![relu], DtypeRepr::F32, shape.clone());

        // Anchor the "won't fit on a single SM" claim above against a real
        // two-level hardware profile: the full [2048, 4096] F32 tile is
        // bigger than shared memory but comfortably smaller than device
        // memory.
        let profile = orin_nano();

        let lowering = TritonLowering::default();
        let (dag, _, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .unwrap();

        let (_, traces) = Anduin::schedule(&dag, &profile).unwrap();
        eprintln!("traces: {:?}", traces);

        let mut codegen = AnduinCodegen::default();
        codegen
            .codegen(&traces[0].events)
            .expect("Codegen should not fail here");

        assert_eq!(codegen.source(), "TO DO: implement codegen");
    }
}
