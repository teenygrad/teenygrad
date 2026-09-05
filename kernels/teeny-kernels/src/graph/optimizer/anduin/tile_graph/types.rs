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

//! The value types a [`super::TileGraph`] is built from: shapes, edges,
//! nodes, and the arena record that ties an edge to its endpoints. Kept
//! separate from `TileGraph` itself (in the parent `tile_graph` module) so
//! every other submodule here can depend on these types without depending
//! on `TileGraph`'s own methods.

use std::collections::HashMap;

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::KernelTileSpec;

use super::NodeId;

/// One axis of a [`TileEdgeShape`]: either a concrete, known extent, or a
/// named symbolic axis (a free variable shared across nodes once the
/// propagation pass unifies matching symbols — see the module doc comment).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TileDim {
    /// A concrete, known extent.
    Fixed(usize),
    /// A named symbolic axis.
    Sym(String),
}

/// A tile-edge shape: one [`TileDim`] per axis.
pub type TileEdgeShape = Vec<TileDim>;

/// Converts a source [`Shape`] into a [`TileEdgeShape`]: known extents
/// become [`TileDim::Fixed`], and each dynamic (`None`) axis becomes a fresh
/// [`TileDim::Sym`] named after the `(node_index, axis)` it came from. Two
/// dynamic axes always get distinct symbols here, even if they will turn out
/// to be the same free variable — that unification is the propagation
/// pass's job, not this structural conversion's.
pub(super) fn to_tile_shape(node_index: NodeId, shape: &Shape) -> TileEdgeShape {
    shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| match dim {
            Some(extent) => TileDim::Fixed(*extent),
            None => TileDim::Sym(format!("n{node_index}d{axis}")),
        })
        .collect()
}

/// One edge in a [`super::TileGraph`]: the shape and memory level of the
/// value it carries. Used both for internal producer→consumer edges and for
/// graph-boundary edges — see the module doc comment.
///
/// `memory_level` is deliberately per-edge, not per-node: Welder §3.1
/// connects two adjacent operator-tiles through a *reuse-tile* "along each
/// adjacent edge," so a producer with several consumers can be connected to
/// each at a different level (e.g. fused into one consumer's kernel at
/// `Register` while a different consumer reads a separately materialized
/// copy from `DeviceMemory`). That's not an inconsistency to prevent — it's
/// exactly the per-edge decision `GraphConnecting` (§3.2) searches over to
/// pick fusion boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct TileEdge {
    /// Shape of the value this edge carries.
    pub shape: TileEdgeShape,
    /// Memory level this value is materialized at on this edge.
    pub memory_level: MemoryLevelKind,
}

impl TileEdge {
    /// This edge's data size in bytes, given the `dtype` of whichever side
    /// produced it (an edge doesn't carry its own dtype — see
    /// [`TileOp::dtype`]), using this edge's own full, untiled shape. A
    /// dynamic ([`TileDim::Sym`]) axis counts as extent 1, since no
    /// concrete size is known for it — this makes the result optimistic,
    /// never an overestimate, of the true size. Use
    /// [`super::TileGraph::mem_footprint_with_config`]/
    /// [`super::TileGraph::mem_traffic_with_config`] instead when a
    /// [`TileConfig`] has a tiled (smaller) shape for this edge.
    pub fn byte_size(&self, dtype: DtypeRepr) -> u64 {
        shape_byte_size(&self.shape, dtype)
    }
}

/// Byte size of `shape` (element count × dtype size), the shared core of
/// [`TileEdge::byte_size`] and every config-aware footprint/traffic
/// computation. A dynamic ([`TileDim::Sym`]) axis counts as extent 1 — see
/// [`TileEdge::byte_size`]'s doc comment.
pub(super) fn shape_byte_size(shape: &TileEdgeShape, dtype: DtypeRepr) -> u64 {
    let elements: u64 = shape
        .iter()
        .map(|dim| match dim {
            TileDim::Fixed(extent) => *extent as u64,
            TileDim::Sym(_) => 1,
        })
        .product();
    elements * dtype_bytes(dtype)
}

fn dtype_bytes(dtype: DtypeRepr) -> u64 {
    match dtype {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

/// Opaque handle to one edge in a [`super::TileGraph`]'s arena. Addresses an
/// internal or a graph-boundary edge uniformly — see the module doc
/// comment. Obtained from [`super::TileGraph::children`],
/// [`super::TileGraph::input_edge_id`], or [`super::TileGraph::output_edge_id`];
/// dereferenced via [`super::TileGraph::edge`]/[`super::TileGraph::connect_level`]
/// and mutated via [`super::TileGraph::set_connect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub(crate) usize);

/// One arena slot: a [`TileEdge`] plus the node indices of its endpoints.
/// `producer`/`consumer` are `None` exactly when this edge is a
/// graph-boundary edge on that side — see the module doc comment.
#[derive(Debug, Clone)]
pub(super) struct TileEdgeRecord {
    pub(super) producer: Option<NodeId>,
    pub(super) consumer: Option<NodeId>,
    pub(super) edge: TileEdge,
}

/// One node in a [`super::TileGraph`]: an [`ExecutableOp`](teeny_core::model::ExecutableOp)'s
/// name and output dtype. Shape is deliberately not here — see the module
/// doc comment — nor are producer/consumer edges, which live in the owning
/// [`super::TileGraph`]'s edge arena.
#[derive(Debug, Clone)]
pub struct TileOp {
    /// This op's name, carried over from
    /// [`ExecutableOp::name`](teeny_core::model::ExecutableOp::name). A
    /// lowered `ExecutableOp` doesn't expose the source
    /// [`Op`](teeny_core::graph::Op) enum it came from — `Anduin` runs
    /// after lowering, on a `Dag<Box<dyn ExecutableOp>>` — so any future
    /// pass that needs to branch on op kind should match on this name (or a
    /// new `ExecutableOp` method), not on `Op`.
    pub name: String,
    /// Output dtype, carried over from
    /// [`ExecutableOp::output_dtype`](teeny_core::model::ExecutableOp::output_dtype).
    pub dtype: DtypeRepr,
    /// Declarative tile-shape metadata, carried over from
    /// [`ExecutableOp::tile_spec`](teeny_core::model::ExecutableOp::tile_spec).
    /// `None` for the vast majority of ops (coverage is opt-in) —
    /// [`super::TileGraph::propagate`] treats a missing spec as a hard
    /// boundary.
    pub tile_spec: Option<KernelTileSpec>,
}

/// Welder §3.2's `Propagate` output (Fig. 6): the tile shape required on
/// each edge to satisfy a target output tile, back-propagated through a
/// node set — see [`super::TileGraph::propagate`].
///
/// Keyed by [`EdgeId`], not [`NodeId`] — deliberately, to preserve the same
/// per-edge flexibility [`TileEdge::memory_level`] already has (a producer
/// with two consumers can legitimately need a different tile shape on each
/// outgoing edge; collapsing to one shape per node would silently force
/// them to agree). Because every incoming edge has exactly one consumer,
/// `propagate` never needs to reconcile two different requests for the same
/// edge — each edge gets written at most once.
#[derive(Debug, Clone, Default)]
pub struct TileConfig {
    pub(super) tiles: HashMap<EdgeId, TileEdgeShape>,
}

impl TileConfig {
    /// The tile shape resolved for `id`, if `propagate` reached it.
    pub fn get(&self, id: EdgeId) -> Option<&TileEdgeShape> {
        self.tiles.get(&id)
    }

    /// Number of edges this config has a resolved tile shape for.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// True if no edge has a resolved tile shape.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

/// One node of the tile-config search tree [`super::TileGraph::sub_graph_tiling`]
/// (Welder Fig. 7's `SubGraphTiling`) builds: the node set this result
/// covers, a chosen [`TileConfig`] for it at one memory level, plus its own
/// recursively-tiled `children` — one per distinct subgraph
/// [`super::TileGraph::extract_subgraph`] finds one memory level up, each
/// child's own `nodes` naming exactly which of `nodes` it covers (needed to
/// walk this tree at all — see
/// [`Trace::trace_graph`](super::super::trace::Trace::trace_graph)).
/// `children` is empty at the top of the memory hierarchy, where the
/// recursion terminates.
#[derive(Debug, Clone)]
pub struct SubGraphTilingResult {
    pub nodes: Vec<NodeId>,
    pub config: TileConfig,
    pub children: Vec<SubGraphTilingResult>,
}
