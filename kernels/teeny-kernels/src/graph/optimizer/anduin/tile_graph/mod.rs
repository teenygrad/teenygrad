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

//! `TileGraph` — the Welder-style DAG Anduin schedules over.
//!
//! [`TileGraph::from_dag`] converts an already-lowered
//! `Dag<Box<dyn ExecutableOp>>` (exactly what
//! [`GraphOptimizer::optimize`](crate::graph::optimizer::GraphOptimizer::optimize)
//! receives) into the same DAG shape with [`TileOp`] nodes in place of
//! [`ExecutableOp`]s. It is a pure structural conversion: producer/consumer
//! edges are carried over one-to-one from the source `Dag`'s own
//! `parents`/`children`. Tile-shape propagation (backward from the graph
//! output, as expressions in shared free variables) and the
//! memory-hierarchy-level search are later passes — see this module's
//! parent doc comment.
//!
//! Shape is an edge concept, not a node concept: a `TileOp` is just an
//! operation, and shape describes a *value* flowing on an edge, which is
//! also exactly what a later tiling pass needs to refine per-edge (the same
//! producer can be tiled differently for different consumers). So shape
//! lives on [`TileEdge`], never on [`TileOp`] — as [`TileEdgeShape`], not the
//! ordinary [`Shape`](teeny_core::graph::Shape): a tile-graph axis can be
//! [`TileDim::Sym`], a named free variable, not just a known-or-dynamic
//! extent. `from_dag` synthesizes one fresh symbol per dynamic axis of the
//! source shape; it does not unify symbols that turn out to be the same free
//! variable across edges — that's the later propagation pass's job.
//!
//! ## Edge arena
//!
//! Every [`TileEdge`] lives exactly once, in the `edges` arena, addressed by
//! an opaque [`EdgeId`]. Each node keeps two lists of `EdgeId`s — `outgoing`
//! (edges it produces) and `incoming` (edges it consumes) — rather than
//! embedding edge data in one direction only. An earlier version of this
//! type stored `TileEdge`s inline in a per-producer `children` list and kept
//! `parents` as plain indices with no edge data at all: that made a
//! consumer-to-producer edge lookup an O(fan-out) scan with no way to
//! mutate it, and had no uniform way to address a graph-boundary edge
//! alongside an internal one. The arena fixes both: mutating an edge (e.g.
//! the scheduler's `SetConnect`, Welder §3.2) is an O(1) write through its
//! `EdgeId` that's immediately visible from both the producer's `outgoing`
//! and the consumer's `incoming` list, because both just hold the same id
//! into the same slot — there is nothing to keep in sync.
//!
//! A graph-boundary edge is simply an edge with one endpoint set to `None`:
//! an input edge (`producer: None`) only appears in its node's `incoming`
//! list, an output edge (`consumer: None`) only appears in its node's
//! `outgoing` list. This is why [`EdgeId`] alone is enough to address any
//! edge — internal or boundary — with no separate edge-kind enum needed.
//!
//! Every node's output shape needs a home on some edge, including nodes at
//! the DAG's boundary that have no producer or consumer in `dag`:
//! - [`TileGraph::parents`] mirrors the source `Dag` node's own `parents`
//!   field: plain, *deduped* producer indices. `Dag::add_edge` already
//!   collapses a repeated `(producer, consumer)` pair before `from_dag` ever
//!   sees it, so an op that reads the same producer through two operand
//!   slots (e.g. `Add(x, x)`) is indistinguishable here from one that reads
//!   it once — that per-operand-slot detail lived in the source
//!   [`teeny_core::graph::Graph`]'s `inputs` list and isn't reconstructible
//!   from `Dag` alone. If a later pass needs it, expose it as a new
//!   [`ExecutableOp`] method instead of threading `Graph` back in here —
//!   lowering has already happened by the time `Anduin` runs.
//! - [`TileGraph::children`] is the fanout view: which distinct consumers
//!   read node `i`'s output, and the [`EdgeId`] of each edge. One entry per
//!   *consumer*, not per operand slot, for the same reason as `parents`.
//! - [`TileGraph::input_edge`]/[`TileGraph::input_edge_id`] are `Some` iff
//!   node `i` has no producer in `dag` (empty `parents(i)`, e.g. a lowered
//!   `Input` op): a boundary edge carrying that node's shape in from outside
//!   the DAG.
//! - [`TileGraph::output_edge`]/[`TileGraph::output_edge_id`] are `Some` iff
//!   node `i` has no consumer in `dag` (empty `children(i)`, i.e. nothing
//!   else in this DAG reads it): a boundary edge carrying that node's shape
//!   out to the DAG's caller.

use std::collections::{HashMap, HashSet};

use teeny_core::device::hardware::{HardwareProfile, MemoryLevelKind};
use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::{ExecutableOp, KernelTileSpec};
use teeny_core::utils::dag::Dag;

/// Index of a node in a [`TileGraph`], stable for the lifetime of that
/// graph (assigned by [`TileGraph::from_dag`]/[`TileGraph::extract_subgraph`]
/// in the order their nodes are visited). A plain alias, not a newtype: it's
/// interchangeable with the source `Dag`'s own node indices, which is what
/// lets `from_dag` carry them over unchanged.
pub type NodeId = usize;

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
fn to_tile_shape(node_index: NodeId, shape: &Shape) -> TileEdgeShape {
    shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| match dim {
            Some(extent) => TileDim::Fixed(*extent),
            None => TileDim::Sym(format!("n{node_index}d{axis}")),
        })
        .collect()
}

/// One edge in a [`TileGraph`]: the shape and memory level of the value it
/// carries. Used both for internal producer→consumer edges and for
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
    /// [`TileGraph::mem_footprint_with_config`]/
    /// [`TileGraph::mem_traffic_with_config`] instead when a
    /// [`TileConfig`] has a tiled (smaller) shape for this edge.
    pub fn byte_size(&self, dtype: DtypeRepr) -> u64 {
        shape_byte_size(&self.shape, dtype)
    }
}

/// Byte size of `shape` (element count × dtype size), the shared core of
/// [`TileEdge::byte_size`] and every config-aware footprint/traffic
/// computation. A dynamic ([`TileDim::Sym`]) axis counts as extent 1 — see
/// [`TileEdge::byte_size`]'s doc comment.
fn shape_byte_size(shape: &TileEdgeShape, dtype: DtypeRepr) -> u64 {
    let elements: u64 = shape
        .iter()
        .map(|dim| match dim {
            TileDim::Fixed(extent) => *extent as u64,
            TileDim::Sym(_) => 1,
        })
        .product();
    elements * dtype_bytes(dtype)
}

/// Upper bound on tiles visited by [`TileGraph::enumerate_subtiles`]'s
/// expanding search, mirroring Welder's own `DFS_smem_tile` visited-tile
/// cap (2000) — kept smaller here since a power-of-two-only search space
/// is already much smaller than Welder's any-divisor one.
const MAX_ENUMERATED_TILES: usize = 512;

/// `1, 2, 4, ..., extent.next_power_of_two()` — the candidate tile-size
/// ladder for one [`TileDim::Fixed`] axis in
/// [`TileGraph::enumerate_subtiles`]. See that method's doc comment for
/// why powers of two, not arbitrary divisors.
fn power_of_two_ladder(extent: usize) -> Vec<usize> {
    let top = extent.max(1).next_power_of_two();
    let mut ladder = Vec::new();
    let mut step = 1usize;
    loop {
        ladder.push(step);
        if step >= top {
            break;
        }
        step *= 2;
    }
    ladder
}

/// Applies a [`TileAxisBinding::divide_by`](teeny_core::model::TileAxisBinding::divide_by)
/// divisor to `dim`, if both are present (teenygrad-1nr.10) -- a
/// [`TileDim::Sym`] (dynamic) axis, or `divide_by: None`, passes through
/// unchanged (`.clone()`d, since the caller always wants an owned value
/// back here, not a borrow of `dim`).
fn apply_divide_by(dim: &TileDim, divide_by: Option<usize>) -> TileDim {
    match (dim, divide_by) {
        (TileDim::Fixed(extent), Some(divisor)) if divisor > 0 => TileDim::Fixed(extent / divisor),
        _ => dim.clone(),
    }
}

/// One [`TileGraph::enumerate_subtiles`] search axis: either an ordinary
/// single real dim, or (when `root`'s tile_spec declares a
/// [`TileAxisBinding`](teeny_core::model::TileAxisBinding) whose `dims`
/// spans more than one real axis — teenygrad-1nr.8/.9) a flattened group
/// of them, searched as one combined ladder.
struct SearchAxis {
    /// Real dims this axis spans, outer to inner (mirrors
    /// [`TileAxisBinding::dims`](teeny_core::model::TileAxisBinding::dims)).
    /// Exactly one entry for the ordinary single-dim case.
    dims: Vec<usize>,
    /// Candidate values for the innermost dim. A single, unchanging
    /// entry means "not enumerable" (a [`TileDim::Sym`] axis, or a
    /// flattened group involving one).
    ladder: Vec<TileDim>,
}

impl SearchAxis {
    /// Writes `value` (one of `self.ladder`'s entries) into `candidate`:
    /// the innermost dim gets `value`, every other spanned dim collapses
    /// to `Fixed(1)` — product-preserving, the same convention
    /// [`TileGraph::propagate`] already applies downstream (see
    /// [`TileAxisBinding::dims`](teeny_core::model::TileAxisBinding::dims)'s
    /// doc comment). A no-op on any dim beyond `candidate`'s own length.
    fn write(&self, candidate: &mut TileEdgeShape, value: &TileDim) {
        let Some((&innermost, outer_dims)) = self.dims.split_last() else {
            return;
        };
        if let Some(dim) = candidate.get_mut(innermost) {
            *dim = value.clone();
        }
        for &outer in outer_dims {
            if let Some(dim) = candidate.get_mut(outer) {
                *dim = TileDim::Fixed(1);
            }
        }
    }
}

fn dtype_bytes(dtype: DtypeRepr) -> u64 {
    match dtype {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

/// Opaque handle to one edge in a [`TileGraph`]'s arena. Addresses an
/// internal or a graph-boundary edge uniformly — see the module doc
/// comment. Obtained from [`TileGraph::children`],
/// [`TileGraph::input_edge_id`], or [`TileGraph::output_edge_id`]; dereferenced
/// via [`TileGraph::edge`]/[`TileGraph::connect_level`] and mutated via
/// [`TileGraph::set_connect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(usize);

/// One arena slot: a [`TileEdge`] plus the node indices of its endpoints.
/// `producer`/`consumer` are `None` exactly when this edge is a
/// graph-boundary edge on that side — see the module doc comment.
#[derive(Debug, Clone)]
struct TileEdgeRecord {
    producer: Option<NodeId>,
    consumer: Option<NodeId>,
    edge: TileEdge,
}

/// Pushes one edge into the arena and links it into `producer`'s `outgoing`
/// list and/or `consumer`'s `incoming` list, whichever side(s) are `Some`.
fn push_edge(
    edges: &mut Vec<TileEdgeRecord>,
    outgoing: &mut [Vec<EdgeId>],
    incoming: &mut [Vec<EdgeId>],
    producer: Option<NodeId>,
    consumer: Option<NodeId>,
    edge: TileEdge,
) {
    let id = EdgeId(edges.len());
    edges.push(TileEdgeRecord {
        producer,
        consumer,
        edge,
    });
    if let Some(p) = producer {
        outgoing[p].push(id);
    }
    if let Some(c) = consumer {
        incoming[c].push(id);
    }
}

/// One node in a [`TileGraph`]: an [`ExecutableOp`]'s name and output dtype.
/// Shape is deliberately not here — see the module doc comment — nor are
/// producer/consumer edges, which live in the owning [`TileGraph`]'s edge
/// arena.
#[derive(Debug, Clone)]
pub struct TileOp {
    /// This op's name, carried over from [`ExecutableOp::name`]. A lowered
    /// `ExecutableOp` doesn't expose the source
    /// [`Op`](teeny_core::graph::Op) enum it came from — `Anduin` runs
    /// after lowering, on a `Dag<Box<dyn ExecutableOp>>` — so any future
    /// pass that needs to branch on op kind should match on this name (or a
    /// new `ExecutableOp` method), not on `Op`.
    pub name: String,
    /// Output dtype, carried over from [`ExecutableOp::output_dtype`].
    pub dtype: DtypeRepr,
    /// Declarative tile-shape metadata, carried over from
    /// [`ExecutableOp::tile_spec`]. `None` for the vast majority of ops
    /// (coverage is opt-in) — [`TileGraph::propagate`] treats a missing
    /// spec as a hard boundary.
    pub tile_spec: Option<KernelTileSpec>,
}

/// Welder §3.2's `Propagate` output (Fig. 6): the tile shape required on
/// each edge to satisfy a target output tile, back-propagated through a
/// node set — see [`TileGraph::propagate`].
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
    tiles: HashMap<EdgeId, TileEdgeShape>,
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

/// One node of the tile-config search tree [`TileGraph::sub_graph_tiling`]
/// (Welder Fig. 7's `SubGraphTiling`) builds: the node set this result
/// covers, a chosen [`TileConfig`] for it at one memory level, plus its own
/// recursively-tiled `children` — one per distinct subgraph
/// [`TileGraph::extract_subgraph`] finds one memory level up, each child's
/// own `nodes` naming exactly which of `nodes` it covers (needed to walk
/// this tree at all — see [`Trace::trace_graph`](super::trace::Trace::trace_graph)).
/// `children` is empty at the top of the memory hierarchy, where the
/// recursion terminates.
#[derive(Debug, Clone)]
pub struct SubGraphTilingResult {
    pub nodes: Vec<NodeId>,
    pub config: TileConfig,
    pub children: Vec<SubGraphTilingResult>,
}

/// An already-lowered `Dag<Box<dyn ExecutableOp>>` converted to Welder's
/// tile-graph form: same DAG structure, one [`TileOp`] per source node, and
/// every edge held once in an arena addressed by [`EdgeId`] — see the module
/// doc comment.
#[derive(Debug, Default)]
pub struct TileGraph {
    nodes: Vec<TileOp>,
    edges: Vec<TileEdgeRecord>,
    outgoing: Vec<Vec<EdgeId>>,
    incoming: Vec<Vec<EdgeId>>,
    /// Boundary edges for each node's *additional* declared outputs
    /// beyond the primary one (index `i` holds `spec.outputs[i + 1]`'s
    /// edge) -- see [`Self::secondary_output_edges`] (teenygrad-1nr.11).
    secondary_outputs: Vec<Vec<EdgeId>>,
    /// The winning [`SubGraphTilingResult`] `schedule_graph`
    /// (`anduin::schedule`) found for whichever edge it was deciding when
    /// it produced it, keyed by that [`EdgeId`]. Empty until
    /// `schedule_graph` runs. See [`Self::record_resolved_tiling`]/
    /// [`Self::resolved_tiling`].
    resolved_tiling: HashMap<EdgeId, SubGraphTilingResult>,
}

impl TileGraph {
    /// Converts `dag` into a `TileGraph` with identical DAG structure: each
    /// `dag` node becomes one [`TileOp`], producer/consumer edges are
    /// carried over verbatim from `dag`'s own `parents`/`children`, and
    /// boundary edges are synthesized for nodes with no producer/consumer in
    /// `dag`. Every edge starts at [`MemoryLevelKind::DeviceMemory`]: every
    /// tensor starts out materialized in device memory until the
    /// memory-level search promotes an edge to a faster level.
    pub fn from_dag(dag: &Dag<Box<dyn ExecutableOp>>) -> Self {
        let n = dag.len();
        let mut nodes = Vec::with_capacity(n);
        let mut edges: Vec<TileEdgeRecord> = Vec::new();
        let mut outgoing: Vec<Vec<EdgeId>> = vec![Vec::new(); n];
        let mut incoming: Vec<Vec<EdgeId>> = vec![Vec::new(); n];
        let mut secondary_outputs: Vec<Vec<EdgeId>> = Vec::with_capacity(n);

        for index in 0..n {
            let node = dag.node(index);
            let shape = node.value.output_shape();
            let tile_spec = node.value.tile_spec();

            nodes.push(TileOp {
                name: node.value.name().to_string(),
                dtype: node.value.output_dtype(),
                tile_spec,
            });

            // Additional declared outputs beyond the primary one
            // (teenygrad-1nr.11): no ground-truth shape exists for these
            // (ExecutableOp::output_shape is singular), so every axis is
            // an unresolved placeholder symbol -- these edges exist only
            // so a caller can seed a requested tile onto them for
            // Self::propagate to pick up; see
            // Self::secondary_output_edges's doc comment for why they're
            // deliberately not linked into outgoing/incoming. Pushed in
            // node-index order (matching `nodes`/`outgoing`/`incoming`),
            // so `secondary_outputs[index]` lines up without ever
            // indexing it directly here.
            let mut node_secondary_outputs = Vec::new();
            if let Some(spec) = &tile_spec {
                for (output_index, output_spec) in spec.outputs.iter().enumerate().skip(1) {
                    let placeholder_shape: TileEdgeShape = (0..output_spec.rank)
                        .map(|axis| TileDim::Sym(format!("n{index}o{output_index}d{axis}")))
                        .collect();
                    let id = EdgeId(edges.len());
                    edges.push(TileEdgeRecord {
                        producer: Some(index),
                        consumer: None,
                        edge: TileEdge {
                            shape: placeholder_shape,
                            memory_level: MemoryLevelKind::DeviceMemory,
                        },
                    });
                    node_secondary_outputs.push(id);
                }
            }
            secondary_outputs.push(node_secondary_outputs);

            if node.parents.is_empty() {
                push_edge(
                    &mut edges,
                    &mut outgoing,
                    &mut incoming,
                    None,
                    Some(index),
                    TileEdge {
                        shape: to_tile_shape(index, shape),
                        memory_level: MemoryLevelKind::DeviceMemory,
                    },
                );
            }

            if node.children.is_empty() {
                push_edge(
                    &mut edges,
                    &mut outgoing,
                    &mut incoming,
                    Some(index),
                    None,
                    TileEdge {
                        shape: to_tile_shape(index, shape),
                        memory_level: MemoryLevelKind::DeviceMemory,
                    },
                );
            } else {
                for &consumer in &node.children {
                    push_edge(
                        &mut edges,
                        &mut outgoing,
                        &mut incoming,
                        Some(index),
                        Some(consumer),
                        TileEdge {
                            shape: to_tile_shape(index, shape),
                            memory_level: MemoryLevelKind::DeviceMemory,
                        },
                    );
                }
            }
        }

        Self {
            nodes,
            edges,
            outgoing,
            incoming,
            secondary_outputs,
            resolved_tiling: HashMap::new(),
        }
    }

    /// Number of tile nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True if this tile graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The tile node at `index`.
    pub fn node(&self, index: NodeId) -> &TileOp {
        &self.nodes[index]
    }

    /// Deduped producer indices for the node at `index` (see the module doc
    /// comment for why duplicate operand slots aren't distinguishable here).
    pub fn parents(&self, index: NodeId) -> Vec<NodeId> {
        self.parent_edges(index)
            .into_iter()
            .map(|(producer, _)| producer)
            .collect()
    }

    /// Producers feeding the node at `index`, as `(producer_index, EdgeId)`
    /// pairs — the incoming-edge mirror of [`Self::children`]. Dereference
    /// the id via [`Self::edge`] to read the edge.
    pub fn parent_edges(&self, index: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.incoming[index]
            .iter()
            .filter_map(|&id| self.edges[id.0].producer.map(|producer| (producer, id)))
            .collect()
    }

    /// Consumers of the node at `index`'s output, as `(consumer_index,
    /// EdgeId)` pairs. One entry per distinct consumer — see the module doc
    /// comment. Dereference the id via [`Self::edge`] to read the edge, or
    /// mutate it via [`Self::set_connect`].
    pub fn children(&self, index: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.outgoing[index]
            .iter()
            .filter_map(|&id| self.edges[id.0].consumer.map(|consumer| (consumer, id)))
            .collect()
    }

    /// The id of the boundary edge carrying node `index`'s value in from
    /// outside the DAG, if it has no producer in `dag` (empty
    /// `parents(index)`).
    pub fn input_edge_id(&self, index: NodeId) -> Option<EdgeId> {
        self.incoming[index]
            .iter()
            .copied()
            .find(|&id| self.edges[id.0].producer.is_none())
    }

    /// The id of the boundary edge carrying node `index`'s value out to the
    /// DAG's caller, if it has no consumer in `dag` (empty
    /// `children(index)`).
    pub fn output_edge_id(&self, index: NodeId) -> Option<EdgeId> {
        self.outgoing[index]
            .iter()
            .copied()
            .find(|&id| self.edges[id.0].consumer.is_none())
    }

    /// The boundary edge carrying node `index`'s value in from outside the
    /// DAG, if it has no producer in `dag`. Convenience wrapper over
    /// [`Self::input_edge_id`] + [`Self::edge`].
    pub fn input_edge(&self, index: NodeId) -> Option<&TileEdge> {
        self.input_edge_id(index).map(|id| self.edge(id))
    }

    /// The boundary edge carrying node `index`'s value out to the DAG's
    /// caller, if it has no consumer in `dag`. Convenience wrapper over
    /// [`Self::output_edge_id`] + [`Self::edge`].
    pub fn output_edge(&self, index: NodeId) -> Option<&TileEdge> {
        self.output_edge_id(index).map(|id| self.edge(id))
    }

    /// Boundary edges for `index`'s *additional* declared outputs beyond
    /// the primary one ([`Self::output_edge_id`]) -- one per
    /// `tile_spec.outputs[1..]`, in order, synthesized by
    /// [`Self::from_dag`] when `index`'s [`TileOp::tile_spec`] declares
    /// more than one output (teenygrad-1nr.11: `flash_attn2`'s real
    /// `o_ptr`+`l_ptr`, `group_norm_forward`'s `y_ptr`/`mean_ptr`/
    /// `rstd_ptr`).
    ///
    /// Deliberately **not** linked into [`Self::children`]/
    /// [`Self::parent_edges`]/[`Self::extract_subgraph`]/
    /// [`Self::boundary_edges`] (i.e. not in `outgoing`/`incoming` at
    /// all) -- invisible to everything except [`Self::propagate`]. There
    /// is no ground-truth shape for these (`ExecutableOp::output_shape`
    /// is singular; every axis here is an unresolved
    /// [`TileDim::Sym`] placeholder), so wiring them into cost-model or
    /// scheduling passes designed around exactly one real output per
    /// node needs its own follow-up, not attempted here. A caller who
    /// wants [`Self::propagate`] to resolve one of these must seed a
    /// requested tile onto its `EdgeId` via `propagate`'s own
    /// `output_tiles` map -- [`Self::edge`]/[`Self::set_connect`] still
    /// work on it like any other arena edge.
    pub fn secondary_output_edges(&self, index: NodeId) -> &[EdgeId] {
        &self.secondary_outputs[index]
    }

    /// The edge identified by `id`.
    pub fn edge(&self, id: EdgeId) -> &TileEdge {
        &self.edges[id.0].edge
    }

    /// The memory level `id`'s edge is currently connected at. Shorthand for
    /// `self.edge(id).memory_level`.
    pub fn connect_level(&self, id: EdgeId) -> MemoryLevelKind {
        self.edge(id).memory_level
    }

    /// Sets the memory level `id`'s edge is materialized at (Welder §3.2's
    /// `SetConnect`). The change is visible immediately from both the
    /// producer's and the consumer's side, since both address the same
    /// arena slot — see the module doc comment.
    pub fn set_connect(&mut self, id: EdgeId, level: MemoryLevelKind) {
        self.edges[id.0].edge.memory_level = level;
    }

    /// Records `result` as the winning [`SubGraphTilingResult`] found while
    /// deciding `id`'s connect level (`anduin::schedule::schedule_graph`).
    /// Overwrites whatever was previously recorded for `id`.
    pub fn record_resolved_tiling(&mut self, id: EdgeId, result: SubGraphTilingResult) {
        self.resolved_tiling.insert(id, result);
    }

    /// The winning [`SubGraphTilingResult`] recorded for `id` via
    /// [`Self::record_resolved_tiling`], if `schedule_graph` has run and
    /// found one.
    pub fn resolved_tiling(&self, id: EdgeId) -> Option<&SubGraphTilingResult> {
        self.resolved_tiling.get(&id)
    }

    /// Welder §3.2's `ExtractSubgraph`: the node set reachable from `node`
    /// by transitively following edges (in either direction) whose connect
    /// level is *above* `level` — i.e. everything `GraphConnecting` has
    /// already decided to fuse more tightly than `level`. Matches Fig. 7's
    /// `SubGraph(nodes)`: a set of [`NodeId`]s into *this* graph, not a
    /// separate rebuilt one — downstream cost-model passes (`MemFootprint`,
    /// `MemTraffic`, ...) can test edge endpoints against the set directly
    /// rather than needing copied-and-relabeled edge data.
    ///
    /// `level` is `None` for "nothing decided yet" (the walk's own
    /// starting point, before any real level has been examined) — every
    /// real edge counts as still-fused in that case, since nothing has
    /// had the chance to cut it yet. `Some(level)` only follows an edge
    /// connected at a real level above it.
    ///
    /// Unlike the paper's naive tree recursion, this shares one visited set
    /// across the whole walk: two paths converging on the same node (e.g. a
    /// diamond fan-out/fan-in) expand it once, not once per incoming path.
    ///
    /// `node` is always included, even in isolation (e.g. every one of its
    /// edges is at or below `level`) — the returned set is never empty.
    pub fn extract_subgraph(&self, node: NodeId, level: Option<MemoryLevelKind>) -> Vec<NodeId> {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = vec![node];
        visited[node] = true;

        while let Some(current) = stack.pop() {
            for &id in self.outgoing[current]
                .iter()
                .chain(self.incoming[current].iter())
            {
                let record = &self.edges[id.0];
                if level.is_some_and(|level| record.edge.memory_level <= level) {
                    continue;
                }
                let other = if record.producer == Some(current) {
                    record.consumer
                } else {
                    record.producer
                };
                if let Some(other) = other
                    && !visited[other]
                {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }

        (0..self.nodes.len())
            .filter(|&index| visited[index])
            .collect()
    }

    /// Every edge with exactly one endpoint in `nodes` — the boundary of
    /// the fused unit `nodes` represents — paired with the dtype of
    /// whichever side produced the data on it. An edge with both endpoints
    /// in `nodes` is internal and is not included. Shared by
    /// [`Self::mem_traffic`] and `anduin`'s `SimpleProfiler`, which both
    /// need exactly this set.
    pub(crate) fn boundary_edges(&self, nodes: &[NodeId]) -> Vec<(EdgeId, DtypeRepr)> {
        let included: HashSet<NodeId> = nodes.iter().copied().collect();
        let mut result = Vec::new();

        for &node in nodes {
            let dtype = self.node(node).dtype;

            for (consumer, edge_id) in self.children(node) {
                if !included.contains(&consumer) {
                    result.push((edge_id, dtype));
                }
            }
            if let Some(edge_id) = self.output_edge_id(node) {
                result.push((edge_id, dtype));
            }

            for (producer, edge_id) in self.parent_edges(node) {
                if !included.contains(&producer) {
                    result.push((edge_id, self.node(producer).dtype));
                }
            }
            if let Some(edge_id) = self.input_edge_id(node) {
                result.push((edge_id, dtype));
            }
        }

        result
    }

    /// `id`'s byte size: `config`'s tiled shape for `id` if present,
    /// otherwise `id`'s own full, untiled shape — the one place every
    /// config-aware footprint/traffic computation resolves "tiled if we
    /// know it, full otherwise".
    fn edge_byte_size(&self, id: EdgeId, dtype: DtypeRepr, config: Option<&TileConfig>) -> u64 {
        let shape = config
            .and_then(|config| config.get(id))
            .unwrap_or(&self.edge(id).shape);
        shape_byte_size(shape, dtype)
    }

    /// Welder §3.1's `MemTraffic`: total bytes crossing the boundary of
    /// `nodes` (see [`Self::boundary_edges`]), using each edge's full
    /// untiled shape.
    pub fn mem_traffic(&self, nodes: &[NodeId]) -> u64 {
        self.mem_traffic_impl(nodes, None)
    }

    /// Like [`Self::mem_traffic`], but uses `config`'s tiled shape for any
    /// boundary edge it resolves (falling back to that edge's full shape
    /// otherwise) — the accurate version once a [`TileConfig`] exists for
    /// `nodes` (e.g. from [`Self::propagate`]).
    pub fn mem_traffic_with_config(&self, nodes: &[NodeId], config: &TileConfig) -> u64 {
        self.mem_traffic_impl(nodes, Some(config))
    }

    fn mem_traffic_impl(&self, nodes: &[NodeId], config: Option<&TileConfig>) -> u64 {
        self.boundary_edges(nodes)
            .into_iter()
            .map(|(edge_id, dtype)| self.edge_byte_size(edge_id, dtype, config))
            .sum()
    }

    /// The byte size of `node`'s own output tile, read from whichever of
    /// its outgoing edges exists (they all carry the same shape — see the
    /// module doc comment) — an output boundary edge if it has no consumer
    /// in `dag`, otherwise its first child edge. `config`'s tiled shape is
    /// used when present, same as [`Self::edge_byte_size`].
    fn output_byte_size(&self, node: NodeId, config: Option<&TileConfig>) -> u64 {
        let dtype = self.node(node).dtype;
        self.output_edge_id(node)
            .or_else(|| self.children(node).first().map(|&(_, id)| id))
            .map(|id| self.edge_byte_size(id, dtype, config))
            .unwrap_or(0)
    }

    /// Welder §3.1's `MemFootprint`: an estimate of the peak memory
    /// resident while executing `nodes` in topological order. Each node's
    /// own output tile becomes live the moment it's produced; it's freed
    /// once its last in-set consumer has run, *unless* it also crosses the
    /// boundary of `nodes` (an output boundary edge, or a consumer outside
    /// `nodes`), in which case it's conservatively kept live for the rest
    /// of this simulation, since an external reader could need it at any
    /// point. Each node's *external* inputs (a producer outside `nodes`, or
    /// a graph-boundary input edge) are transient: resident only while that
    /// one node runs.
    ///
    /// This is a peak-live-set estimate, not a literal simulation of the
    /// paper's best-fit allocator — a real allocator can pack a smaller
    /// footprint by reusing freed space below this peak, so this is an
    /// upper bound, not a minimum. It also assumes one materialization per
    /// node regardless of per-edge connect-level divergence (see
    /// [`TileEdge`]'s doc comment on why levels can legitimately differ per
    /// consumer) — a node whose edges end up at genuinely different levels
    /// could in principle need more than one buffer, which this doesn't
    /// model yet. Uses each edge's full untiled shape — see
    /// [`Self::mem_footprint_with_config`] for the tiled version.
    pub fn mem_footprint(&self, nodes: &[NodeId]) -> u64 {
        self.mem_footprint_impl(nodes, None)
    }

    /// Like [`Self::mem_footprint`], but uses `config`'s tiled shape for
    /// any edge it resolves (falling back to that edge's full shape
    /// otherwise).
    pub fn mem_footprint_with_config(&self, nodes: &[NodeId], config: &TileConfig) -> u64 {
        self.mem_footprint_impl(nodes, Some(config))
    }

    fn mem_footprint_impl(&self, nodes: &[NodeId], config: Option<&TileConfig>) -> u64 {
        let included: HashSet<NodeId> = nodes.iter().copied().collect();

        let mut remaining_consumers: HashMap<NodeId, usize> = nodes
            .iter()
            .map(|&node| {
                let count = self
                    .children(node)
                    .iter()
                    .filter(|(consumer, _)| included.contains(consumer))
                    .count();
                (node, count)
            })
            .collect();

        let crosses_boundary = |node: NodeId| -> bool {
            self.output_edge(node).is_some()
                || self
                    .children(node)
                    .iter()
                    .any(|(consumer, _)| !included.contains(consumer))
        };

        let mut live: HashMap<NodeId, u64> = HashMap::new();
        let mut total_live: u64 = 0;
        let mut peak: u64 = 0;

        for node in self.topological_sort() {
            if !included.contains(&node) {
                continue;
            }
            let dtype = self.node(node).dtype;

            let input_bytes: u64 = self
                .parent_edges(node)
                .into_iter()
                .filter(|(producer, _)| !included.contains(producer))
                .map(|(producer, id)| self.edge_byte_size(id, self.node(producer).dtype, config))
                .sum::<u64>()
                + self
                    .input_edge_id(node)
                    .map(|id| self.edge_byte_size(id, dtype, config))
                    .unwrap_or(0);

            let out_bytes = self.output_byte_size(node, config);

            peak = peak.max(total_live + input_bytes + out_bytes);

            total_live += out_bytes;
            live.insert(node, out_bytes);

            for producer in self.parents(node) {
                if !included.contains(&producer) {
                    continue;
                }
                if let Some(count) = remaining_consumers.get_mut(&producer) {
                    *count -= 1;
                    if *count == 0
                        && !crosses_boundary(producer)
                        && let Some(bytes) = live.remove(&producer)
                    {
                        total_live -= bytes;
                    }
                }
            }
        }

        peak
    }

    /// Any one of `node`'s outgoing edges' shapes — internal or boundary,
    /// it doesn't matter which, since every outgoing edge of a node carries
    /// that node's own (untiled) output shape (see the module doc
    /// comment). Every node has at least one outgoing edge by construction
    /// (`from_dag` always gives a childless node a boundary output edge).
    fn node_output_shape(&self, node: NodeId) -> &TileEdgeShape {
        let &id = self.outgoing[node]
            .first()
            .expect("every node has at least one outgoing edge");
        &self.edges[id.0].edge.shape
    }

    /// Welder §3.2's `Propagate` (Fig. 6): back-propagates a target output
    /// tile through `nodes`, resolving the tile shape required on every
    /// reachable edge. `output_tiles` seeds the tile shape requested on
    /// `nodes`'s own boundary output edges (typically from
    /// [`Self::boundary_edges`]/[`Self::output_edge_id`]) — this is our
    /// analogue of Welder's `Map<Axis, Dim> config`.
    ///
    /// Unlike Welder's own reference implementation (which walks a symbolic
    /// tensor-expression IR we don't have — see
    /// `TILE_GRAPH_SCHEDULING_PLAN.md`), this resolves axes by declared
    /// name: each node's [`TileOp::tile_spec`] names its axes with
    /// `extent_param` strings, and any two axes sharing a name (anywhere,
    /// input or output, any tensor) are the same free variable. Seeding a
    /// node's output tile resolves every axis on its parent edges that
    /// shares one of those names; an axis whose name never appears in the
    /// output (e.g. a reduction axis) is *correctly* left unresolved — its
    /// tile size isn't derivable from the output alone, so this falls back
    /// to that axis's full, untiled extent (never smaller than needed,
    /// same optimistic-not-pessimistic philosophy as
    /// [`TileEdge::byte_size`]). The same fallback covers a dim with no
    /// [`TileAxisBinding`](teeny_core::model::TileAxisBinding) at all —
    /// [`TensorTileSpec::untiled_dims`](teeny_core::model::TensorTileSpec::untiled_dims)'s
    /// case — since every input/output tile is built at that tensor's
    /// full [`TensorTileSpec::rank`](teeny_core::model::TensorTileSpec::rank),
    /// not `axes.len()`, with only the dims `axes` actually names
    /// overwritten.
    ///
    /// Degrades to a hard boundary — stops there, doesn't guess — at any
    /// node with no declared `tile_spec` (the overwhelming majority; see
    /// [`teeny_core::model::KernelTileSpec`]'s doc comment on coverage
    /// being opt-in), whose output tile's axis count doesn't match its
    /// spec's declared output rank, or whose parent-edge count doesn't
    /// match its spec's declared input count (the same positional
    /// producer-to-spec correspondence limitation the design this revives
    /// already had — see that module's doc comment).
    pub fn propagate(
        &self,
        nodes: &[NodeId],
        output_tiles: &HashMap<EdgeId, TileEdgeShape>,
    ) -> TileConfig {
        let included: HashSet<NodeId> = nodes.iter().copied().collect();
        let mut tiles: HashMap<EdgeId, TileEdgeShape> = output_tiles.clone();

        for node in self.topological_sort().into_iter().rev() {
            if !included.contains(&node) {
                continue;
            }
            let Some(spec) = &self.node(node).tile_spec else {
                continue; // hard boundary: no declared spec
            };
            let Some(output_spec) = spec.outputs.first() else {
                continue; // hard boundary: spec declares no output
            };

            // This node's own required output tile: any outgoing edge
            // already resolved (a boundary output edge, or an internal
            // edge to an already-processed in-set consumer), falling back
            // to the full untiled shape if nothing downstream constrained
            // it yet.
            let output_tile: TileEdgeShape = self
                .children(node)
                .iter()
                .map(|&(_, id)| id)
                .chain(self.output_edge_id(node))
                .find_map(|id| tiles.get(&id).cloned())
                .unwrap_or_else(|| self.node_output_shape(node).clone());

            if output_tile.len() != output_spec.rank {
                continue; // hard boundary: declared rank doesn't match reality
            }

            // Seed resolved extent_param -> TileDim from the output tile,
            // indexed by each axis's declared innermost `dims` entry --
            // not position, since `axes` may cover fewer dims than `rank`
            // (see `untiled_dims`) and needn't list them in tensor-dim
            // order. A flattened axis's block-sized value lives on its
            // innermost real dim -- see `TileAxisBinding::dims`'s doc
            // comment. `divide_by` (teenygrad-1nr.10) applies uniformly
            // right here, so every later consumer of this name already
            // sees the divided value -- owned, not borrowed, since a
            // divided value is freshly computed, not a view into
            // `output_tile`.
            let mut resolved: HashMap<&'static str, TileDim> = output_spec
                .axes
                .iter()
                .filter_map(|axis| {
                    let &innermost = axis.dims.last()?;
                    let dim = output_tile.get(innermost)?;
                    Some((axis.extent_param, apply_divide_by(dim, axis.divide_by)))
                })
                .collect();

            // Additional declared outputs (teenygrad-1nr.11): each one
            // whose synthesized boundary edge the caller has seeded
            // contributes more resolved extent_param -> TileDim entries,
            // on top of the primary output's above. Unlike the primary
            // output, there's no full-shape fallback for these -- an
            // unseeded or rank-mismatched secondary output simply
            // contributes nothing, rather than hard-boundarying the
            // whole node the way a bad primary output does above.
            //
            // Collected into owned storage up front (mirroring
            // `output_tile` above) rather than read from `tiles` inline
            // below: `tiles` itself is mutated later in this same node's
            // processing (`tiles.insert` for each parent edge).
            let secondary_tiles: Vec<Option<TileEdgeShape>> = spec
                .outputs
                .iter()
                .enumerate()
                .skip(1)
                .map(|(secondary_index, secondary_spec)| {
                    let &edge_id = self.secondary_output_edges(node).get(secondary_index - 1)?;
                    let tile = tiles.get(&edge_id)?.clone();
                    (tile.len() == secondary_spec.rank).then_some(tile)
                })
                .collect();
            for (secondary_spec, tile) in spec.outputs.iter().skip(1).zip(secondary_tiles.iter()) {
                let Some(tile) = tile else {
                    continue; // not seeded, or declared rank doesn't match reality
                };
                for axis in secondary_spec.axes.iter() {
                    let Some(&innermost) = axis.dims.last() else {
                        continue;
                    };
                    if let Some(dim) = tile.get(innermost) {
                        resolved.insert(axis.extent_param, apply_divide_by(dim, axis.divide_by));
                    }
                }
            }

            let parents = self.parent_edges(node);
            if parents.len() != spec.inputs.len() {
                continue; // hard boundary: positional correspondence unsafe
            }
            for ((_, edge_id), input_spec) in parents.iter().zip(spec.inputs.iter()) {
                let full_shape = &self.edge(*edge_id).shape;
                if full_shape.len() != input_spec.rank {
                    continue; // this input's declared rank doesn't match reality
                }
                // Full rank, not `axes.len()`: a dim with no `TileAxisBinding`
                // (an untiled dim, or a reduction axis with no output-side
                // counterpart) keeps its full extent via this fallback,
                // rather than being dropped from the result entirely.
                let mut input_tile: TileEdgeShape = full_shape.clone();
                for axis in input_spec.axes.iter() {
                    let Some((&innermost, outer_dims)) = axis.dims.split_last() else {
                        continue; // empty dims: nothing to bind (spec-authoring bug)
                    };
                    match resolved.get(axis.extent_param) {
                        Some(resolved_dim) => {
                            if let Some(dim) = input_tile.get_mut(innermost) {
                                *dim = resolved_dim.clone();
                            }
                            // Flattened-away outer dims collapse to 1 --
                            // product-preserving, see
                            // `TileAxisBinding::dims`'s doc comment.
                            for &outer in outer_dims {
                                if let Some(dim) = input_tile.get_mut(outer) {
                                    *dim = TileDim::Fixed(1);
                                }
                            }
                        }
                        None => {
                            // Unresolved: every spanned dim keeps its
                            // full-shape fallback -- except the
                            // innermost, which `divide_by` (if declared)
                            // adjusts from the raw full extent to this
                            // axis's real, usable extent (teenygrad-1nr.10,
                            // e.g. GroupNorm's channels_per_group = C / G).
                            if axis.divide_by.is_some()
                                && let Some(dim) = input_tile.get_mut(innermost)
                            {
                                *dim = apply_divide_by(dim, axis.divide_by);
                            }
                        }
                    }
                }
                tiles.insert(*edge_id, input_tile);
            }
        }

        TileConfig { tiles }
    }

    /// Propagates `tile` as `root`'s requested output shape (seeded on
    /// `output_edge`) through `nodes`, and scores the result as
    /// [`Self::mem_traffic_with_config`] bytes per output element —
    /// `None` if [`Self::mem_footprint_with_config`] exceeds `capacity`
    /// (an invalid candidate). Lower is better. Shared by
    /// [`Self::enumerate_subtiles`]'s base-tile growth and expanding
    /// search, which both need exactly this.
    fn score_candidate_tile(
        &self,
        nodes: &[NodeId],
        output_edge: EdgeId,
        capacity: u64,
        tile: &TileEdgeShape,
    ) -> Option<f64> {
        let mut seed = HashMap::new();
        seed.insert(output_edge, tile.clone());
        let config = self.propagate(nodes, &seed);
        if self.mem_footprint_with_config(nodes, &config) > capacity {
            return None;
        }
        let traffic = self.mem_traffic_with_config(nodes, &config);
        let elements: u64 = tile
            .iter()
            .map(|dim| match dim {
                TileDim::Fixed(extent) => *extent as u64,
                TileDim::Sym(_) => 1,
            })
            .product::<u64>()
            .max(1);
        Some(traffic as f64 / elements as f64)
    }

    /// Welder §4.1's `EnumerateSubtiles`: a Roller-style expanding search
    /// over candidate output-tile shapes for `root` (rank/axes taken from
    /// its own full output shape), for [`Self::sub_graph_tiling`] to
    /// `propagate` and score.
    ///
    /// Candidate sizes per [`TileDim::Fixed`] axis are restricted to
    /// powers of two — `1, 2, 4, ..., extent.next_power_of_two()` — a
    /// hard requirement of this codebase's Triton backend (`tl.arange`
    /// and friends need power-of-two extents; see e.g.
    /// `next_power_of_two` at `TritonLowering`'s softmax/reduction
    /// lowering sites, and the `BLOCK_SIZE`-must-be-a-power-of-two doc
    /// comments across `nn::attention::flash_attn2`, `nn::norm::groupnorm`,
    /// `nn::activation::softmax`, `nn::loss::{embedding,nll,ranking}`).
    /// This is *preferred*, not literally universal: when an axis's
    /// extent isn't itself a power of two, a chosen block size can still
    /// leave a partial/masked last tile when a grid steps across that
    /// axis — this search never needs to compute or represent that
    /// remainder tile itself, only the candidate block size, exactly like
    /// those existing kernels already handle it. A [`TileDim::Sym`]
    /// (dynamic) axis isn't enumerable this way and is left unchanged in
    /// every candidate.
    ///
    /// When `root`'s own `tile_spec` declares a usable output
    /// [`TensorTileSpec`] (its `rank` matching `root`'s real output
    /// rank), the search is driven by that spec's [`TileAxisBinding`]s
    /// instead of raw per-real-dim ladders (teenygrad-1nr.9): each
    /// binding becomes one [`SearchAxis`], whose ladder ranges over the
    /// *combined* extent of every real dim it spans (the product, for a
    /// flattened multi-dim binding — teenygrad-1nr.8's `dims`), written
    /// with the same innermost-gets-the-value/other-spanned-dims-collapse-
    /// to-`Fixed(1)` convention [`Self::propagate`] already applies
    /// downstream — so every candidate this search returns is one
    /// `propagate` (and the real kernel) can actually realize. A real dim
    /// with no axis binding at all (untiled) is never varied, staying at
    /// its full extent in every candidate, matching `propagate`'s own
    /// fallback. Falls back to the previous one-axis-per-real-dim search
    /// when `root` has no usable tile_spec — still the overwhelming
    /// majority of nodes (coverage is opt-in) — a strict,
    /// behavior-preserving extension for every currently-covered
    /// single-dim-axes spec (`Relu`/`MatMul`).
    ///
    /// Unlike Welder's own `DFS_smem_tile` (`../Welder/python/welder/policy/default.py`),
    /// whose candidate steps are *any* divisor of the axis extent (with a
    /// handful of powers of two spliced in only for large primes), this
    /// search's candidate steps are the power-of-two ladder above —
    /// actually a *smaller* search space (`O(log extent)` per axis
    /// instead of `O(divisor count)`).
    ///
    /// Algorithm, adapted from `get_base_tile` + `DFS_smem_tile`:
    /// 1. **Base tile**: starting from all-`Fixed(1)`, grow one axis at a
    ///    time through its ladder while [`Self::score_candidate_tile`]
    ///    (traffic per output element) keeps improving, stopping at the
    ///    first non-improving step — Welder's own `get_base_tile` does the
    ///    same per-axis greedy growth (a "workload per item" metric there;
    ///    `MemTraffic` here, since that's what this search is ultimately
    ///    ranking candidates by anyway).
    /// 2. **Expanding search**: from the base tile, repeatedly take the
    ///    best-scoring visited tile not yet expanded and try bumping each
    ///    axis to its next ladder step, scoring and recording every new
    ///    tile — Welder's own priority-queue neighbor expansion, capped at
    ///    [`MAX_ENUMERATED_TILES`] visited tiles (same idea as Welder's own
    ///    2000-tile cap, just a smaller bound given this search space is
    ///    already much smaller).
    ///
    /// Returns every valid (footprint ≤ `capacity`) visited tile, sorted by
    /// ascending score (best first).
    pub fn enumerate_subtiles(
        &self,
        nodes: &[NodeId],
        root: NodeId,
        capacity: u64,
    ) -> Vec<TileEdgeShape> {
        let full_shape = self.node_output_shape(root).clone();
        let Some(output_edge) = self
            .output_edge_id(root)
            .or_else(|| self.children(root).first().map(|&(_, id)| id))
        else {
            return Vec::new();
        };

        let (search_axes, mut base) = Self::search_axes_for(&full_shape, self.node(root));

        let mut visited: HashMap<TileEdgeShape, Option<f64>> = HashMap::new();
        let mut queue: Vec<(f64, TileEdgeShape)> = Vec::new();

        // Base tile: grow each axis independently while the score keeps
        // improving. Every candidate tried (not just the axis's final
        // choice) is recorded via `visit_candidate`, so it's also a
        // returnable result, not just a stepping stone.
        for axis in &search_axes {
            if axis.ladder.len() <= 1 {
                continue;
            }
            let mut best_idx = 0usize;
            let mut best_score = self.visit_candidate(
                nodes,
                output_edge,
                capacity,
                base.clone(),
                &mut visited,
                &mut queue,
            );
            for (idx, step) in axis.ladder.iter().enumerate().skip(1) {
                let mut candidate = base.clone();
                axis.write(&mut candidate, step);
                let Some(candidate_score) = self.visit_candidate(
                    nodes,
                    output_edge,
                    capacity,
                    candidate,
                    &mut visited,
                    &mut queue,
                ) else {
                    continue;
                };
                let improved = match best_score {
                    Some(best) => candidate_score < best,
                    None => true,
                };
                if !improved {
                    break;
                }
                best_score = Some(candidate_score);
                best_idx = idx;
            }
            axis.write(&mut base, &axis.ladder[best_idx]);
        }

        // Expanding neighbor search: repeatedly take the best-scoring
        // visited-but-not-yet-expanded tile and try bumping each axis to
        // its next ladder step.
        while !queue.is_empty() && visited.len() < MAX_ENUMERATED_TILES {
            let (min_idx, _) = queue
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
                .expect("queue is non-empty");
            let (_, tile) = queue.remove(min_idx);

            for axis in &search_axes {
                let Some(&innermost) = axis.dims.last() else {
                    continue;
                };
                let Some(current) = tile.get(innermost) else {
                    continue;
                };
                let Some(idx) = axis.ladder.iter().position(|dim| dim == current) else {
                    continue;
                };
                if idx + 1 >= axis.ladder.len() {
                    continue;
                }
                let mut neighbor = tile.clone();
                axis.write(&mut neighbor, &axis.ladder[idx + 1]);
                self.visit_candidate(
                    nodes,
                    output_edge,
                    capacity,
                    neighbor,
                    &mut visited,
                    &mut queue,
                );
            }
        }

        let mut results: Vec<(f64, TileEdgeShape)> = visited
            .into_iter()
            .filter_map(|(tile, score)| score.map(|score| (score, tile)))
            .collect();
        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        results.into_iter().map(|(_, tile)| tile).collect()
    }

    /// Builds [`Self::enumerate_subtiles`]'s search axes and initial base
    /// tile for `full_shape`, driven by `node`'s own declared output
    /// [`TensorTileSpec`] when it's usable (its `rank` matches
    /// `full_shape.len()`) — one [`SearchAxis`] per [`TileAxisBinding`],
    /// spanning a flattened group of real dims when `dims` names more
    /// than one, with every uncovered real dim left untiled (never
    /// varied, always its own full extent). Falls back to one ordinary
    /// per-real-dim axis each when `node` has no usable tile_spec.
    fn search_axes_for(
        full_shape: &TileEdgeShape,
        node: &TileOp,
    ) -> (Vec<SearchAxis>, TileEdgeShape) {
        let output_spec = node
            .tile_spec
            .as_ref()
            .and_then(|spec| spec.outputs.first())
            .filter(|output_spec| output_spec.rank == full_shape.len());

        let Some(output_spec) = output_spec else {
            let base: TileEdgeShape = full_shape
                .iter()
                .map(|dim| match dim {
                    TileDim::Fixed(_) => TileDim::Fixed(1),
                    TileDim::Sym(name) => TileDim::Sym(name.clone()),
                })
                .collect();
            let axes = (0..full_shape.len())
                .map(|d| SearchAxis {
                    dims: vec![d],
                    ladder: match &full_shape[d] {
                        TileDim::Fixed(extent) => power_of_two_ladder(*extent)
                            .into_iter()
                            .map(TileDim::Fixed)
                            .collect(),
                        TileDim::Sym(name) => vec![TileDim::Sym(name.clone())],
                    },
                })
                .collect();
            return (axes, base);
        };

        // Untiled dims (no axis binding at all) keep their full extent,
        // never varied -- start from a full-shape clone and only
        // overwrite the dims an axis binding actually spans.
        let mut base = full_shape.clone();
        let mut axes = Vec::new();
        for axis in output_spec.axes.iter() {
            let Some((&innermost, _)) = axis.dims.split_last() else {
                continue; // empty dims: nothing to bind (spec-authoring bug)
            };
            let all_fixed = axis
                .dims
                .iter()
                .all(|&d| matches!(full_shape.get(d), Some(TileDim::Fixed(_))));
            let ladder = if all_fixed {
                let combined_extent: usize = axis
                    .dims
                    .iter()
                    .filter_map(|&d| match full_shape.get(d) {
                        Some(TileDim::Fixed(extent)) => Some(*extent),
                        _ => None,
                    })
                    .product();
                power_of_two_ladder(combined_extent)
                    .into_iter()
                    .map(TileDim::Fixed)
                    .collect()
            } else {
                // A Sym-typed dim is in this group: not enumerable,
                // mirrors the ordinary per-dim Sym case (a single
                // unchanging entry).
                vec![
                    full_shape
                        .get(innermost)
                        .cloned()
                        .unwrap_or(TileDim::Fixed(1)),
                ]
            };
            for &d in axis.dims.iter() {
                if let Some(dim @ TileDim::Fixed(_)) = base.get_mut(d) {
                    *dim = TileDim::Fixed(1);
                }
            }
            axes.push(SearchAxis {
                dims: axis.dims.to_vec(),
                ladder,
            });
        }
        (axes, base)
    }

    /// Scores `tile` via [`Self::score_candidate_tile`] and records it in
    /// `visited`/`queue`, unless it's already been visited (returns the
    /// cached score in that case, without rescoring or re-queuing). Shared
    /// by [`Self::enumerate_subtiles`]'s base-tile growth and expanding
    /// search, so every candidate either phase tries ends up in the
    /// returned result set, not just each axis's final choice.
    fn visit_candidate(
        &self,
        nodes: &[NodeId],
        output_edge: EdgeId,
        capacity: u64,
        tile: TileEdgeShape,
        visited: &mut HashMap<TileEdgeShape, Option<f64>>,
        queue: &mut Vec<(f64, TileEdgeShape)>,
    ) -> Option<f64> {
        if let Some(&existing) = visited.get(&tile) {
            return existing;
        }
        let score = self.score_candidate_tile(nodes, output_edge, capacity, &tile);
        visited.insert(tile.clone(), score);
        if let Some(score) = score {
            queue.push((score, tile));
        }
        score
    }

    /// Welder Fig. 7's `SubGraphTiling(g, level, c)`: enumerates candidate
    /// output tiles for `root` via [`Self::enumerate_subtiles`] (bounded by
    /// `level`'s capacity in `hardware`), `propagate`s each into a
    /// [`TileConfig`], keeps the `top_k` lowest-`MemTraffic` ones, and
    /// recurses one memory level up on the (deduplicated) subgraphs
    /// [`Self::extract_subgraph`] finds there.
    ///
    /// Recursion terminates at the top of `hardware`'s declared memory
    /// hierarchy (Fig. 7's "return empty sub-graph at top level to exit
    /// recursion") — bounded by the number of distinct
    /// [`MemoryLevelKind`]s `hardware` declares, so this always halts.
    ///
    /// Deviates from the paper in one respect, noted in
    /// `TILE_GRAPH_SCHEDULING_PLAN.md`/teenygrad-1nr.4: this always
    /// re-derives each level's own candidates fresh from that level's own
    /// root, rather than threading the paper's `c` (the config chosen one
    /// level down) into the next level's candidate search — the paper's
    /// pseudocode doesn't specify enough about how `c` constrains
    /// `EnumerateSubtiles` to port literally, and even Welder's own
    /// shipped implementation (`policy/default.py`'s `emit_config`) calls
    /// its search once from a single base tile rather than implementing
    /// this literal recursive threading.
    pub fn sub_graph_tiling(
        &self,
        nodes: &[NodeId],
        root: NodeId,
        level: Option<MemoryLevelKind>,
        hardware: &HardwareProfile,
        top_k: usize,
    ) -> Vec<SubGraphTilingResult> {
        let capacity = level
            .and_then(|level| hardware.level(level))
            .map(|memory_level| memory_level.capacity)
            .unwrap_or(u64::MAX);

        let Some(output_edge) = self
            .output_edge_id(root)
            .or_else(|| self.children(root).first().map(|&(_, id)| id))
        else {
            return Vec::new();
        };

        let mut scored: Vec<(TileConfig, u64)> = self
            .enumerate_subtiles(nodes, root, capacity)
            .into_iter()
            .filter_map(|subtile| {
                let mut seed = HashMap::new();
                seed.insert(output_edge, subtile);
                let config = self.propagate(nodes, &seed);
                if self.mem_footprint_with_config(nodes, &config) > capacity {
                    return None;
                }
                let traffic = self.mem_traffic_with_config(nodes, &config);
                Some((config, traffic))
            })
            .collect();
        scored.sort_by_key(|&(_, traffic)| traffic);
        scored.truncate(top_k.max(1));

        let next_level = hardware.next_memory_level(level);

        scored
            .into_iter()
            .map(|(config, _)| {
                let children = match next_level {
                    None => Vec::new(),
                    Some(next_level) => {
                        let mut seen: Vec<Vec<NodeId>> = Vec::new();
                        let mut children = Vec::new();
                        for &node in nodes {
                            let subgraph = self.extract_subgraph(node, Some(next_level));
                            if seen.contains(&subgraph) {
                                continue;
                            }
                            seen.push(subgraph.clone());
                            children.extend(self.sub_graph_tiling(
                                &subgraph,
                                node,
                                Some(next_level),
                                hardware,
                                top_k,
                            ));
                        }
                        children
                    }
                };
                SubGraphTilingResult {
                    nodes: nodes.to_vec(),
                    config,
                    children,
                }
            })
            .collect()
    }

    /// Returns node indices in topological order (producers before
    /// consumers) using Kahn's algorithm. Panics if the graph contains a
    /// cycle.
    pub fn topological_sort(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for record in &self.edges {
            if let (Some(_), Some(consumer)) = (record.producer, record.consumer) {
                in_degree[consumer] += 1;
            }
        }
        let mut queue: Vec<NodeId> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(index) = queue.pop() {
            order.push(index);
            for &id in &self.outgoing[index] {
                if let Some(consumer) = self.edges[id.0].consumer {
                    in_degree[consumer] -= 1;
                    if in_degree[consumer] == 0 {
                        queue.push(consumer);
                    }
                }
            }
        }

        assert_eq!(order.len(), n, "tile graph contains a cycle");
        order
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use teeny_core::device::hardware::{HardwareProfile, MemoryLevel, MemoryLevelKind};
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::{ExecutableOp, KernelTileSpec};
    use teeny_core::utils::dag::Dag;

    use super::{
        NodeId, SubGraphTilingResult, TileConfig, TileDim, TileEdge, TileEdgeShape, TileGraph,
    };

    /// Minimal [`ExecutableOp`] test double: just enough surface
    /// (name/shape/dtype/tile_spec) for [`TileGraph::from_dag`] to convert on.
    struct TestOp {
        name: &'static str,
        dtype: DtypeRepr,
        shape: Shape,
        is_input: bool,
        tile_spec: Option<KernelTileSpec>,
    }

    impl ExecutableOp for TestOp {
        fn name(&self) -> &str {
            self.name
        }

        fn is_input(&self) -> bool {
            self.is_input
        }

        fn forward_kernel_source(&self) -> &str {
            ""
        }

        fn forward_kernel_entry_point(&self) -> &str {
            ""
        }

        fn output_shape(&self) -> &Shape {
            &self.shape
        }

        fn output_dtype(&self) -> DtypeRepr {
            self.dtype
        }

        fn tile_spec(&self) -> Option<KernelTileSpec> {
            self.tile_spec
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn op(name: &'static str, shape: Shape, is_input: bool) -> Box<dyn ExecutableOp> {
        Box::new(TestOp {
            name,
            dtype: DtypeRepr::F32,
            shape,
            is_input,
            tile_spec: None,
        })
    }

    fn op_with_tile_spec(
        name: &'static str,
        shape: Shape,
        is_input: bool,
        tile_spec: KernelTileSpec,
    ) -> Box<dyn ExecutableOp> {
        Box::new(TestOp {
            name,
            dtype: DtypeRepr::F32,
            shape,
            is_input,
            tile_spec: Some(tile_spec),
        })
    }

    #[test]
    fn empty_dag_produces_empty_tile_graph() {
        let dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let tile_graph = TileGraph::from_dag(&dag);
        assert!(tile_graph.is_empty());
        assert_eq!(tile_graph.len(), 0);
    }

    #[test]
    fn linear_chain_preserves_node_count_and_edges() {
        let shape = vec![Some(4), Some(8)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let input = dag.add_node(op("input", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape.clone(), false));
        dag.add_edge(input, relu);
        let sigmoid = dag.add_node(op("sigmoid", shape.clone(), false));
        dag.add_edge(relu, sigmoid);

        let tile_graph = TileGraph::from_dag(&dag);
        assert_eq!(tile_graph.len(), 3);

        assert_eq!(tile_graph.node(input).name, "input");
        assert!(tile_graph.parents(input).is_empty());

        assert_eq!(tile_graph.node(relu).name, "relu");
        assert_eq!(tile_graph.parents(relu), vec![input]);

        assert_eq!(tile_graph.node(sigmoid).name, "sigmoid");
        assert_eq!(tile_graph.parents(sigmoid), vec![relu]);

        let fixed_shape = vec![TileDim::Fixed(4), TileDim::Fixed(8)];

        // Fanout mirrors the operand edges above, one hop forward, carrying
        // the producer's shape.
        let input_children = tile_graph.children(input);
        assert_eq!(input_children.len(), 1);
        assert_eq!(input_children[0].0, relu);
        assert_eq!(tile_graph.edge(input_children[0].1).shape, fixed_shape);

        let relu_children = tile_graph.children(relu);
        assert_eq!(relu_children.len(), 1);
        assert_eq!(relu_children[0].0, sigmoid);
        assert_eq!(tile_graph.edge(relu_children[0].1).shape, fixed_shape);

        assert!(tile_graph.children(sigmoid).is_empty());
    }

    #[test]
    fn fan_in_preserves_distinct_producers_in_insertion_order() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), true));
        let add = dag.add_node(op("add", shape.clone(), false));
        dag.add_edge(a, add);
        dag.add_edge(b, add);

        let tile_graph = TileGraph::from_dag(&dag);

        assert_eq!(tile_graph.parents(add), vec![a, b]);

        let a_children = tile_graph.children(a);
        assert_eq!(a_children.len(), 1);
        assert_eq!(a_children[0].0, add);
        assert_eq!(
            tile_graph.edge(a_children[0].1),
            &TileEdge {
                shape: vec![TileDim::Fixed(4)],
                memory_level: MemoryLevelKind::DeviceMemory,
            }
        );

        let b_children = tile_graph.children(b);
        assert_eq!(b_children.len(), 1);
        assert_eq!(b_children[0].0, add);
        assert_eq!(
            tile_graph.edge(b_children[0].1),
            &TileEdge {
                shape: vec![TileDim::Fixed(4)],
                memory_level: MemoryLevelKind::DeviceMemory,
            }
        );
    }

    #[test]
    fn self_referential_operand_collapses_to_a_single_parent_entry() {
        // Add(x, x): both operand slots read the same producer. `Dag::add_edge`
        // already dedups a repeated (producer, consumer) pair before `from_dag`
        // ever sees it, so `parents`/`children` here can only reflect that `x`
        // is used *at all* by `add`, not how many operand slots referenced it
        // — see the module doc comment.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("x", shape.clone(), true));
        let add = dag.add_node(op("add", shape.clone(), false));
        dag.add_edge(x, add);
        dag.add_edge(x, add);

        let tile_graph = TileGraph::from_dag(&dag);

        assert_eq!(tile_graph.parents(add), vec![x]);
        let x_children = tile_graph.children(x);
        assert_eq!(x_children.len(), 1);
        assert_eq!(x_children[0].0, add);
        assert_eq!(
            tile_graph.edge(x_children[0].1),
            &TileEdge {
                shape: vec![TileDim::Fixed(4)],
                memory_level: MemoryLevelKind::DeviceMemory,
            }
        );
    }

    #[test]
    fn fan_out_preserves_multiple_distinct_consumers() {
        // x feeds two different downstream ops.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("x", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape.clone(), false));
        dag.add_edge(x, relu);
        let sigmoid = dag.add_node(op("sigmoid", shape.clone(), false));
        dag.add_edge(x, sigmoid);

        let tile_graph = TileGraph::from_dag(&dag);

        let x_children = tile_graph.children(x);
        assert_eq!(x_children.len(), 2);

        let mut consumers: Vec<usize> = x_children.iter().map(|&(c, _)| c).collect();
        consumers.sort_unstable();
        assert_eq!(consumers, {
            let mut expected = vec![relu, sigmoid];
            expected.sort_unstable();
            expected
        });
    }

    #[test]
    fn from_dag_leaves_every_edge_in_device_memory() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), true));
        let add = dag.add_node(op("add", shape.clone(), false));
        dag.add_edge(a, add);
        dag.add_edge(b, add);

        let tile_graph = TileGraph::from_dag(&dag);

        for &node in &[a, b] {
            for (_, id) in tile_graph.children(node) {
                assert_eq!(
                    tile_graph.edge(id).memory_level,
                    MemoryLevelKind::DeviceMemory
                );
            }
        }
    }

    #[test]
    fn nodes_with_no_producer_get_an_input_edge() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let input = dag.add_node(op("input", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape, false));
        dag.add_edge(input, relu);

        let tile_graph = TileGraph::from_dag(&dag);

        let edge = tile_graph
            .input_edge(input)
            .expect("input node has no producer in dag");
        assert_eq!(edge.shape, vec![TileDim::Fixed(4)]);
        assert_eq!(edge.memory_level, MemoryLevelKind::DeviceMemory);

        // relu has a real producer, so it's not a DAG-input boundary node.
        assert!(tile_graph.input_edge(relu).is_none());
    }

    #[test]
    fn input_edge_condition_is_structural_not_the_is_input_flag() {
        // The condition is structural (empty `parents`), not
        // `ExecutableOp::is_input()` specifically — e.g. a lowered constant
        // op (`is_input: false`) with no producer is still a boundary node.
        let shape = vec![Some(2), Some(2)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let constant = dag.add_node(op("constant", shape, false));

        let tile_graph = TileGraph::from_dag(&dag);
        let edge = tile_graph
            .input_edge(constant)
            .expect("zero-parent constant node is a DAG-input boundary node");
        assert_eq!(edge.shape, vec![TileDim::Fixed(2), TileDim::Fixed(2)]);
    }

    #[test]
    fn nodes_with_no_consumer_get_an_output_edge() {
        let shape = vec![Some(4), Some(8)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let input = dag.add_node(op("input", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape.clone(), false));
        dag.add_edge(input, relu);
        let sigmoid = dag.add_node(op("sigmoid", shape, false));
        dag.add_edge(relu, sigmoid);

        let tile_graph = TileGraph::from_dag(&dag);

        // Only the DAG's true sink (sigmoid) is a DAG-output boundary node.
        assert!(tile_graph.output_edge(input).is_none());
        assert!(tile_graph.output_edge(relu).is_none());

        let edge = tile_graph
            .output_edge(sigmoid)
            .expect("sigmoid has no consumer in dag");
        assert_eq!(edge.shape, vec![TileDim::Fixed(4), TileDim::Fixed(8)]);
        assert_eq!(edge.memory_level, MemoryLevelKind::DeviceMemory);
    }

    #[test]
    fn fan_out_node_with_all_consumers_present_has_no_output_edge() {
        // x has two consumers in dag, so it is not a DAG output even though
        // it also happens to be a DAG input.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("x", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape.clone(), false));
        dag.add_edge(x, relu);
        let sigmoid = dag.add_node(op("sigmoid", shape, false));
        dag.add_edge(x, sigmoid);

        let tile_graph = TileGraph::from_dag(&dag);
        assert!(tile_graph.input_edge(x).is_some());
        assert!(tile_graph.output_edge(x).is_none());
    }

    #[test]
    fn dynamic_axis_becomes_a_symbolic_dim() {
        // A `None` (dynamic/unknown) axis in the source shape becomes a
        // synthesized `TileDim::Sym`, not a `Fixed` extent — from_dag
        // doesn't know the runtime value, and unifying symbols that are
        // actually the same free variable across edges is the later
        // propagation pass's job.
        let shape = vec![None, Some(8)]; // e.g. a dynamic batch axis
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let input = dag.add_node(op("input", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape, false));
        dag.add_edge(input, relu);

        let tile_graph = TileGraph::from_dag(&dag);

        let input_children = tile_graph.children(input);
        let edge = tile_graph.edge(input_children[0].1);
        assert_eq!(edge.shape[1], TileDim::Fixed(8));
        assert!(matches!(edge.shape[0], TileDim::Sym(_)));
    }

    #[test]
    fn distinct_dynamic_axes_get_distinct_symbols() {
        // from_dag must not accidentally collide two different (node, axis)
        // dynamic dims onto the same synthesized name — that would silently
        // assert a shared-free-variable relationship that doesn't exist yet
        // (unification is the propagation pass's job).
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![None], true));
        let b = dag.add_node(op("b", vec![None], true));

        let tile_graph = TileGraph::from_dag(&dag);

        let a_sym = tile_graph.input_edge(a).unwrap().shape[0].clone();
        let b_sym = tile_graph.input_edge(b).unwrap().shape[0].clone();
        assert_ne!(a_sym, b_sym);
    }

    #[test]
    fn topological_sort_orders_producers_before_consumers() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), true));
        let add = dag.add_node(op("add", shape.clone(), false));
        dag.add_edge(a, add);
        dag.add_edge(b, add);
        let relu = dag.add_node(op("relu", shape, false));
        dag.add_edge(add, relu);

        let tile_graph = TileGraph::from_dag(&dag);
        let order = tile_graph.topological_sort();

        assert_eq!(order.len(), 4);
        let position = |node: usize| order.iter().position(|&i| i == node).unwrap();
        assert!(position(a) < position(add));
        assert!(position(b) < position(add));
        assert!(position(add) < position(relu));
    }

    #[test]
    fn topological_sort_handles_self_referential_operands() {
        // Regression test: in-degree must be derived from the deduped
        // internal edges, not double-counted — otherwise `add`'s in-degree
        // never reaches zero and this would panic with "tile graph contains
        // a cycle".
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("x", shape.clone(), true));
        let add = dag.add_node(op("add", shape.clone(), false));
        dag.add_edge(x, add);
        dag.add_edge(x, add);
        let relu = dag.add_node(op("relu", shape, false));
        dag.add_edge(add, relu);

        let tile_graph = TileGraph::from_dag(&dag);
        let order = tile_graph.topological_sort();

        assert_eq!(order.len(), 3);
        let position = |node: usize| order.iter().position(|&i| i == node).unwrap();
        assert!(position(x) < position(add));
        assert!(position(add) < position(relu));
    }

    #[test]
    fn set_connect_updates_the_level_visible_from_both_directions() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("x", shape.clone(), true));
        let relu = dag.add_node(op("relu", shape, false));
        dag.add_edge(x, relu);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let id = tile_graph.children(x)[0].1;

        assert_eq!(tile_graph.connect_level(id), MemoryLevelKind::DeviceMemory);

        tile_graph.set_connect(id, MemoryLevelKind::SharedMemory);

        assert_eq!(tile_graph.connect_level(id), MemoryLevelKind::SharedMemory);
        assert_eq!(
            tile_graph.edge(id).memory_level,
            MemoryLevelKind::SharedMemory
        );
        // Same id, addressed from the consumer's incoming list this time —
        // there is only one copy of this edge's data to be stale.
        assert_eq!(tile_graph.children(x)[0].1, id);
    }

    #[test]
    fn set_connect_mutates_a_boundary_edge_via_its_id() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let input = dag.add_node(op("input", shape, true));

        let mut tile_graph = TileGraph::from_dag(&dag);
        let id = tile_graph
            .input_edge_id(input)
            .expect("input node has a boundary input edge");

        tile_graph.set_connect(id, MemoryLevelKind::Register);

        assert_eq!(
            tile_graph.input_edge(input).unwrap().memory_level,
            MemoryLevelKind::Register
        );
    }

    #[test]
    fn extract_subgraph_of_a_node_with_no_qualifying_edges_is_a_singleton() {
        // Every edge starts at the from_dag default (DeviceMemory), so
        // extracting at that same level finds nothing strictly above it to
        // pull in.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let sub = tile_graph.extract_subgraph(a, Some(MemoryLevelKind::DeviceMemory));

        assert_eq!(sub, vec![a]);
    }

    #[test]
    fn extract_subgraph_follows_qualifying_edges_in_both_directions() {
        // a -> b -> c, both edges connected above the DeviceMemory
        // threshold: flooding from the middle node must reach both ends.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let mut tile_graph = TileGraph::from_dag(&dag);
        tile_graph.set_connect(tile_graph.children(a)[0].1, MemoryLevelKind::HostMemory);
        tile_graph.set_connect(tile_graph.children(b)[0].1, MemoryLevelKind::HostMemory);

        let mut sub = tile_graph.extract_subgraph(b, Some(MemoryLevelKind::DeviceMemory));
        sub.sort_unstable();

        assert_eq!(sub, vec![a, b, c]);
    }

    #[test]
    fn extract_subgraph_stops_at_edges_at_or_below_the_level() {
        // a -> b connected above the threshold (fused in); b -> c left at
        // the threshold (not fused) -- c must not be pulled in.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let mut tile_graph = TileGraph::from_dag(&dag);
        tile_graph.set_connect(tile_graph.children(a)[0].1, MemoryLevelKind::HostMemory);
        // b -> c stays at the from_dag default: DeviceMemory.

        let mut sub = tile_graph.extract_subgraph(a, Some(MemoryLevelKind::DeviceMemory));
        sub.sort_unstable();

        assert_eq!(sub, vec![a, b]);
    }

    #[test]
    fn extract_subgraph_does_not_cross_a_disqualified_edge_in_reverse_either() {
        // a -> b -> c: only b -> c qualifies, so extracting from c must not
        // reach back through the disqualified a -> b edge.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let mut tile_graph = TileGraph::from_dag(&dag);
        tile_graph.set_connect(tile_graph.children(b)[0].1, MemoryLevelKind::HostMemory);
        // a -> b stays at the from_dag default: DeviceMemory.

        let mut sub = tile_graph.extract_subgraph(c, Some(MemoryLevelKind::DeviceMemory));
        sub.sort_unstable();

        assert_eq!(sub, vec![b, c]);
    }

    #[test]
    fn mem_traffic_of_a_single_boundary_node_sums_its_input_and_output_edges() {
        // A lone F32 [4] node: an input boundary edge in, an output
        // boundary edge out. 4 elements * 4 bytes = 16 bytes each.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape, true));

        let tile_graph = TileGraph::from_dag(&dag);

        assert_eq!(tile_graph.mem_traffic(&[a]), 32);
    }

    #[test]
    fn mem_traffic_ignores_edges_internal_to_the_node_set() {
        // a -> b, both included: the internal edge doesn't count, only
        // a's input boundary and b's output boundary do.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);

        assert_eq!(tile_graph.mem_traffic(&[a, b]), 32);
    }

    #[test]
    fn mem_traffic_counts_an_excluded_neighbors_edge_as_boundary_traffic() {
        // a -> b -> c, extracting {a, b}: b's edge to the excluded c
        // still counts, from b's outgoing side.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);

        // a's input edge (16B) + b -> c edge (16B); a -> b is internal.
        assert_eq!(tile_graph.mem_traffic(&[a, b]), 32);
    }

    #[test]
    fn mem_traffic_counts_an_excluded_producers_edge_from_the_consumer_side() {
        // a -> b -> c, extracting {b, c}: b's edge from the excluded a
        // still counts, read from b's parent_edges.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);

        // a -> b edge (16B) + c's output edge (16B); b -> c is internal.
        assert_eq!(tile_graph.mem_traffic(&[b, c]), 32);
    }

    #[test]
    fn mem_footprint_of_a_linear_chain_never_exceeds_two_live_tiles() {
        // a -> b -> c, F32 [4] (16 bytes each): at most a producer and its
        // one consumer are ever resident at the same time.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);

        assert_eq!(tile_graph.mem_footprint(&[a, b, c]), 32);
    }

    #[test]
    fn mem_footprint_of_a_diamond_peaks_above_its_own_boundary_traffic() {
        // a fans out to b and c, both of which feed d. a must stay live
        // until *both* b and c have consumed it, so all three of a, b, c
        // are resident at once just before d runs -- a genuine peak above
        // what mem_traffic (just the a-in, d-out boundary) would suggest.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let c = dag.add_node(op("c", shape.clone(), false));
        dag.add_edge(a, c);
        let d = dag.add_node(op("d", shape, false));
        dag.add_edge(b, d);
        dag.add_edge(c, d);

        let tile_graph = TileGraph::from_dag(&dag);
        let nodes = [a, b, c, d];

        assert_eq!(tile_graph.mem_footprint(&nodes), 48);
        // The peak (a + b + c all live at once) is real extra cost that
        // pure boundary traffic (a's input + d's output) doesn't see.
        assert_eq!(tile_graph.mem_traffic(&nodes), 32);
    }

    #[test]
    fn mem_footprint_keeps_a_boundary_crossing_producer_live_past_its_last_in_set_consumer() {
        // a feeds both b (in the extracted set) and e (outside it). Even
        // after b -- a's only *in-set* consumer -- has run, a must stay
        // resident: e might still need to read it. That keeps a alive
        // through c's allocation too, raising the peak above what it would
        // be if a were freed as soon as b ran.
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), false));
        dag.add_edge(a, b);
        let _e = dag.add_node(op("e", shape.clone(), false));
        dag.add_edge(a, _e);
        let c = dag.add_node(op("c", shape, false));
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);

        // a (16) + b (16) + c (16) all resident when c is allocated,
        // since a can't be freed until the excluded e is done with it.
        assert_eq!(tile_graph.mem_footprint(&[a, b, c]), 48);
    }

    use teeny_core::model::{TensorTileSpec, TileAxisBinding};

    /// A flat, single-axis elementwise spec: input and output share one
    /// `extent_param` name (`"n_elements"`), so resolving the output
    /// resolves the input with no arithmetic at all.
    fn flat_unary_spec() -> KernelTileSpec {
        const AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_SIZE",
            extent_param: "n_elements",
            window: None,
            divide_by: None,
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 1,
            axes: &[AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    /// GEMM-shaped: `a_ptr: [M, K]`, `b_ptr: [K, N]`, `c_ptr: [M, N]` —
    /// mirrors the real `MATMUL_TILE_SPEC` in `graph/mod.rs`.
    fn gemm_shaped_spec() -> KernelTileSpec {
        const A: TensorTileSpec = TensorTileSpec {
            param: "a_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dims: &[0],
                    block_const: "BLOCK_M",
                    extent_param: "M",
                    window: None,
                    divide_by: None,
                },
                TileAxisBinding {
                    dims: &[1],
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                    divide_by: None,
                },
            ],
            reduction_axis: Some(1),
            untiled_dims: &[],
        };
        const B: TensorTileSpec = TensorTileSpec {
            param: "b_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dims: &[0],
                    block_const: "BLOCK_K",
                    extent_param: "K",
                    window: None,
                    divide_by: None,
                },
                TileAxisBinding {
                    dims: &[1],
                    block_const: "BLOCK_N",
                    extent_param: "N",
                    window: None,
                    divide_by: None,
                },
            ],
            reduction_axis: Some(0),
            untiled_dims: &[],
        };
        const C: TensorTileSpec = TensorTileSpec {
            param: "c_ptr",
            rank: 2,
            axes: &[
                TileAxisBinding {
                    dims: &[0],
                    block_const: "BLOCK_M",
                    extent_param: "M",
                    window: None,
                    divide_by: None,
                },
                TileAxisBinding {
                    dims: &[1],
                    block_const: "BLOCK_N",
                    extent_param: "N",
                    window: None,
                    divide_by: None,
                },
            ],
            reduction_axis: None,
            untiled_dims: &[],
        };
        KernelTileSpec {
            inputs: &[A, B],
            outputs: &[C],
        }
    }

    #[test]
    fn propagate_resolves_flat_elementwise_identity() {
        // input(a) -> relu(b), b declares flat_unary_spec. Seeding b's
        // output tile at 500 (smaller than its full 1000) must resolve the
        // a -> b edge to that same 500, via the shared "n_elements" name.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(500)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        assert_eq!(config.get(ab_edge), Some(&vec![TileDim::Fixed(500)]));
    }

    #[test]
    fn propagate_leaves_the_reduction_axis_at_its_full_extent() {
        // a: [M=256, K=96] -> c; b: [K=96, N=128] -> c; c = matmul(a, b).
        // Seeding c's output tile at [M=64, N=32] must resolve M on a and N
        // on b, while K (no output-side counterpart) falls back to its own
        // full extent (96) on both — not derived from M, N, or each other.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(256), Some(96)], true));
        let b = dag.add_node(op("b", vec![Some(96), Some(128)], true));
        let c = dag.add_node(op_with_tile_spec(
            "c",
            vec![Some(256), Some(128)],
            false,
            gemm_shaped_spec(),
        ));
        dag.add_edge(a, c);
        dag.add_edge(b, c);

        let tile_graph = TileGraph::from_dag(&dag);
        let ac_edge = tile_graph.children(a)[0].1;
        let bc_edge = tile_graph.children(b)[0].1;
        let c_output_edge = tile_graph
            .output_edge_id(c)
            .expect("c has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(c_output_edge, vec![TileDim::Fixed(64), TileDim::Fixed(32)]);
        let config = tile_graph.propagate(&[a, b, c], &seed);

        assert_eq!(
            config.get(ac_edge),
            Some(&vec![TileDim::Fixed(64), TileDim::Fixed(96)])
        );
        assert_eq!(
            config.get(bc_edge),
            Some(&vec![TileDim::Fixed(96), TileDim::Fixed(32)])
        );
    }

    #[test]
    fn propagate_stops_at_a_node_with_no_declared_tile_spec() {
        // a -> b, but b has no tile_spec: propagate must not guess a's
        // required tile shape from b's seeded output tile.
        let shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape, false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph.output_edge_id(b).unwrap();

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(500)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        assert_eq!(config.get(b_output_edge), Some(&vec![TileDim::Fixed(500)]));
        assert_eq!(config.get(ab_edge), None);
    }

    /// A conv2d-style spec: only the width axis (dim 0 here) is genuinely
    /// block-tiled (`BLOCK_OW`/`"OW"`, mirroring the real
    /// `conv2d_forward` kernel's only per-axis block const); the other
    /// real dimension (channels) is grid-driven with no block-size
    /// generic of its own, so it's named in `untiled_dims` instead of
    /// getting a `TileAxisBinding` -- exactly the case `untiled_dims`'s
    /// own doc comment in `teeny_core::model::tile_spec` describes as its
    /// reason to exist. `rank` (2) still reflects the tensor's real,
    /// full rank; only `axes` (1 entry) is partial.
    fn partially_tiled_spec() -> KernelTileSpec {
        const WIDTH: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_OW",
            extent_param: "OW",
            window: None,
            divide_by: None,
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 2,
            axes: &[WIDTH],
            reduction_axis: None,
            untiled_dims: &["C"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_of_a_partially_tiled_spec_should_not_drop_the_untiled_dims() {
        // a -> b, b declares partially_tiled_spec: axes covers only its
        // width dim (dim 0); channels (dim 1) is named in untiled_dims
        // instead, per that field's documented purpose. propagate must
        // resolve a full rank-2 tile for the a->b edge -- channels at its
        // real full extent (8), width at whatever b's seeded output tile
        // requests (16) -- not silently drop the untiled channels
        // dimension. Regression test for teenygrad-1nr.7: propagate used
        // to build `resolved`/`input_tile` by positionally zipping only
        // `axes` (len 1) against the real rank-2 output tile/edge shape,
        // producing a length-1 TileEdgeShape for a rank-2 edge.
        let full_shape = vec![Some(64), Some(8)]; // [OW, C]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            partially_tiled_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(16), TileDim::Fixed(8)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved.len(),
            full_shape.len(),
            "expected a rank-{}-consistent tile for the a->b edge (width \
             16, channels at its full extent 8), got a rank-{} shape \
             instead: {resolved:?} -- propagate silently dropped \
             untiled_dims' channels dimension",
            full_shape.len(),
            resolved.len(),
        );
        assert_eq!(
            resolved,
            &vec![TileDim::Fixed(16), TileDim::Fixed(8)],
            "width should resolve to b's seeded 16, channels should fall \
             back to its full extent 8 -- got {resolved:?}"
        );
    }

    /// A batchnorm2d-style spec: mirrors the real
    /// `batch_norm_2d_nchw_forward_inference` kernel (grid `[C, B]`, one
    /// `BLOCK_HW`-wide loop over the *flattened* `H*W` range per CTA) --
    /// the case `TileAxisBinding::dims` having more than one entry exists
    /// for (teenygrad-1nr.8). NCHW: dim 0 = B, dim 1 = C (both untiled,
    /// grid-driven), dims 2/3 = H/W flattened into one binding (`dims:
    /// &[2, 3]`, W innermost, matching NCHW's row-major layout). Input
    /// and output share `"HW"`, so this is shape-preserving elementwise
    /// like `flat_unary_spec`, just spanning a flattened pair of real
    /// dims instead of one.
    fn batchnorm2d_shaped_spec() -> KernelTileSpec {
        const HW: TileAxisBinding = TileAxisBinding {
            dims: &[2, 3],
            block_const: "BLOCK_HW",
            extent_param: "HW",
            window: None,
            divide_by: None,
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 4,
            axes: &[HW],
            reduction_axis: None,
            untiled_dims: &["B", "C"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_of_a_flattened_multi_dim_axis_collapses_outer_dims_to_one() {
        // a -> b, b declares batchnorm2d_shaped_spec. Seeding b's output
        // tile with W (the innermost of the flattened [H, W] pair) at 24
        // must resolve the a -> b edge to B/C at their full extent (2, 4
        // -- untiled), H collapsed to 1 (the flattened-away outer dim),
        // W at the resolved 24 -- not literal H*W=24 spread back across
        // both axes (there's no way to invert a flat block size into
        // separate per-axis extents in general), but product-preserving:
        // 2*4*1*24 matches what a real BLOCK_HW=24 tile's element count
        // would be, times B/C's own untiled extents.
        let full_shape = vec![Some(2), Some(4), Some(16), Some(32)]; // [B, C, H, W]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            batchnorm2d_shaped_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(
            b_output_edge,
            vec![
                TileDim::Fixed(2),
                TileDim::Fixed(4),
                TileDim::Fixed(16),
                TileDim::Fixed(24),
            ],
        );
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved,
            &vec![
                TileDim::Fixed(2),  // B: untiled, full extent
                TileDim::Fixed(4),  // C: untiled, full extent
                TileDim::Fixed(1),  // H: flattened-away outer dim
                TileDim::Fixed(24), // W: the resolved HW value
            ],
            "expected B/C at full extent, H collapsed to 1, W at the \
             resolved HW value -- got {resolved:?}"
        );
    }

    /// A groupnorm-style spec: mirrors the real `group_norm_forward`
    /// kernel's tiling shape (grid `[N*G]`, one CTA per (sample, group),
    /// iterating `BLOCK_NL`-wide tiles over `channels_per_group * L`
    /// where `channels_per_group = C / G`) -- teenygrad-1nr.10. Only `L`
    /// (dim 2) gets a `TileAxisBinding`; the channel axis (dim 1) is
    /// deliberately left out of `axes` entirely -- the best available
    /// authoring choice, since `TensorTileSpec` has no way to say "this
    /// axis's real per-tile extent is `C` divided by the compile-time
    /// constant `G`."
    fn groupnorm_shaped_spec() -> KernelTileSpec {
        const L_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[2],
            block_const: "BLOCK_NL",
            extent_param: "group_size",
            window: None,
            divide_by: None,
        };
        // channels_per_group = C / G (teenygrad-1nr.10): its own axis,
        // never resolved via name-matching (nothing else names
        // "channels_per_group"), so its value always comes from the
        // divide_by fallback -- both here (from the seeded output tile)
        // and on the input side.
        const C_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[1],
            block_const: "",
            extent_param: "channels_per_group",
            window: None,
            divide_by: Some(2), // G
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 3,
            axes: &[L_AXIS, C_AXIS],
            reduction_axis: None,
            untiled_dims: &["N"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            ..X
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_resolves_a_dim_subdivided_by_a_grid_constant() {
        // Fixed for teenygrad-1nr.10: group_norm_forward's real per-CTA
        // tile spans channels_per_group (= C/G) * L -- only a fraction
        // of the channel axis, not the whole of it. `TileAxisBinding::divide_by`
        // now lets a spec say so directly: with N=2, C=8, L=16, G=2
        // (channels_per_group=4), the channel axis must resolve to 4,
        // not C's full extent 8.
        let full_shape = vec![Some(2), Some(8), Some(16)]; // [N, C, L]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            groupnorm_shaped_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(
            b_output_edge,
            vec![TileDim::Fixed(2), TileDim::Fixed(8), TileDim::Fixed(16)],
        );
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        const CHANNELS_PER_GROUP: usize = 8 / 2; // C / G
        assert_eq!(
            resolved[1],
            TileDim::Fixed(CHANNELS_PER_GROUP),
            "expected the channel axis to resolve to channels_per_group \
             ({CHANNELS_PER_GROUP}), matching group_norm_forward's real \
             per-CTA tile -- got {:?}",
            resolved[1]
        );
    }

    /// Mirrors `groupnorm_shaped_spec`, except the output side never
    /// declares the channels axis at all (only `L` is tiled on `y_ptr`).
    /// Used to exercise `propagate`'s *input-side* `divide_by` fallback
    /// directly (teenygrad-1nr.10): `groupnorm_shaped_spec`'s `Y` happens
    /// to also declare `C_AXIS` (via `..X`), so its own test resolves
    /// `"channels_per_group"` on the output side first, never actually
    /// reaching the input-side fallback branch.
    fn groupnorm_input_only_divide_by_spec() -> KernelTileSpec {
        const L_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[2],
            block_const: "BLOCK_NL",
            extent_param: "group_size",
            window: None,
            divide_by: None,
        };
        const C_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[1],
            block_const: "",
            extent_param: "channels_per_group",
            window: None,
            divide_by: Some(2), // G
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 3,
            axes: &[L_AXIS, C_AXIS],
            reduction_axis: None,
            untiled_dims: &["N"],
        };
        const Y: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 3,
            axes: &[L_AXIS], // channels not declared here at all
            reduction_axis: None,
            untiled_dims: &["N", "C"],
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y],
        }
    }

    #[test]
    fn propagate_applies_divide_by_on_the_input_side_fallback_when_never_resolved_via_output() {
        // Companion to propagate_resolves_a_dim_subdivided_by_a_grid_constant:
        // that test's spec happens to also declare the divide_by axis on
        // the output side, so "channels_per_group" gets resolved (and
        // divided) there first. This spec's output never declares it at
        // all, so the input-side fallback branch is what has to apply
        // divide_by directly to the raw full extent.
        let full_shape = vec![Some(2), Some(8), Some(16)]; // [N, C, L]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            groupnorm_input_only_divide_by_spec(),
        ));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(
            b_output_edge,
            vec![TileDim::Fixed(2), TileDim::Fixed(8), TileDim::Fixed(16)],
        );
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved[1],
            TileDim::Fixed(4), // C / G = 8 / 2
            "expected the channel axis to fall back to channels_per_group \
             via divide_by, since nothing resolves it via name-matching \
             -- got {:?}",
            resolved[1]
        );
    }

    /// Simulates a two-output kernel (mirrors `flash_attn2`'s real
    /// `o_ptr` + `l_ptr` outputs, and `group_norm_forward`'s three --
    /// teenygrad-1nr.11): `outputs[0]` (`Y1`) names one `extent_param`
    /// (`"M"`); `outputs[1]` (`Y2`) names a *different* one (`"aux"`)
    /// that only appears on one input axis.
    fn two_output_spec() -> KernelTileSpec {
        const X_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_AUX",
            extent_param: "aux", // only Y2 names this -- Y1 never does
            window: None,
            divide_by: None,
        };
        const X: TensorTileSpec = TensorTileSpec {
            param: "x_ptr",
            rank: 1,
            axes: &[X_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const Y1_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_M",
            extent_param: "M",
            window: None,
            divide_by: None,
        };
        const Y1: TensorTileSpec = TensorTileSpec {
            param: "y_ptr",
            rank: 1,
            axes: &[Y1_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        const Y2_AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_AUX",
            extent_param: "aux",
            window: None,
            divide_by: None,
        };
        const Y2: TensorTileSpec = TensorTileSpec {
            param: "l_ptr",
            rank: 1,
            axes: &[Y2_AXIS],
            reduction_axis: None,
            untiled_dims: &[],
        };
        KernelTileSpec {
            inputs: &[X],
            outputs: &[Y1, Y2],
        }
    }

    #[test]
    fn propagate_resolves_a_second_declared_output_when_its_edge_is_seeded() {
        // Fixed for teenygrad-1nr.11: KernelTileSpec.outputs can declare
        // more than one real output tensor, motivated by flash_attn2's
        // real o_ptr + l_ptr (logsumexp) outputs and
        // group_norm_forward's y_ptr/mean_ptr/rstd_ptr. propagate used to
        // only ever read spec.outputs.first() -- an extent_param that
        // appears only on a second/later output was never seeded into
        // `resolved`, so an input axis bound to it always fell back to
        // its full extent, no matter what was seeded.
        //
        // TileGraph::from_dag now synthesizes one extra boundary edge per
        // additional declared output (TileGraph::secondary_output_edges)
        // -- seeding *that* edge (not b's primary output edge) lets
        // propagate resolve "aux" for real.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, two_output_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");
        let b_aux_edge = *tile_graph
            .secondary_output_edges(b)
            .first()
            .expect("b's two-output spec should have synthesized one secondary output edge");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(50)]); // Y1's "M"
        seed.insert(b_aux_edge, vec![TileDim::Fixed(20)]); // Y2's "aux"
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved,
            &vec![TileDim::Fixed(20)],
            "expected a's tile to resolve via Y2's seeded \"aux\" value \
             (20), not Y1's \"M\" (50, which a's axis isn't bound to) or \
             the full-extent fallback (1000) -- got {resolved:?}"
        );
    }

    #[test]
    fn propagate_leaves_an_unseeded_second_output_without_a_full_shape_fallback() {
        // Companion to the test above: unlike the primary output (which
        // falls back to the node's own full shape when nothing seeds
        // it), a second declared output has no ground-truth shape to
        // fall back to at all (ExecutableOp::output_shape is singular).
        // Leaving it unseeded must simply contribute nothing -- b's
        // primary-output resolution (and the whole node) still proceeds
        // normally, and a's "aux"-bound axis falls back to its own full
        // extent, exactly like an ordinary unresolved axis (e.g. a
        // reduction axis) always has.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, two_output_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let ab_edge = tile_graph.children(a)[0].1;
        let b_output_edge = tile_graph
            .output_edge_id(b)
            .expect("b has no consumer in dag");

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(50)]); // Y1's "M" only
        let config = tile_graph.propagate(&[a, b], &seed);

        let resolved = config
            .get(ab_edge)
            .expect("propagate should have resolved the a->b edge");
        assert_eq!(
            resolved,
            &vec![TileDim::Fixed(1000)],
            "expected a's \"aux\"-bound axis to fall back to its own full \
             extent (1000) when Y2 is never seeded -- got {resolved:?}"
        );
    }

    #[test]
    fn propagate_resolves_a_fan_out_producers_two_edges_independently() {
        // a feeds both b and c, each with their own flat_unary_spec and
        // their own independently-seeded output tile. EdgeId-keying means
        // a's two outgoing edges must resolve to two different tile
        // shapes, with no merging between them.
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape.clone(),
            false,
            flat_unary_spec(),
        ));
        dag.add_edge(a, b);
        let c = dag.add_node(op_with_tile_spec("c", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, c);

        let tile_graph = TileGraph::from_dag(&dag);
        let a_children = tile_graph.children(a);
        let ab_edge = a_children.iter().find(|&&(n, _)| n == b).unwrap().1;
        let ac_edge = a_children.iter().find(|&&(n, _)| n == c).unwrap().1;
        let b_output_edge = tile_graph.output_edge_id(b).unwrap();
        let c_output_edge = tile_graph.output_edge_id(c).unwrap();

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(10)]);
        seed.insert(c_output_edge, vec![TileDim::Fixed(20)]);
        let config = tile_graph.propagate(&[a, b, c], &seed);

        assert_eq!(config.get(ab_edge), Some(&vec![TileDim::Fixed(10)]));
        assert_eq!(config.get(ac_edge), Some(&vec![TileDim::Fixed(20)]));
    }

    #[test]
    fn propagate_of_an_empty_node_set_returns_the_seed_unchanged() {
        let tile_graph = TileGraph::default();
        let dummy_edge = {
            // Build a throwaway single-node graph just to mint a valid
            // EdgeId to seed with -- propagate on an empty node set must
            // not touch it either way.
            let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
            dag.add_node(op("a", vec![Some(4)], true));
            let g = TileGraph::from_dag(&dag);
            g.output_edge_id(0).unwrap()
        };

        let mut seed = HashMap::new();
        seed.insert(dummy_edge, vec![TileDim::Fixed(4)]);
        let config = tile_graph.propagate(&[], &seed);

        assert_eq!(config.len(), 1);
        assert_eq!(config.get(dummy_edge), Some(&vec![TileDim::Fixed(4)]));
    }

    fn two_level_hardware(register_capacity: u64, device_capacity: u64) -> HardwareProfile {
        HardwareProfile {
            name: "test-device".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::Register,
                    capacity: register_capacity,
                    bandwidth: None,
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: device_capacity,
                    bandwidth: None,
                    latency: None,
                },
            ],
        }
    }

    #[test]
    fn enumerate_subtiles_only_returns_power_of_two_extents() {
        // 100 isn't a power of two -- every candidate extent must still be.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(100)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let results = tile_graph.enumerate_subtiles(&[a], a, u64::MAX);

        assert!(!results.is_empty());
        for shape in &results {
            let TileDim::Fixed(extent) = shape[0] else {
                panic!("expected a Fixed axis");
            };
            assert!(extent.is_power_of_two(), "{extent} is not a power of two");
        }
    }

    #[test]
    fn enumerate_subtiles_never_exceeds_capacity() {
        // a is isolated: an F32 [64] input boundary edge (256B, fixed,
        // unaffected by the candidate) and an output boundary edge (the
        // seeded candidate). Capacity 320 = 256 + 64B admits candidate
        // extents up to 16 (16 * 4B = 64B) but not 32 (128B -> 384B total).
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let results = tile_graph.enumerate_subtiles(&[a], a, 320);

        assert!(!results.is_empty());
        for shape in &results {
            let TileDim::Fixed(extent) = shape[0] else {
                panic!("expected a Fixed axis");
            };
            assert!(
                extent <= 16,
                "extent {extent} should have exceeded capacity"
            );
        }
    }

    #[test]
    fn enumerate_subtiles_treats_a_dynamic_axis_as_unenumerable() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![None, Some(8)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let dynamic_axis = tile_graph.output_edge(a).unwrap().shape[0].clone();
        let results = tile_graph.enumerate_subtiles(&[a], a, u64::MAX);

        assert!(!results.is_empty());
        for shape in &results {
            assert_eq!(shape[0], dynamic_axis);
            assert!(matches!(shape[1], TileDim::Fixed(extent) if extent.is_power_of_two()));
        }
    }

    #[test]
    fn enumerate_subtiles_is_sorted_by_ascending_score() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let output_edge = tile_graph.output_edge_id(a).unwrap();
        let results = tile_graph.enumerate_subtiles(&[a], a, u64::MAX);

        assert!(
            results.len() > 1,
            "need at least 2 results to check ordering"
        );
        let scores: Vec<f64> = results
            .iter()
            .map(|tile| {
                tile_graph
                    .score_candidate_tile(&[a], output_edge, u64::MAX, tile)
                    .expect("every returned tile should still score as valid")
            })
            .collect();
        for pair in scores.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "results are not sorted ascending: {scores:?}"
            );
        }
    }

    #[test]
    fn enumerate_subtiles_ignores_flattened_multi_dim_axes() {
        // Regression test for teenygrad-1nr.9: b declares
        // batchnorm2d_shaped_spec (H, W flattened into one BLOCK_HW-style
        // axis via `dims: &[2, 3]`), but enumerate_subtiles builds its
        // search space purely from b's real, per-axis full shape -- with
        // no notion that dims 2 and 3 are jointly driven by one flattened
        // tile_spec axis. It grows H (dim 2) and W (dim 3) independently,
        // producing candidates like H=2, W=4 that no real
        // batch_norm_2d_nchw_forward_inference kernel configuration could
        // ever realize -- the kernel can only pick one flat BLOCK_HW count
        // over the combined H*W range (see TileAxisBinding::dims's doc
        // comment, and propagate's own convention -- teenygrad-1nr.8 -- of
        // collapsing every outer flattened dim to Fixed(1)).
        //
        // A flattened-axis-aware search should never grow H independently:
        // every candidate's dim-2 entry should stay at Fixed(1), exactly
        // like propagate already collapses it on the input side. This
        // currently FAILS -- enumerate_subtiles has no awareness of
        // tile_spec at all.
        let full_shape = vec![Some(2), Some(4), Some(8), Some(8)]; // [B, C, H, W]
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let b = dag.add_node(op_with_tile_spec(
            "b",
            full_shape,
            false,
            batchnorm2d_shaped_spec(),
        ));

        let tile_graph = TileGraph::from_dag(&dag);
        let results = tile_graph.enumerate_subtiles(&[b], b, u64::MAX);

        assert!(!results.is_empty());
        let independently_varied_h: Vec<&TileEdgeShape> = results
            .iter()
            .filter(|shape| !matches!(shape[2], TileDim::Fixed(1)))
            .collect();
        assert!(
            independently_varied_h.is_empty(),
            "expected every candidate to keep H (dim 2, the outer half of \
             the flattened HW axis) at Fixed(1) -- enumerate_subtiles has \
             no notion that H and W are jointly driven by one tile_spec \
             axis, so it grows them independently. Candidates with \
             H != 1: {independently_varied_h:?}"
        );
    }

    #[test]
    fn mem_traffic_with_config_uses_the_configured_shape_when_present() {
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let b_output_edge = tile_graph.output_edge_id(b).unwrap();

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(500)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        let full_traffic = tile_graph.mem_traffic(&[a, b]);
        let tiled_traffic = tile_graph.mem_traffic_with_config(&[a, b], &config);

        assert_eq!(full_traffic, 8000); // (1000 + 1000) * 4B
        // a's own input boundary edge is a *different* edge from a -> b
        // (which propagate did resolve to 500) -- a has no tile_spec, so
        // propagate never touches it, and it stays at its full 1000 * 4B.
        // Only b's output boundary edge (the seeded one) shrinks.
        assert_eq!(tiled_traffic, 6000); // 1000*4B (a's full input) + 500*4B (b's configured output)
    }

    #[test]
    fn mem_traffic_with_config_falls_back_to_full_shape_for_unconfigured_edges() {
        // a is isolated and has no tile_spec, so propagate resolves only
        // the seeded output edge -- a's input boundary edge stays
        // unconfigured and must fall back to its full shape.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let output_edge = tile_graph.output_edge_id(a).unwrap();

        let mut seed = HashMap::new();
        seed.insert(output_edge, vec![TileDim::Fixed(2)]);
        let config = tile_graph.propagate(&[a], &seed);
        assert_eq!(config.len(), 1, "only the seeded edge should be resolved");

        // input edge: full 64 * 4B = 256B (unconfigured); output edge:
        // configured 2 * 4B = 8B.
        assert_eq!(tile_graph.mem_traffic_with_config(&[a], &config), 264);
    }

    #[test]
    fn mem_footprint_with_config_uses_the_configured_shape() {
        let full_shape = vec![Some(1000)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let b_output_edge = tile_graph.output_edge_id(b).unwrap();

        let mut seed = HashMap::new();
        seed.insert(b_output_edge, vec![TileDim::Fixed(500)]);
        let config = tile_graph.propagate(&[a, b], &seed);

        let full_footprint = tile_graph.mem_footprint(&[a, b]);
        let tiled_footprint = tile_graph.mem_footprint_with_config(&[a, b], &config);

        assert!(
            tiled_footprint < full_footprint,
            "tiled footprint {tiled_footprint} should be smaller than full {full_footprint}"
        );
    }

    #[test]
    fn sub_graph_tiling_returns_configs_that_fit_the_level_capacity() {
        let full_shape = vec![Some(64)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", full_shape.clone(), true));
        let b = dag.add_node(op_with_tile_spec("b", full_shape, false, flat_unary_spec()));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        // a has no tile_spec, so a's own input boundary edge (64 * 4B =
        // 256B) is a fixed floor on every candidate's footprint,
        // regardless of the chosen tile -- capacity must clear it.
        let hardware = two_level_hardware(2000, u64::MAX);

        let results =
            tile_graph.sub_graph_tiling(&[a, b], b, Some(MemoryLevelKind::Register), &hardware, 5);

        assert!(!results.is_empty());
        for result in &results {
            let footprint = tile_graph.mem_footprint_with_config(&[a, b], &result.config);
            assert!(
                footprint <= 2000,
                "footprint {footprint} exceeds capacity 2000"
            );
        }
    }

    #[test]
    fn sub_graph_tiling_has_no_children_at_the_top_of_the_hierarchy() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = HardwareProfile {
            name: "single-level".to_string(),
            compute_units: 1,
            memory_levels: vec![MemoryLevel {
                kind: MemoryLevelKind::DeviceMemory,
                capacity: u64::MAX,
                bandwidth: None,
                latency: None,
            }],
        };

        let results =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::DeviceMemory), &hardware, 3);

        assert!(!results.is_empty());
        for result in &results {
            assert!(result.children.is_empty());
        }
    }

    #[test]
    fn sub_graph_tiling_recurses_one_level_when_a_higher_level_exists() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(u64::MAX, u64::MAX);

        let results =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::Register), &hardware, 3);

        assert!(!results.is_empty());
        for result in &results {
            assert!(
                !result.children.is_empty(),
                "expected recursion into DeviceMemory"
            );
            for child in &result.children {
                assert!(
                    child.children.is_empty(),
                    "DeviceMemory is the top declared level, recursion should stop there"
                );
            }
        }
    }

    #[test]
    fn sub_graph_tiling_of_none_recurses_into_the_hardwares_lowest_declared_level() {
        // Unlike `two_level_hardware` (which declares Register as a real
        // level, exercised above), real Triton hardware profiles never
        // declare Register at all -- only SharedMemory/DeviceMemory
        // (e.g. `orin_nano_hardware_profile`). `level: None` ("nothing
        // decided yet") must still correctly recurse into that lowest
        // *declared* level as its first real child, not skip it or
        // require a Register entry to exist.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = HardwareProfile {
            name: "shared-and-device".to_string(),
            compute_units: 1,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::SharedMemory,
                    capacity: u64::MAX,
                    bandwidth: None,
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: u64::MAX,
                    bandwidth: None,
                    latency: None,
                },
            ],
        };

        let results = tile_graph.sub_graph_tiling(&[a], a, None, &hardware, 3);

        assert!(!results.is_empty());
        for result in &results {
            assert!(
                !result.children.is_empty(),
                "expected recursion into SharedMemory, the hardware's lowest declared level"
            );
            for shared_memory_child in &result.children {
                assert!(
                    !shared_memory_child.children.is_empty(),
                    "expected recursion from SharedMemory into DeviceMemory"
                );
                for device_memory_child in &shared_memory_child.children {
                    assert!(
                        device_memory_child.children.is_empty(),
                        "DeviceMemory is the top declared level, recursion should stop there"
                    );
                }
            }
        }
    }

    #[test]
    fn sub_graph_tiling_respects_top_k() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(u64::MAX, u64::MAX);

        let top_1 =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::Register), &hardware, 1);
        let top_3 =
            tile_graph.sub_graph_tiling(&[a], a, Some(MemoryLevelKind::Register), &hardware, 3);

        assert_eq!(top_1.len(), 1);
        assert_eq!(top_3.len(), 3);
    }

    #[test]
    fn sub_graph_tiling_results_carry_the_node_set_they_cover() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(64)], true));
        let b = dag.add_node(op("b", vec![Some(64)], false));
        dag.add_edge(a, b);

        let tile_graph = TileGraph::from_dag(&dag);
        let hardware = two_level_hardware(u64::MAX, u64::MAX);

        let results =
            tile_graph.sub_graph_tiling(&[a, b], b, Some(MemoryLevelKind::Register), &hardware, 1);

        assert!(!results.is_empty());
        for result in &results {
            assert_eq!(result.nodes, vec![a, b]);
            // a -> b is left at from_dag's default (DeviceMemory), so
            // extract_subgraph(_, DeviceMemory)'s strict "> level" test
            // doesn't qualify it -- each child stays a singleton, one per
            // node, rather than merging into one [a, b] child.
            let mut all_child_nodes: Vec<NodeId> = result
                .children
                .iter()
                .flat_map(|c| c.nodes.clone())
                .collect();
            all_child_nodes.sort_unstable();
            assert_eq!(all_child_nodes, vec![a, b]);
            for child in &result.children {
                assert_eq!(child.nodes.len(), 1);
            }
        }
    }

    #[test]
    fn resolved_tiling_round_trips_through_record_and_get() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", vec![Some(4)], true));
        let b = dag.add_node(op("b", vec![Some(4)], false));
        dag.add_edge(a, b);

        let mut tile_graph = TileGraph::from_dag(&dag);
        let edge_id = tile_graph.children(a)[0].1;
        assert!(tile_graph.resolved_tiling(edge_id).is_none());

        let result = SubGraphTilingResult {
            nodes: vec![a, b],
            config: TileConfig::default(),
            children: Vec::new(),
        };
        tile_graph.record_resolved_tiling(edge_id, result);

        let recorded = tile_graph
            .resolved_tiling(edge_id)
            .expect("just recorded a result for this edge");
        assert_eq!(recorded.nodes, vec![a, b]);
    }
}
