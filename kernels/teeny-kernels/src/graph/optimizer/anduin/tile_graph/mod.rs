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

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::ExecutableOp;
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// [`TileOp::dtype`]). A dynamic ([`TileDim::Sym`]) axis counts as
    /// extent 1, since no real tile shape exists yet
    /// (`Propagate`/`TileConfig` are still unbuilt) — this makes the result
    /// optimistic, never an overestimate, of the true size.
    pub fn byte_size(&self, dtype: DtypeRepr) -> u64 {
        let elements: u64 = self
            .shape
            .iter()
            .map(|dim| match dim {
                TileDim::Fixed(extent) => *extent as u64,
                TileDim::Sym(_) => 1,
            })
            .product();
        elements * dtype_bytes(dtype)
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

        for index in 0..n {
            let node = dag.node(index);
            let shape = node.value.output_shape();

            nodes.push(TileOp {
                name: node.value.name().to_string(),
                dtype: node.value.output_dtype(),
            });

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

    /// Welder §3.2's `ExtractSubgraph`: the node set reachable from `node`
    /// by transitively following edges (in either direction) whose connect
    /// level is *above* `level` — i.e. everything `GraphConnecting` has
    /// already decided to fuse more tightly than `level`. Matches Fig. 7's
    /// `SubGraph(nodes)`: a set of [`NodeId`]s into *this* graph, not a
    /// separate rebuilt one — downstream cost-model passes (`MemFootprint`,
    /// `MemTraffic`, ...) can test edge endpoints against the set directly
    /// rather than needing copied-and-relabeled edge data.
    ///
    /// Unlike the paper's naive tree recursion, this shares one visited set
    /// across the whole walk: two paths converging on the same node (e.g. a
    /// diamond fan-out/fan-in) expand it once, not once per incoming path.
    ///
    /// `node` is always included, even in isolation (e.g. every one of its
    /// edges is at or below `level`) — the returned set is never empty.
    pub fn extract_subgraph(&self, node: NodeId, level: MemoryLevelKind) -> Vec<NodeId> {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = vec![node];
        visited[node] = true;

        while let Some(current) = stack.pop() {
            for &id in self.outgoing[current]
                .iter()
                .chain(self.incoming[current].iter())
            {
                let record = &self.edges[id.0];
                if record.edge.memory_level <= level {
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

        (0..self.nodes.len()).filter(|&index| visited[index]).collect()
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

    /// Welder §3.1's `MemTraffic`: total bytes crossing the boundary of
    /// `nodes` (see [`Self::boundary_edges`]), using the full untiled shape
    /// `from_dag` captured for each edge — no distinct tile shape exists
    /// yet (`Propagate`/`TileConfig` are still unbuilt), so this is exactly
    /// correct only in the degenerate case where the whole tensor *is* the
    /// tile; otherwise it's what the paper's traffic formula would compute
    /// before multiplying by the number of tile-graphs needed to cover the
    /// full output.
    pub fn mem_traffic(&self, nodes: &[NodeId]) -> u64 {
        self.boundary_edges(nodes)
            .into_iter()
            .map(|(edge_id, dtype)| self.edge(edge_id).byte_size(dtype))
            .sum()
    }

    /// The byte size of `node`'s own output tile, read from whichever of
    /// its outgoing edges exists (they all carry the same shape — see the
    /// module doc comment) — an output boundary edge if it has no consumer
    /// in `dag`, otherwise its first child edge.
    fn output_byte_size(&self, node: NodeId) -> u64 {
        let dtype = self.node(node).dtype;
        self.output_edge(node)
            .map(|edge| edge.byte_size(dtype))
            .or_else(|| {
                self.children(node)
                    .first()
                    .map(|&(_, id)| self.edge(id).byte_size(dtype))
            })
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
    /// model yet.
    pub fn mem_footprint(&self, nodes: &[NodeId]) -> u64 {
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
                .map(|(producer, id)| self.edge(id).byte_size(self.node(producer).dtype))
                .sum::<u64>()
                + self
                    .input_edge(node)
                    .map(|edge| edge.byte_size(dtype))
                    .unwrap_or(0);

            let out_bytes = self.output_byte_size(node);

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
    use teeny_core::device::hardware::MemoryLevelKind;
    use teeny_core::graph::{DtypeRepr, Shape};
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::{TileDim, TileEdge, TileGraph};

    /// Minimal [`ExecutableOp`] test double: just enough surface
    /// (name/shape/dtype) for [`TileGraph::from_dag`] to convert on.
    struct TestOp {
        name: &'static str,
        dtype: DtypeRepr,
        shape: Shape,
        is_input: bool,
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

        assert_eq!(
            tile_graph.connect_level(id),
            MemoryLevelKind::DeviceMemory
        );

        tile_graph.set_connect(id, MemoryLevelKind::SharedMemory);

        assert_eq!(
            tile_graph.connect_level(id),
            MemoryLevelKind::SharedMemory
        );
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
        let sub = tile_graph.extract_subgraph(a, MemoryLevelKind::DeviceMemory);

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

        let mut sub = tile_graph.extract_subgraph(b, MemoryLevelKind::DeviceMemory);
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

        let mut sub = tile_graph.extract_subgraph(a, MemoryLevelKind::DeviceMemory);
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

        let mut sub = tile_graph.extract_subgraph(c, MemoryLevelKind::DeviceMemory);
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
}
