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
//!
//! ## Files
//!
//! [`types`] holds the value types (`TileDim`, `TileEdge`, `TileOp`, ...)
//! everything else builds on. [`builder`] holds [`TileGraph::from_dag`].
//! Everything else here is `TileGraph`'s own inherent methods, split by
//! concern: this file for construction-independent navigation (accessors,
//! `extract_subgraph`, `topological_sort`), [`memory`] for the
//! `MemTraffic`/`MemFootprint` cost model, [`propagate`] for `Propagate`,
//! [`search`] for `EnumerateSubtiles`, and [`sub_graph_tiling`] for
//! `SubGraphTiling`'s recursion up the memory hierarchy.

use std::collections::HashMap;

use teeny_core::device::hardware::MemoryLevelKind;

mod builder;
mod memory;
mod propagate;
mod search;
mod sub_graph_tiling;
#[cfg(test)]
mod testing;
mod types;

use types::TileEdgeRecord;
pub use types::{
    EdgeId, SubGraphTilingResult, TileConfig, TileDim, TileEdge, TileEdgeShape, TileOp,
};

/// Index of a node in a [`TileGraph`], stable for the lifetime of that
/// graph (assigned by [`TileGraph::from_dag`]/[`TileGraph::extract_subgraph`]
/// in the order their nodes are visited). A plain alias, not a newtype: it's
/// interchangeable with the source `Dag`'s own node indices, which is what
/// lets `from_dag` carry them over unchanged.
pub type NodeId = usize;

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
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::TileGraph;
    use super::testing::op;

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
}
