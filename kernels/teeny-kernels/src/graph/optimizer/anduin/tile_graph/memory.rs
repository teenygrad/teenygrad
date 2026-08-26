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

//! Welder §3.1's `MemTraffic`/`MemFootprint` cost-model estimates over a
//! node set.

use std::collections::{HashMap, HashSet};

use teeny_core::graph::DtypeRepr;

use super::types::{TileConfig, shape_byte_size};
use super::{EdgeId, NodeId, TileGraph};

impl TileGraph {
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
    /// [`super::types::TileEdge`]'s doc comment on why levels can
    /// legitimately differ per consumer) — a node whose edges end up at
    /// genuinely different levels could in principle need more than one
    /// buffer, which this doesn't model yet. Uses each edge's full untiled
    /// shape — see [`Self::mem_footprint_with_config`] for the tiled
    /// version.
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::super::testing::{flat_unary_spec, op, op_with_tile_spec};
    use super::super::{TileDim, TileGraph};

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
}
