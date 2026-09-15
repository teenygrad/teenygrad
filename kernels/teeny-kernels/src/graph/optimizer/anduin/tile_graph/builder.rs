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

//! [`TileGraphBuilder`] — accumulates a [`TileGraph`]'s nodes and edge
//! arena one source `Dag` node at a time, then hands them off to
//! [`TileGraph::from_dag`] to finish. Kept separate from the growing state
//! itself (rather than building it up in a handful of parallel local
//! `Vec`s the way a single free function would) so the per-node step
//! (`push_node`) and the arena-linking step (`push_edge`) are each just one
//! self-contained method to read, with no risk of one of the parallel
//! `Vec`s falling out of sync with the others.

use std::collections::HashMap;

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::{Dag, Node};

use super::types::{TileDim, TileEdge, TileEdgeRecord, TileOp, to_tile_shape};
use super::{EdgeId, NodeId, TileGraph};

/// Accumulates the arrays a [`TileGraph`] is made of: one [`TileOp`] and
/// its edges per source `Dag` node, pushed via [`Self::push_node`] in
/// index order, then finished with [`Self::build`].
struct TileGraphBuilder {
    nodes: Vec<TileOp>,
    edges: Vec<TileEdgeRecord>,
    outgoing: Vec<Vec<EdgeId>>,
    incoming: Vec<Vec<EdgeId>>,
    secondary_outputs: Vec<Vec<EdgeId>>,
}

impl TileGraphBuilder {
    /// A builder pre-sized for `n` source nodes.
    fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
            edges: Vec::new(),
            outgoing: vec![Vec::new(); n],
            incoming: vec![Vec::new(); n],
            secondary_outputs: Vec::with_capacity(n),
        }
    }

    /// Pushes one arena edge and links it into `producer`'s `outgoing`
    /// list and/or `consumer`'s `incoming` list, whichever side(s) are
    /// `Some`.
    fn push_edge(&mut self, producer: Option<NodeId>, consumer: Option<NodeId>, edge: TileEdge) {
        let id = EdgeId(self.edges.len());
        self.edges.push(TileEdgeRecord {
            producer,
            consumer,
            edge,
        });
        if let Some(p) = producer {
            self.outgoing[p].push(id);
        }
        if let Some(c) = consumer {
            self.incoming[c].push(id);
        }
    }

    /// Converts one source `dag` node (at `index`) into a [`TileOp`] plus
    /// its edges: the node's own outgoing edge(s) (a boundary output edge
    /// if it has no consumer in `dag`, otherwise one per child), a
    /// boundary input edge if it has no producer in `dag`, and one
    /// placeholder boundary edge per additional declared output beyond the
    /// primary one (teenygrad-1nr.11).
    fn push_node(&mut self, index: NodeId, node: &Node<Box<dyn ExecutableOp>>) {
        let shape = node.value.output_shape();
        let tile_spec = node.value.tile_spec();

        self.nodes.push(TileOp {
            name: node.value.name().to_string(),
            dtype: node.value.output_dtype(),
            tile_spec,
        });

        // Additional declared outputs beyond the primary one
        // (teenygrad-1nr.11): no ground-truth shape exists for these
        // (ExecutableOp::output_shape is singular), so every axis is
        // an unresolved placeholder symbol -- these edges exist only
        // so a caller can seed a requested tile onto them for
        // TileGraph::propagate to pick up; see
        // TileGraph::secondary_output_edges's doc comment for why they're
        // deliberately not linked into outgoing/incoming.
        let mut node_secondary_outputs = Vec::new();
        if let Some(spec) = &tile_spec {
            for (output_index, output_spec) in spec.outputs.iter().enumerate().skip(1) {
                let placeholder_shape = (0..output_spec.rank)
                    .map(|axis| TileDim::Sym(format!("n{index}o{output_index}d{axis}")))
                    .collect();
                let id = EdgeId(self.edges.len());
                self.edges.push(TileEdgeRecord {
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
        self.secondary_outputs.push(node_secondary_outputs);

        if node.parents.is_empty() {
            self.push_edge(
                None,
                Some(index),
                TileEdge {
                    shape: to_tile_shape(index, shape),
                    memory_level: MemoryLevelKind::DeviceMemory,
                },
            );
        }

        if node.children.is_empty() {
            self.push_edge(
                Some(index),
                None,
                TileEdge {
                    shape: to_tile_shape(index, shape),
                    memory_level: MemoryLevelKind::DeviceMemory,
                },
            );
        } else {
            for &consumer in &node.children {
                self.push_edge(
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

    /// Finishes accumulation into a [`TileGraph`].
    fn build(self) -> TileGraph {
        TileGraph {
            nodes: self.nodes,
            edges: self.edges,
            outgoing: self.outgoing,
            incoming: self.incoming,
            secondary_outputs: self.secondary_outputs,
            resolved_tiling: HashMap::new(),
        }
    }
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
        let mut builder = TileGraphBuilder::with_capacity(dag.len());
        for index in 0..dag.len() {
            builder.push_node(index, dag.node(index));
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::device::hardware::MemoryLevelKind;
    use teeny_core::model::ExecutableOp;
    use teeny_core::utils::dag::Dag;

    use super::super::testing::op;
    use super::super::{TileDim, TileEdge, TileGraph};

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
}
