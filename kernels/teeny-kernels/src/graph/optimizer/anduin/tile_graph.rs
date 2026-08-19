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
//! [`TileGraph::from_graph`] converts a [`Graph`] into the same DAG shape with
//! [`TileOp`] nodes in place of [`GraphNode`](teeny_core::graph::GraphNode)s.
//! It is a pure structural conversion: producer/consumer edges are carried
//! over one-to-one. Tile-shape propagation (backward from the graph output,
//! as expressions in shared free variables) and the memory-hierarchy-level
//! search are later passes — see this module's parent doc comment.
//!
//! Shape is an edge concept, not a node concept: a `TileOp` is just an
//! operation, and shape describes a *value* flowing on an edge, which is
//! also exactly what a later tiling pass needs to refine per-edge (the same
//! producer can be tiled differently for different consumers). So shape
//! lives on [`TileEdge`], never on [`TileOp`] — as [`TileEdgeShape`], not the
//! ordinary [`Shape`](teeny_core::graph::Shape): a tile-graph axis can be
//! [`TileDim::Sym`], a named free variable, not just a known-or-dynamic
//! extent. `from_graph` synthesizes one fresh symbol per dynamic axis of the
//! source shape; it does not unify symbols that turn out to be the same free
//! variable across edges — that's the later propagation pass's job.
//!
//! Every node's output shape therefore needs a home on some edge, including
//! nodes at the graph's boundary that have no in-graph producer or consumer:
//! - `parents[i]` mirrors [`GraphNode::inputs`](teeny_core::graph::GraphNode::inputs)
//!   exactly — the ordered producer indices a node needs for shape
//!   propagation, duplicates and all (e.g. `Add(x, x)` keeps both `x`
//!   entries). Plain indices; a node doesn't care what memory level its
//!   operands arrive at to determine its own output shape.
//! - `children[i]` is the fanout view: which distinct consumers read node
//!   `i`'s output, and at which shape/memory level. One entry per
//!   *consumer*, not per operand slot: a single materialized tile lives at
//!   one memory level, so a producer read twice by the same consumer (e.g.
//!   `Add(x, x)`) is still one edge, not two that could disagree.
//! - `input_edges[i]` is `Some` iff node `i` has no in-graph producer (empty
//!   `parents[i]`, e.g. `Op::Input`, `Op::Constant`): a boundary edge
//!   carrying that node's shape in from outside the graph.
//! - `output_edges[i]` is `Some` iff node `i` has no in-graph consumer (empty
//!   `children[i]`, i.e. nothing else in this graph reads it): a boundary
//!   edge carrying that node's shape out to the graph's caller.

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::graph::{DtypeRepr, Graph, Op, Shape};

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
fn to_tile_shape(node_index: usize, shape: &Shape) -> TileEdgeShape {
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
/// carries. Used both for internal producer→consumer edges (`children`) and
/// for graph-boundary edges (`input_edges`/`output_edges`) — see the module
/// doc comment.
#[derive(Debug, Clone, PartialEq)]
pub struct TileEdge {
    /// Shape of the value this edge carries.
    pub shape: TileEdgeShape,
    /// Memory level this value is materialized at on this edge.
    pub memory_level: MemoryLevelKind,
}

/// One node in a [`TileGraph`]: an [`Op`] plus its output dtype, mirroring
/// [`GraphNode`](teeny_core::graph::GraphNode). Shape is deliberately not
/// here — see the module doc comment — nor are producer/consumer edges,
/// which live in the owning [`TileGraph`]'s `parents`/`children`/boundary
/// tables.
#[derive(Debug, Clone)]
pub struct TileOp {
    /// The source operation this tile node computes.
    pub op: Op,
    /// Output dtype, carried over from the source [`GraphNode`](teeny_core::graph::GraphNode).
    pub dtype: DtypeRepr,
}

/// A [`Graph`] converted to Welder's tile-graph form: same DAG structure, one
/// [`TileOp`] per [`GraphNode`](teeny_core::graph::GraphNode). See the module
/// doc comment for how `parents`/`children`/boundary edges divide up edge data.
#[derive(Debug, Default)]
pub struct TileGraph {
    nodes: Vec<TileOp>,
    parents: Vec<Vec<usize>>,
    children: Vec<Vec<(usize, TileEdge)>>,
    input_edges: Vec<Option<TileEdge>>,
    output_edges: Vec<Option<TileEdge>>,
}

impl TileGraph {
    /// Converts `graph` into a `TileGraph` with identical DAG structure: each
    /// [`GraphNode`](teeny_core::graph::GraphNode) becomes one [`TileOp`],
    /// `inputs` is carried over verbatim into `parents`, `children` is built
    /// as the reverse index (deduped to one entry per distinct
    /// `(producer, consumer)` pair), and boundary edges are filled in for
    /// nodes with no in-graph producer/consumer. Every edge starts at
    /// [`MemoryLevelKind::DeviceMemory`]: every tensor starts out
    /// materialized in device memory until the memory-level search promotes
    /// an edge to a faster level.
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.nodes.len();
        let mut nodes = Vec::with_capacity(n);
        let mut parents: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut children: Vec<Vec<(usize, TileEdge)>> = vec![Vec::new(); n];
        let mut input_edges: Vec<Option<TileEdge>> = vec![None; n];

        for (index, node) in graph.nodes.iter().enumerate() {
            nodes.push(TileOp {
                op: node.op.clone(),
                dtype: node.dtype,
            });
            parents.push(node.inputs.clone());

            if node.inputs.is_empty() {
                input_edges[index] = Some(TileEdge {
                    shape: to_tile_shape(index, &node.shape),
                    memory_level: MemoryLevelKind::DeviceMemory,
                });
            }

            for &producer in &node.inputs {
                let already_connected =
                    children[producer].iter().any(|&(consumer, _)| consumer == index);
                if !already_connected {
                    children[producer].push((
                        index,
                        TileEdge {
                            shape: to_tile_shape(producer, &graph.nodes[producer].shape),
                            memory_level: MemoryLevelKind::DeviceMemory,
                        },
                    ));
                }
            }
        }

        let output_edges = (0..n)
            .map(|index| {
                children[index].is_empty().then(|| TileEdge {
                    shape: to_tile_shape(index, &graph.nodes[index].shape),
                    memory_level: MemoryLevelKind::DeviceMemory,
                })
            })
            .collect();

        Self {
            nodes,
            parents,
            children,
            input_edges,
            output_edges,
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
    pub fn node(&self, index: usize) -> &TileOp {
        &self.nodes[index]
    }

    /// Ordered producer indices for the node at `index` (its operands, same
    /// order as the source [`GraphNode::inputs`](teeny_core::graph::GraphNode::inputs)).
    pub fn parents(&self, index: usize) -> &[usize] {
        &self.parents[index]
    }

    /// Consumers of the node at `index`'s output, as `(consumer_index, edge)`
    /// pairs. Unordered; a node with two consumers (or one consumer reading
    /// it twice) has one entry per connection.
    pub fn children(&self, index: usize) -> &[(usize, TileEdge)] {
        &self.children[index]
    }

    /// The boundary edge carrying node `index`'s value in from outside the
    /// graph, if it has no in-graph producer (empty `parents(index)`).
    pub fn input_edge(&self, index: usize) -> Option<&TileEdge> {
        self.input_edges[index].as_ref()
    }

    /// The boundary edge carrying node `index`'s value out to the graph's
    /// caller, if it has no in-graph consumer (empty `children(index)`).
    pub fn output_edge(&self, index: usize) -> Option<&TileEdge> {
        self.output_edges[index].as_ref()
    }

    /// Returns node indices in topological order (producers before
    /// consumers) using Kahn's algorithm. Panics if the graph contains a
    /// cycle.
    ///
    /// In-degree is derived from `children`, not `parents.len()`: `parents`
    /// keeps duplicate operand references (e.g. `Add(x, x)` has two `x`
    /// entries), but `children` dedups those to one edge, so in-degree must
    /// count the same deduped edges that will actually be decremented below.
    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for node_children in &self.children {
            for &(child, _) in node_children {
                in_degree[child] += 1;
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(index) = queue.pop() {
            order.push(index);
            for &(child, _) in &self.children[index] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push(child);
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
    use teeny_core::graph::{DtypeRepr, Graph, Op};

    use super::{TileDim, TileEdge, TileGraph};

    #[test]
    fn empty_graph_produces_empty_tile_graph() {
        let graph = Graph::new();
        let tile_graph = TileGraph::from_graph(&graph);
        assert!(tile_graph.is_empty());
        assert_eq!(tile_graph.len(), 0);
    }

    #[test]
    fn linear_chain_preserves_node_count_and_edges() {
        let mut graph = Graph::new();
        let shape = vec![Some(4), Some(8)];
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
        let sigmoid = graph.add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);
        assert_eq!(tile_graph.len(), 3);

        assert!(matches!(tile_graph.node(input).op, Op::Input));
        assert!(tile_graph.parents(input).is_empty());

        assert!(matches!(tile_graph.node(relu).op, Op::Relu));
        assert_eq!(tile_graph.parents(relu), &[input]);

        assert!(matches!(tile_graph.node(sigmoid).op, Op::Sigmoid));
        assert_eq!(tile_graph.parents(sigmoid), &[relu]);

        let fixed_shape = vec![TileDim::Fixed(4), TileDim::Fixed(8)];

        // Fanout mirrors the operand edges above, one hop forward, carrying
        // the producer's shape.
        assert_eq!(tile_graph.children(input).len(), 1);
        assert_eq!(tile_graph.children(input)[0].0, relu);
        assert_eq!(tile_graph.children(input)[0].1.shape, fixed_shape);

        assert_eq!(tile_graph.children(relu).len(), 1);
        assert_eq!(tile_graph.children(relu)[0].0, sigmoid);
        assert_eq!(tile_graph.children(relu)[0].1.shape, fixed_shape);

        assert!(tile_graph.children(sigmoid).is_empty());
    }

    #[test]
    fn fan_in_preserves_distinct_producers_in_operand_order() {
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let a = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let b = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let add = graph.add_node(Op::Add, vec![a, b], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

        // Operand order (not insertion order) must survive the conversion.
        assert_eq!(tile_graph.parents(add), &[a, b]);

        assert_eq!(
            tile_graph.children(a),
            &[(
                add,
                TileEdge {
                    shape: vec![TileDim::Fixed(4)],
                    memory_level: MemoryLevelKind::DeviceMemory,
                }
            )]
        );
        assert_eq!(
            tile_graph.children(b),
            &[(
                add,
                TileEdge {
                    shape: vec![TileDim::Fixed(4)],
                    memory_level: MemoryLevelKind::DeviceMemory,
                }
            )]
        );
    }

    #[test]
    fn self_referential_operand_produces_a_single_edge() {
        // Add(x, x): the same producer feeds two different operand slots on
        // the same consumer, but it's still one materialized tile read once
        // by that consumer — `children` must collapse this to one edge (a
        // tile can't live at two memory levels at once). `parents` still
        // keeps both operand references, since shape propagation needs them.
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let add = graph.add_node(Op::Add, vec![x, x], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

        assert_eq!(tile_graph.parents(add), &[x, x]);
        assert_eq!(
            tile_graph.children(x),
            &[(
                add,
                TileEdge {
                    shape: vec![TileDim::Fixed(4)],
                    memory_level: MemoryLevelKind::DeviceMemory,
                }
            )]
        );
    }

    #[test]
    fn fan_out_preserves_multiple_distinct_consumers() {
        // x feeds two different downstream ops.
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![x], DtypeRepr::F32, shape.clone());
        let sigmoid = graph.add_node(Op::Sigmoid, vec![x], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

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
    fn from_graph_leaves_every_edge_in_device_memory() {
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let a = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let b = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        graph.add_node(Op::Add, vec![a, b], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

        for &node in &[a, b] {
            for (_, edge) in tile_graph.children(node) {
                assert_eq!(edge.memory_level, MemoryLevelKind::DeviceMemory);
            }
        }
    }

    #[test]
    fn nodes_with_no_producer_get_an_input_edge() {
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

        let edge = tile_graph
            .input_edge(input)
            .expect("Input node has no in-graph producer");
        assert_eq!(edge.shape, vec![TileDim::Fixed(4)]);
        assert_eq!(edge.memory_level, MemoryLevelKind::DeviceMemory);

        // relu has a real producer, so it's not a graph-input boundary node.
        assert!(tile_graph.input_edge(relu).is_none());
    }

    #[test]
    fn input_edge_applies_to_any_zero_input_op_not_just_op_input() {
        // The condition is structural (empty `inputs`), not `Op::Input`
        // specifically — e.g. `Op::Constant` also has no in-graph producer.
        let mut graph = Graph::new();
        let shape = vec![Some(2), Some(2)];
        let constant = graph.add_node(
            Op::Constant {
                dtype: teeny_core::graph::DtypeRepr::F32,
                shape: shape.clone(),
            },
            vec![],
            DtypeRepr::F32,
            shape,
        );

        let tile_graph = TileGraph::from_graph(&graph);
        let edge = tile_graph
            .input_edge(constant)
            .expect("zero-input Constant node is a graph-input boundary node");
        assert_eq!(edge.shape, vec![TileDim::Fixed(2), TileDim::Fixed(2)]);
    }

    #[test]
    fn nodes_with_no_consumer_get_an_output_edge() {
        let mut graph = Graph::new();
        let shape = vec![Some(4), Some(8)];
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
        let sigmoid = graph.add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

        // Only the graph's true sink (sigmoid) is a graph-output boundary node.
        assert!(tile_graph.output_edge(input).is_none());
        assert!(tile_graph.output_edge(relu).is_none());

        let edge = tile_graph
            .output_edge(sigmoid)
            .expect("sigmoid has no in-graph consumer");
        assert_eq!(edge.shape, vec![TileDim::Fixed(4), TileDim::Fixed(8)]);
        assert_eq!(edge.memory_level, MemoryLevelKind::DeviceMemory);
    }

    #[test]
    fn fan_out_node_with_all_consumers_present_has_no_output_edge() {
        // x has two in-graph consumers, so it is not a graph output even
        // though it also happens to be a graph input.
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        graph.add_node(Op::Relu, vec![x], DtypeRepr::F32, shape.clone());
        graph.add_node(Op::Sigmoid, vec![x], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);
        assert!(tile_graph.input_edge(x).is_some());
        assert!(tile_graph.output_edge(x).is_none());
    }

    #[test]
    fn dynamic_axis_becomes_a_symbolic_dim() {
        // A `None` (dynamic/unknown) axis in the source shape becomes a
        // synthesized `TileDim::Sym`, not a `Fixed` extent — from_graph
        // doesn't know the runtime value, and unifying symbols that are
        // actually the same free variable across edges is the later
        // propagation pass's job.
        let mut graph = Graph::new();
        let shape = vec![None, Some(8)]; // e.g. a dynamic batch axis
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);

        let edge = &tile_graph.children(input)[0].1;
        assert_eq!(edge.shape[1], TileDim::Fixed(8));
        assert!(matches!(edge.shape[0], TileDim::Sym(_)));
    }

    #[test]
    fn distinct_dynamic_axes_get_distinct_symbols() {
        // from_graph must not accidentally collide two different (node,
        // axis) dynamic dims onto the same synthesized name — that would
        // silently assert a shared-free-variable relationship that doesn't
        // exist yet (unification is the propagation pass's job).
        let mut graph = Graph::new();
        let a = graph.add_node(Op::Input, vec![], DtypeRepr::F32, vec![None]);
        let b = graph.add_node(Op::Input, vec![], DtypeRepr::F32, vec![None]);

        let tile_graph = TileGraph::from_graph(&graph);

        let a_sym = tile_graph.input_edge(a).unwrap().shape[0].clone();
        let b_sym = tile_graph.input_edge(b).unwrap().shape[0].clone();
        assert_ne!(a_sym, b_sym);
    }

    #[test]
    fn topological_sort_orders_producers_before_consumers() {
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let a = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let b = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let add = graph.add_node(Op::Add, vec![a, b], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![add], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);
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
        // `children` edges, not from `parents.len()` (which double-counts
        // `Add(x, x)`) — otherwise `add`'s in-degree never reaches zero and
        // this would panic with "tile graph contains a cycle".
        let mut graph = Graph::new();
        let shape = vec![Some(4)];
        let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let add = graph.add_node(Op::Add, vec![x, x], DtypeRepr::F32, shape.clone());
        let relu = graph.add_node(Op::Relu, vec![add], DtypeRepr::F32, shape);

        let tile_graph = TileGraph::from_graph(&graph);
        let order = tile_graph.topological_sort();

        assert_eq!(order.len(), 3);
        let position = |node: usize| order.iter().position(|&i| i == node).unwrap();
        assert!(position(x) < position(add));
        assert!(position(add) < position(relu));
    }
}
