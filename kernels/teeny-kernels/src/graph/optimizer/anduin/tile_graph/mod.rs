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
//! Every node's output shape therefore needs a home on some edge, including
//! nodes at the DAG's boundary that have no producer or consumer in `dag`:
//! - `parents[i]` mirrors the source `Dag` node's own `parents` field
//!   exactly: plain, *deduped* producer indices. `Dag::add_edge` already
//!   collapses a repeated `(producer, consumer)` pair before `from_dag` ever
//!   sees it, so an op that reads the same producer through two operand
//!   slots (e.g. `Add(x, x)`) is indistinguishable here from one that reads
//!   it once — that per-operand-slot detail lived in the source
//!   [`teeny_core::graph::Graph`]'s `inputs` list and isn't reconstructible
//!   from `Dag` alone. If a later pass needs it, expose it as a new
//!   [`ExecutableOp`] method instead of threading `Graph` back in here —
//!   lowering has already happened by the time `Anduin` runs.
//! - `children[i]` is the fanout view: which distinct consumers read node
//!   `i`'s output, and at which shape/memory level. One entry per
//!   *consumer*, not per operand slot, for the same reason as `parents`.
//! - `input_edges[i]` is `Some` iff node `i` has no producer in `dag` (empty
//!   `parents[i]`, e.g. a lowered `Input` op): a boundary edge carrying that
//!   node's shape in from outside the DAG.
//! - `output_edges[i]` is `Some` iff node `i` has no consumer in `dag`
//!   (empty `children[i]`, i.e. nothing else in this DAG reads it): a
//!   boundary edge carrying that node's shape out to the DAG's caller.

use teeny_core::device::hardware::MemoryLevelKind;
use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::ExecutableOp;
use teeny_core::utils::dag::Dag;

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

/// One node in a [`TileGraph`]: an [`ExecutableOp`]'s name and output dtype.
/// Shape is deliberately not here — see the module doc comment — nor are
/// producer/consumer edges, which live in the owning [`TileGraph`]'s
/// `parents`/`children`/boundary tables.
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
/// tile-graph form: same DAG structure, one [`TileOp`] per source node. See
/// the module doc comment for how `parents`/`children`/boundary edges
/// divide up edge data.
#[derive(Debug, Default)]
pub struct TileGraph {
    nodes: Vec<TileOp>,
    parents: Vec<Vec<usize>>,
    children: Vec<Vec<(usize, TileEdge)>>,
    input_edges: Vec<Option<TileEdge>>,
    output_edges: Vec<Option<TileEdge>>,
}

impl TileGraph {
    /// Converts `dag` into a `TileGraph` with identical DAG structure: each
    /// `dag` node becomes one [`TileOp`], `parents`/`children` are carried
    /// over verbatim from `dag`'s own producer/consumer edges, and boundary
    /// edges are filled in for nodes with no producer/consumer in `dag`.
    /// Every edge starts at [`MemoryLevelKind::DeviceMemory`]: every tensor
    /// starts out materialized in device memory until the memory-level
    /// search promotes an edge to a faster level.
    pub fn from_dag(dag: &Dag<Box<dyn ExecutableOp>>) -> Self {
        let n = dag.len();
        let mut nodes = Vec::with_capacity(n);
        let mut parents: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut children: Vec<Vec<(usize, TileEdge)>> = Vec::with_capacity(n);
        let mut input_edges: Vec<Option<TileEdge>> = Vec::with_capacity(n);

        for index in 0..n {
            let node = dag.node(index);
            let shape = node.value.output_shape();

            nodes.push(TileOp {
                name: node.value.name().to_string(),
                dtype: node.value.output_dtype(),
            });
            parents.push(node.parents.clone());
            input_edges.push(node.parents.is_empty().then(|| TileEdge {
                shape: to_tile_shape(index, shape),
                memory_level: MemoryLevelKind::DeviceMemory,
            }));
            children.push(
                node.children
                    .iter()
                    .map(|&consumer| {
                        (
                            consumer,
                            TileEdge {
                                shape: to_tile_shape(index, shape),
                                memory_level: MemoryLevelKind::DeviceMemory,
                            },
                        )
                    })
                    .collect(),
            );
        }

        let output_edges = (0..n)
            .map(|index| {
                children[index].is_empty().then(|| TileEdge {
                    shape: to_tile_shape(index, dag.node(index).value.output_shape()),
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

    /// Deduped producer indices for the node at `index` (see the module doc
    /// comment for why duplicate operand slots aren't distinguishable here).
    pub fn parents(&self, index: usize) -> &[usize] {
        &self.parents[index]
    }

    /// Consumers of the node at `index`'s output, as `(consumer_index, edge)`
    /// pairs. One entry per distinct consumer — see the module doc comment.
    pub fn children(&self, index: usize) -> &[(usize, TileEdge)] {
        &self.children[index]
    }

    /// The boundary edge carrying node `index`'s value in from outside the
    /// DAG, if it has no producer in `dag` (empty `parents(index)`).
    pub fn input_edge(&self, index: usize) -> Option<&TileEdge> {
        self.input_edges[index].as_ref()
    }

    /// The boundary edge carrying node `index`'s value out to the DAG's
    /// caller, if it has no consumer in `dag` (empty `children(index)`).
    pub fn output_edge(&self, index: usize) -> Option<&TileEdge> {
        self.output_edges[index].as_ref()
    }

    /// Returns node indices in topological order (producers before
    /// consumers) using Kahn's algorithm. Panics if the graph contains a
    /// cycle.
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
        assert_eq!(tile_graph.parents(relu), &[input]);

        assert_eq!(tile_graph.node(sigmoid).name, "sigmoid");
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
    fn fan_in_preserves_distinct_producers_in_insertion_order() {
        let shape = vec![Some(4)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let a = dag.add_node(op("a", shape.clone(), true));
        let b = dag.add_node(op("b", shape.clone(), true));
        let add = dag.add_node(op("add", shape.clone(), false));
        dag.add_edge(a, add);
        dag.add_edge(b, add);

        let tile_graph = TileGraph::from_dag(&dag);

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

        assert_eq!(tile_graph.parents(add), &[x]);
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
            for (_, edge) in tile_graph.children(node) {
                assert_eq!(edge.memory_level, MemoryLevelKind::DeviceMemory);
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

        let edge = &tile_graph.children(input)[0].1;
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
        // `children` edges, not double-counted — otherwise `add`'s in-degree
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
}
