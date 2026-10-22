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

use std::boxed::Box;

use teeny_core::{
    graph::DtypeRepr,
    model::{ExecutableOp, GridSpec, KernelTileSpec},
    utils::dag::Dag,
};

use crate::errors::{Error, Result};
use crate::graph::tile_resolve::ConstLookup;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(pub usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct EdgeId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Placeholder,
    Output,
    Op,
}

/// Stored ONCE, in `TileGraph::edges` (fix #14). Ops are single-output (P9), so
/// the producer side is just a node.
pub struct Edge {
    pub id: EdgeId,
    pub src: NodeId,
    pub dst: NodeId,
    pub dst_port: usize,
}

pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub name: String,
    pub in_edges: Vec<EdgeId>, // in_edges[p] feeds operand p  (P5)
    pub out_edges: Vec<EdgeId>,
    pub shape: Vec<u64>, // output shape, static (P1)
    pub dtype: DtypeRepr,
    pub meta: Option<OpMeta>, // None ⇒ opaque: never fused
}

pub struct OpMeta {
    pub spec: KernelTileSpec,
    pub grid: GridSpec,
    pub consts: Box<dyn ConstLookup>, // this node's values (P3)
    pub rider: bool, // fusion_core().is_some(): may be spliced into another program
    pub flops_per_output: f64, // includes the op's own reduction work per output element
}

pub struct TileGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub order: Vec<NodeId>, // A0: Welder priority topological order
    pub pos: Vec<usize>,    // pos[n.0] = index of n in `order`
}

impl TileGraph {
    /// Convert an already-lowered `Dag` (e.g. from [`TritonLowering`]) into a
    /// `TileGraph`, walking the DAG in topological order.
    ///
    /// Each `is_input()` op becomes a `Placeholder` and every other op an `Op`;
    /// every DAG sink also feeds a trailing `Output` node. A node's `p`-th DAG
    /// parent feeds its operand `p`. `Dag::add_edge` drops duplicate edges, so an
    /// op that reads the same tensor twice (`x * x`) gets a single in-edge.
    ///
    /// `op_meta(dag_idx, op)` supplies each `Op` node's [`OpMeta`]; `None` makes
    /// the node opaque (never fused). The DAG's ops expose only their
    /// `tile_spec()`, so the grid spec, consts, rider flag and flop count have
    /// to come from the caller.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if an op's output shape has a dynamic
    /// dim, since the scheduler only handles static shapes (P1).
    ///
    /// [`TritonLowering`]: crate::graph::TritonLowering
    pub fn from_dag(
        dag: &Dag<Box<dyn ExecutableOp>>,
        mut op_meta: impl FnMut(usize, &dyn ExecutableOp) -> Option<OpMeta>,
    ) -> Result<Self> {
        let mut graph = TileGraph {
            nodes: Vec::with_capacity(dag.len() + 1),
            edges: Vec::with_capacity(dag.len()),
            order: vec![],
            pos: vec![],
        };
        let mut dag_to_tile: Vec<Option<NodeId>> = vec![None; dag.len()];

        for dag_idx in dag.topological_sort() {
            let dag_node = dag.node(dag_idx);
            let op = dag_node.value.as_ref();
            let shape = static_shape(dag_idx, op)?;
            let dtype = op.output_dtype();

            let (kind, meta) = if op.is_input() {
                (NodeKind::Placeholder, None)
            } else {
                (NodeKind::Op, op_meta(dag_idx, op))
            };
            let id = graph.add_node(kind, op.name(), shape.clone(), dtype, meta);
            for (port, &parent) in dag_node.parents.iter().enumerate() {
                let src =
                    dag_to_tile[parent].expect("a parent precedes its child in topological order");
                graph.connect(src, id, port);
            }
            dag_to_tile[dag_idx] = Some(id);

            if dag_node.children.is_empty() {
                let output = graph.add_node(NodeKind::Output, "output", shape, dtype, None);
                graph.connect(id, output, 0);
            }
        }

        graph.order = priority_order(&graph);
        graph.pos = vec![0; graph.nodes.len()];
        for (i, n) in graph.order.iter().enumerate() {
            graph.pos[n.0] = i;
        }
        Ok(graph)
    }

    fn add_node(
        &mut self,
        kind: NodeKind,
        name: &str,
        shape: Vec<u64>,
        dtype: DtypeRepr,
        meta: Option<OpMeta>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            id,
            kind,
            name: name.to_string(),
            in_edges: vec![],
            out_edges: vec![],
            shape,
            dtype,
            meta,
        });
        id
    }

    fn connect(&mut self, src: NodeId, dst: NodeId, dst_port: usize) {
        let id = EdgeId(self.edges.len());
        self.edges.push(Edge {
            id,
            src,
            dst,
            dst_port,
        });
        self.nodes[src.0].out_edges.push(id);
        self.nodes[dst.0].in_edges.push(id);
    }
}

/// `op`'s output shape with every dim known, as the scheduler requires (P1).
fn static_shape(dag_idx: usize, op: &dyn ExecutableOp) -> Result<Vec<u64>> {
    let shape = op.output_shape();
    shape
        .iter()
        .map(|dim| dim.map(|d| d as u64))
        .collect::<Option<Vec<u64>>>()
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "dag node {dag_idx} (\"{}\") has output shape {shape:?} with a dynamic dim; \
                 the tile graph needs every dim to be static",
                op.name()
            ))
            .into()
        })
}

/// A0, Welder `find_topo_sort_priority`: a DFS post-order from each `Output`
/// node that visits a node's producers deepest first, where a node's depth is
/// its longest path from any source.
///
/// Relies on `from_dag` creating nodes in topological order, so one pass over
/// the node ids computes every depth.
fn priority_order(g: &TileGraph) -> Vec<NodeId> {
    let mut depth = vec![0usize; g.nodes.len()];
    for node in &g.nodes {
        depth[node.id.0] = node
            .in_edges
            .iter()
            .map(|e| depth[g.edges[e.0].src.0] + 1)
            .max()
            .unwrap_or(0);
    }

    let producers = |n: NodeId| -> Vec<NodeId> {
        let mut ps: Vec<NodeId> = g.nodes[n.0]
            .in_edges
            .iter()
            .map(|e| g.edges[e.0].src)
            .collect();
        ps.sort_by_key(|p| (std::cmp::Reverse(depth[p.0]), p.0));
        ps.dedup();
        ps
    };

    let mut order = Vec::with_capacity(g.nodes.len());
    let mut visited = vec![false; g.nodes.len()];
    let outputs = g.nodes.iter().filter(|n| n.kind == NodeKind::Output);
    for output in outputs {
        if visited[output.id.0] {
            continue;
        }
        visited[output.id.0] = true;
        // (node, its producers, index of the next producer to visit)
        let mut stack = vec![(output.id, producers(output.id), 0usize)];
        while let Some((n, ps, next)) = stack.last_mut() {
            match ps.get(*next).copied() {
                Some(p) => {
                    *next += 1;
                    if !visited[p.0] {
                        visited[p.0] = true;
                        stack.push((p, producers(p), 0));
                    }
                }
                None => {
                    order.push(*n);
                    stack.pop();
                }
            }
        }
    }
    debug_assert_eq!(order.len(), g.nodes.len(), "every node reaches an Output");
    order
}

#[cfg(test)]
mod tests {
    use super::{NodeKind, TileGraph};
    use teeny_core::{
        graph::{DtypeRepr, Shape},
        model::ExecutableOp,
        utils::dag::Dag,
    };

    /// Just enough of an `ExecutableOp` for `from_dag`: a name, a shape, and
    /// whether it is an input.
    struct FakeOp {
        name: &'static str,
        shape: Shape,
    }

    impl ExecutableOp for FakeOp {
        fn name(&self) -> &str {
            self.name
        }
        fn is_input(&self) -> bool {
            self.name == "input"
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
            DtypeRepr::F32
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn op(name: &'static str, shape: Shape) -> Box<dyn ExecutableOp> {
        Box::new(FakeOp { name, shape })
    }

    #[test]
    fn test_from_dag_wires_a_diamond_with_ports_outputs_and_priority_order() {
        // input -> a -> c
        //      \-> b -/
        let shape: Shape = vec![Some(4), Some(8)];
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("input", shape.clone()));
        let a = dag.add_node(op("a", shape.clone()));
        let b = dag.add_node(op("b", shape.clone()));
        let c = dag.add_node(op("c", shape));
        dag.add_edge(x, a);
        dag.add_edge(x, b);
        dag.add_edge(a, c);
        dag.add_edge(b, c);

        let g = TileGraph::from_dag(&dag, |_, _| None).expect("every shape is static");

        let id_of = |name: &str| g.nodes.iter().find(|n| n.name == name).unwrap().id;
        let kind_of = |name: &str| g.nodes[id_of(name).0].kind;
        assert_eq!(
            g.nodes.len(),
            5,
            "4 DAG nodes plus one Output for the sink c"
        );
        assert_eq!(kind_of("input"), NodeKind::Placeholder);
        assert_eq!(kind_of("c"), NodeKind::Op);
        assert_eq!(kind_of("output"), NodeKind::Output);

        // c's operands follow its DAG parent order: a feeds port 0, b port 1.
        let c_in: Vec<_> = g.nodes[id_of("c").0]
            .in_edges
            .iter()
            .map(|e| (g.edges[e.0].src, g.edges[e.0].dst_port))
            .collect();
        assert_eq!(c_in, vec![(id_of("a"), 0), (id_of("b"), 1)]);
        assert_eq!(g.nodes[id_of("input").0].out_edges.len(), 2);
        assert!(g.nodes.iter().all(|n| n.shape == vec![4, 8]));

        // `order` is a topological order of every node, and `pos` inverts it.
        assert_eq!(g.order.len(), g.nodes.len());
        for (i, n) in g.order.iter().enumerate() {
            assert_eq!(g.pos[n.0], i);
        }
        for e in &g.edges {
            assert!(
                g.pos[e.src.0] < g.pos[e.dst.0],
                "edge {:?} runs backwards in `order`",
                e.id
            );
        }
    }

    #[test]
    fn test_from_dag_rejects_a_dynamic_dim() {
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        dag.add_node(op("input", vec![None, Some(8)]));

        let err = TileGraph::from_dag(&dag, |_, _| None)
            .err()
            .expect("dim 0 is dynamic");
        assert!(err.to_string().contains("dynamic dim"), "{err}");
    }
}
