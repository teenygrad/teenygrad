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
use crate::graph::tile_resolve::{ConstLookup, NoConsts};

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

/// A tileable op: one whose kernel declares both a tile spec and a grid spec.
/// Its spec, grid and rider flag are read off the op itself (the rider flag
/// derived from the spec); only what the op can't say about itself is stored
/// beside it.
pub struct OpMeta {
    op: Box<dyn ExecutableOp>,        // the lowered op this node was built from
    pub consts: Box<dyn ConstLookup>, // values the op can't give itself (P3); see `const_value`
    pub flops_per_output: f64,        // includes the op's own reduction work per output element
}

impl OpMeta {
    /// `None` if `op`'s kernel lacks a tile spec or a grid spec, i.e. the node
    /// is opaque.
    pub fn new(
        op: Box<dyn ExecutableOp>,
        consts: Box<dyn ConstLookup>,
        flops_per_output: f64,
    ) -> Option<Self> {
        if op.tile_spec().is_none() || op.grid_spec().is_none() {
            return None;
        }
        Some(Self {
            op,
            consts,
            flops_per_output,
        })
    }

    /// The [`TileGraph::from_dag`] callback for a DAG `TritonLowering`
    /// produced: every lowered kernel gives its own specs and consts, so the
    /// op is all it needs. `None` (opaque) for an op without both specs. The
    /// flop count is a placeholder of 1 per output element until ops report
    /// their own.
    pub fn lowered(_dag_idx: usize, op: Box<dyn ExecutableOp>) -> Option<Self> {
        Self::new(op, Box::new(NoConsts), 1.0)
    }

    pub fn op(&self) -> &dyn ExecutableOp {
        self.op.as_ref()
    }

    pub fn spec(&self) -> KernelTileSpec {
        self.op
            .tile_spec()
            .expect("OpMeta::new only accepts an op with a tile spec")
    }

    /// The value of the const `name` a spec mentions (P3): the op's own,
    /// from its kernel's const generics, else `consts`. The op answers for
    /// every lowered kernel; `consts` is for an op that can't (a hand-built
    /// one in a test).
    pub fn const_value(&self, name: &str) -> Option<usize> {
        self.op.const_value(name).or_else(|| self.consts.get(name))
    }

    pub fn grid(&self) -> GridSpec {
        self.op
            .grid_spec()
            .expect("OpMeta::new only accepts an op with a grid spec")
    }

    /// Whether this op can run on another program's decoded coordinates:
    /// its spec is a broadcast map ([`KernelTileSpec::is_broadcast_map`]),
    /// derived from the kernel, so elementwise ops, BatchNorm2d inference and
    /// the row-broadcast Sub/Div are riders and conv2d and the reductions are
    /// not (teenygrad-8x1t).
    ///
    /// [`KernelTileSpec::is_broadcast_map`]: teeny_core::model::KernelTileSpec::is_broadcast_map
    pub fn rider(&self) -> bool {
        self.spec().is_broadcast_map()
    }
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
    /// Takes ownership of `dag`: each `Op` node's op is handed to
    /// `op_meta(dag_idx, op)`, which moves it into the node's [`OpMeta`];
    /// returning `None` makes the node opaque (never fused) and drops the op, as
    /// does every `Placeholder`. The op supplies its own tile spec, grid spec
    /// and rider flag, so only the consts and flop count come from the caller.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if an op's output shape has a dynamic
    /// dim, since the scheduler only handles static shapes (P1).
    ///
    /// [`TritonLowering`]: crate::graph::TritonLowering
    pub fn from_dag(
        dag: Dag<Box<dyn ExecutableOp>>,
        mut op_meta: impl FnMut(usize, Box<dyn ExecutableOp>) -> Option<OpMeta>,
    ) -> Result<Self> {
        let mut graph = TileGraph {
            nodes: Vec::with_capacity(dag.len() + 1),
            edges: Vec::with_capacity(dag.len()),
            order: vec![],
            pos: vec![],
        };
        let mut dag_to_tile: Vec<Option<NodeId>> = vec![None; dag.len()];
        let order = dag.topological_sort();
        let mut dag_nodes: Vec<_> = dag.into_iter().map(Some).collect();

        for dag_idx in order {
            let dag_node = dag_nodes[dag_idx]
                .take()
                .expect("a topological order visits each node once");
            let op = dag_node.value;
            let shape = static_shape(dag_idx, op.as_ref())?;
            let dtype = op.output_dtype();
            let name = op.name().to_string();

            // A parameter operand (conv2d's weight) is not a DAG edge, so it
            // gets a placeholder of its own on its port (P5): then it is read,
            // staged and costed like any external input. Created before the
            // op, as `priority_order` needs producers first.
            if let Some(p) = op
                .parameters()
                .iter()
                .find(|p| p.port < dag_node.parents.len())
            {
                return Err(Error::InvalidArgument(format!(
                    "dag node {dag_idx} (\"{name}\") declares a parameter operand on port {}, \
                     which its DAG input {} already occupies; parameters follow the edge inputs",
                    p.port, p.port
                ))
                .into());
            }
            let params: Vec<(usize, NodeId)> = op
                .parameters()
                .iter()
                .map(|p| {
                    let shape = p.shape.iter().map(|&d| d as u64).collect();
                    let pname = format!("{name}.param{}", p.port);
                    let src = graph.add_node(NodeKind::Placeholder, pname, shape, dtype, None);
                    (p.port, src)
                })
                .collect();
            let (kind, meta) = if op.is_input() {
                (NodeKind::Placeholder, None)
            } else {
                (NodeKind::Op, op_meta(dag_idx, op))
            };
            let id = graph.add_node(kind, name, shape.clone(), dtype, meta);
            for (port, &parent) in dag_node.parents.iter().enumerate() {
                let src =
                    dag_to_tile[parent].expect("a parent precedes its child in topological order");
                graph.connect(src, id, port);
            }
            for (port, src) in params {
                graph.connect(src, id, port);
            }
            dag_to_tile[dag_idx] = Some(id);

            if dag_node.children.is_empty() {
                let output =
                    graph.add_node(NodeKind::Output, "output".to_string(), shape, dtype, None);
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
        name: String,
        shape: Vec<u64>,
        dtype: DtypeRepr,
        meta: Option<OpMeta>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            id,
            kind,
            name,
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
        model::{ExecutableOp, ParameterOperand},
        utils::dag::Dag,
    };

    /// Just enough of an `ExecutableOp` for `from_dag`: a name, a shape,
    /// whether it is an input, and its parameter operands.
    struct FakeOp {
        name: &'static str,
        shape: Shape,
        params: Vec<ParameterOperand>,
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
        fn parameters(&self) -> &[ParameterOperand] {
            &self.params
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn op(name: &'static str, shape: Shape) -> Box<dyn ExecutableOp> {
        with_params(name, shape, vec![])
    }

    fn with_params(
        name: &'static str,
        shape: Shape,
        params: Vec<ParameterOperand>,
    ) -> Box<dyn ExecutableOp> {
        Box::new(FakeOp {
            name,
            shape,
            params,
        })
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

        let g = TileGraph::from_dag(dag, |_, _| None).expect("every shape is static");

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

        let err = TileGraph::from_dag(dag, |_, _| None)
            .err()
            .expect("dim 0 is dynamic");
        assert!(err.to_string().contains("dynamic dim"), "{err}");
    }

    #[test]
    fn test_from_dag_gives_a_parameter_operand_a_placeholder_on_its_port() {
        // conv2d's shape of things: x is the DAG edge on port 0, and the
        // weight, which the op holds, is read on port 1 (P5).
        let weight = ParameterOperand {
            port: 1,
            shape: vec![8, 3, 3, 3],
        };
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("input", vec![Some(2), Some(3), Some(32), Some(32)]));
        let conv = dag.add_node(with_params(
            "conv",
            vec![Some(2), Some(8), Some(32), Some(32)],
            vec![weight.clone()],
        ));
        dag.add_edge(x, conv);

        let g = TileGraph::from_dag(dag, |_, _| None).expect("every shape is static");
        let id_of = |name: &str| g.nodes.iter().find(|n| n.name == name).unwrap().id;
        let w = id_of("conv.param1");
        assert_eq!(g.nodes[w.0].kind, NodeKind::Placeholder);
        assert_eq!(g.nodes[w.0].shape, vec![8, 3, 3, 3]);
        let conv_in: Vec<_> = g.nodes[id_of("conv").0]
            .in_edges
            .iter()
            .map(|e| (g.edges[e.0].src, g.edges[e.0].dst_port))
            .collect();
        assert_eq!(
            conv_in,
            vec![(id_of("input"), 0), (w, 1)],
            "the DAG edge first, then the parameter on its own port"
        );
        assert!(
            g.pos[w.0] < g.pos[id_of("conv").0],
            "a producer precedes its consumer"
        );

        // A parameter can't take a port a DAG edge already has.
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        let x = dag.add_node(op("input", vec![Some(4)]));
        let bad = dag.add_node(with_params(
            "bad",
            vec![Some(4)],
            vec![ParameterOperand { port: 0, ..weight }],
        ));
        dag.add_edge(x, bad);
        let err = TileGraph::from_dag(dag, |_, _| None)
            .err()
            .expect("port 0 is x's");
        assert!(err.to_string().contains("port 0"), "{err}");
    }
}
