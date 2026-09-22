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
    graph::{DtypeRepr, Shape},
    model::ExecutableOp,
    utils::dag::Dag,
};

use crate::errors::{Error, Result};

use self::node::{Edge, Node, NodeKind};

pub use node::{EdgeId, NodeId};

pub mod node;

#[derive(Debug, Default)]
pub struct TileGraph {
    nodes: Vec<Node>,
}

impl TileGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert an already-lowered `Dag` (e.g. from [`TritonLowering`]) into a
    /// `TileGraph`, walking the DAG in topological order.
    pub fn from_dag(dag: &Dag<Box<dyn ExecutableOp>>) -> Self {
        let mut tile_graph = TileGraph::new();
        let n = dag.len();
        let mut dag_to_tile: Vec<Option<NodeId>> = (0..n).map(|_| None).collect();

        for dag_idx in dag.topological_sort() {
            let dag_node = dag.node(dag_idx);
            let op = &dag_node.value;

            let tile_id = if op.is_input() {
                tile_graph.add_placeholder(op.name())
            } else {
                let inputs: Vec<(NodeId, EdgeId)> = dag_node
                    .parents
                    .iter()
                    .map(|&parent| {
                        (
                            dag_to_tile[parent]
                                .expect("parent must appear earlier in topological order"),
                            EdgeId(0),
                        )
                    })
                    .collect();
                tile_graph
                    .add_ir_node(op.name(), &inputs)
                    .unwrap_or_else(|e| panic!("failed to add dag node {dag_idx}: {e}"))
            };

            let shape = op.output_shape().clone();
            let dtype = op.output_dtype();
            tile_graph.set_port_metadata(tile_id, vec![shape.clone()], vec![dtype]);
            dag_to_tile[dag_idx] = Some(tile_id);

            if dag_node.children.is_empty() {
                let output_id = tile_graph
                    .add_output((tile_id, EdgeId(0)))
                    .unwrap_or_else(|e| panic!("failed to add output for dag node {dag_idx}: {e}"));
                tile_graph.set_port_metadata(output_id, vec![shape], vec![dtype]);
            }
        }

        tile_graph
    }

    pub fn node(&self, node_id: NodeId) -> Result<&Node> {
        self.nodes
            .get(node_id.0)
            .ok_or_else(|| Error::NodeNotFound(node_id.0).into())
    }

    pub fn node_mut(&mut self, node_id: NodeId) -> Result<&mut Node> {
        self.nodes
            .get_mut(node_id.0)
            .ok_or_else(|| Error::NodeNotFound(node_id.0).into())
    }

    pub fn add_placeholder(&mut self, name: &str) -> NodeId {
        let node_id = NodeId(self.nodes.len());

        let node = Node {
            id: node_id,
            kind: NodeKind::Placeholder,
            name: name.to_string(),
            in_edges: vec![],
            out_edges: vec![],
            shapes: vec![],
            dtypes: vec![],
        };

        self.nodes.push(node);
        node_id
    }

    pub fn add_output(&mut self, producer: (NodeId, EdgeId)) -> Result<NodeId> {
        let node_id = NodeId(self.nodes.len());

        let node = Node {
            id: node_id,
            kind: NodeKind::Output,
            name: "output".to_string(),
            in_edges: Vec::with_capacity(1),
            out_edges: vec![],
            shapes: vec![],
            dtypes: vec![],
        };

        self.nodes.push(node);
        self.connect_inputs(node_id, &[producer])?;
        Ok(node_id)
    }

    pub fn add_ir_node(&mut self, name: &str, inputs: &[(NodeId, EdgeId)]) -> Result<NodeId> {
        let dst_node_id = NodeId(self.nodes.len());

        let node = Node {
            id: dst_node_id,
            kind: NodeKind::IRNode,
            name: name.to_string(),
            in_edges: Vec::with_capacity(inputs.len()),
            out_edges: vec![],
            shapes: vec![],
            dtypes: vec![],
        };

        self.nodes.push(node);
        self.connect_inputs(dst_node_id, inputs)?;
        Ok(dst_node_id)
    }

    fn set_port_metadata(&mut self, node_id: NodeId, shapes: Vec<Shape>, dtypes: Vec<DtypeRepr>) {
        if let Ok(node) = self.node_mut(node_id) {
            node.shapes = shapes;
            node.dtypes = dtypes;
        }
    }

    /// Wire `dst` to its producers and back-patch each producer's `out_edges`.
    fn connect_inputs(&mut self, dst: NodeId, inputs: &[(NodeId, EdgeId)]) -> Result<()> {
        for (dst_edge_id, (src_node_id, src_edge_id)) in inputs.iter().enumerate() {
            let edge = Edge {
                src_node_id: *src_node_id,
                src_edge_id: *src_edge_id,
                dst_node_id: dst,
                dst_edge_id: EdgeId(dst_edge_id),
            };

            self.node_mut(dst)?.in_edges.push(edge);
            self.node_mut(*src_node_id)?.out_edges_mut().push(edge);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::{
        graph::{DtypeRepr, Graph, Shape, SymTensor},
        model::LoweringMode,
        nn::{Layer, activation::sigmoid::Silu, batchnorm::BatchNorm2d, conv2d::Conv2d},
        sequential,
    };

    use crate::graph::TritonLowering;

    use super::TileGraph;
    use super::node::{Edge, EdgeId, NodeId, NodeKind};

    /// Traces `Conv2d(3->8, 3x3, pad=1) -> BatchNorm2d(8) -> SiLU` over a
    /// `[10, 3, 32, 32]` input.
    ///
    /// Built from real `nn` layers and recorded through [`SymTensor`] rather
    /// than from hand-written `Op`s, so the traced graph -- including each
    /// node's inferred output shape -- is the one a caller actually gets
    /// from a model. A change in how these layers record themselves then
    /// shows up here, instead of being papered over by hardwired shapes.
    ///
    /// Same-padding keeps the 32x32 spatial dims, so only the channel count
    /// changes (3 -> 8). The returned shapes are the input shape and the
    /// output shape shared by all three ops.
    fn conv2d_bn_silu_graph() -> (Graph, Shape, Shape) {
        let in_shape: Shape = vec![Some(10), Some(3), Some(32), Some(32)];
        let out_shape: Shape = vec![Some(10), Some(8), Some(32), Some(32)];

        let (input, graph) = SymTensor::input(DtypeRepr::F32, in_shape.clone());
        let model = sequential![
            Conv2d::<f32, _, _, 4>::new(3, 8, (3, 3), (1, 1), (1, 1), false),
            BatchNorm2d::<f32, _, _, 4>::new(8),
            Silu::<f32, _, 4>::new()
        ];
        let _output = Layer::call(&model, input);

        let graph = graph.borrow().clone();
        (graph, in_shape, out_shape)
    }

    /// The single edge from `src`'s first output port into `dst`'s first
    /// input port — every node in this graph is single-in/single-out, so
    /// both `EdgeId`s are always 0.
    fn edge(src: usize, dst: usize) -> Edge {
        Edge {
            src_node_id: NodeId(src),
            src_edge_id: EdgeId(0),
            dst_node_id: NodeId(dst),
            dst_edge_id: EdgeId(0),
        }
    }

    #[test]
    fn test_conv2d_batchnorm_silu_to_tile_graph() {
        let (graph, in_shape, out_shape) = conv2d_bn_silu_graph();
        assert_eq!(
            graph.nodes.len(),
            4,
            "tracing conv2d+bn+silu should record 1 Input plus 3 ops"
        );

        let lowering = TritonLowering::default();
        let (dag, _, _) = lowering
            .lower_with_mapping(&graph, LoweringMode::Inference)
            .expect("triton lowering should succeed");
        let tile_graph = TileGraph::from_dag(&dag);

        // Lowered `input` becomes a `Placeholder`; every kernel becomes an
        // `IRNode`; the DAG sink gets an `Output`.
        let kinds: Vec<&NodeKind> = tile_graph.nodes.iter().map(|node| &node.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &NodeKind::Placeholder,
                &NodeKind::IRNode,
                &NodeKind::IRNode,
                &NodeKind::IRNode,
                &NodeKind::Output,
            ]
        );

        // Every tile node but the trailing `Output` takes its name straight
        // from its lowered kernel's `ExecutableOp::name()` -- compared
        // against the DAG itself rather than against literals, so renaming
        // a kernel can't quietly stop being checked here. Today those four
        // names are `input`, `conv2d_forward`,
        // `batch_norm_2d_nchw_forward_inference` and `silu_forward`.
        let names: Vec<&str> = tile_graph.nodes.iter().map(|node| node.name()).collect();
        let dag_names: Vec<&str> = (0..dag.len()).map(|i| dag.node(i).value.name()).collect();
        assert_eq!(names[..dag.len()], dag_names[..]);
        assert_eq!(names[0], "input", "the lowered `Op::Input` node");
        assert_eq!(names[names.len() - 1], "output");

        // This chain is lowered in producer-before-consumer order, so tile
        // `NodeId`s line up with DAG indices (plus the trailing `Output`).
        for (id, expected_in, expected_out) in [
            (0usize, vec![], vec![edge(0, 1)]),
            (1, vec![edge(0, 1)], vec![edge(1, 2)]),
            (2, vec![edge(1, 2)], vec![edge(2, 3)]),
            (3, vec![edge(2, 3)], vec![edge(3, 4)]),
            (4, vec![edge(3, 4)], vec![]),
        ] {
            let node = tile_graph
                .node(NodeId(id))
                .unwrap_or_else(|e| panic!("node {id} should exist: {e}"));
            assert_eq!(node.in_edges(), &expected_in, "in_edges of node {id}");
            assert_eq!(node.out_edges(), &expected_out, "out_edges of node {id}");
        }

        // One entry per output port. Conv2d widens 3 -> 8 channels, and
        // batchnorm2d/silu are both shape-preserving, so only the
        // placeholder carries the 3-channel input shape.
        let shapes: Vec<&Vec<Shape>> = tile_graph.nodes.iter().map(|node| node.shapes()).collect();
        assert_eq!(
            shapes,
            vec![
                // The `Output` node mirrors its producer's shape.
                &vec![in_shape],
                &vec![out_shape.clone()],
                &vec![out_shape.clone()],
                &vec![out_shape.clone()],
                &vec![out_shape],
            ]
        );

        // Likewise one dtype per output port; nothing in this graph
        // changes dtype.
        for node in tile_graph.nodes.iter() {
            assert_eq!(
                node.dtypes(),
                &vec![DtypeRepr::F32],
                "dtypes of node {} (\"{}\")",
                node.id().0,
                node.name()
            );
        }
    }
}
