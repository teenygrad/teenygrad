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

pub struct TileGraph {
    nodes: Vec<Node>,
}

impl TileGraph {
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }

    /// Convert an already-lowered `Dag` (e.g. from [`TritonLowering`]) into a
    /// `TileGraph`, preserving DAG node indices and wiring a synthetic
    /// [`NodeKind::Output`] on each sink.
    pub fn from_dag(dag: &Dag<Box<dyn ExecutableOp>>) -> Self {
        let mut tile_graph = TileGraph::new();
        let n = dag.len();

        for i in 0..n {
            let dag_node = dag.node(i);
            let op = &dag_node.value;
            let shapes = vec![op.output_shape().clone()];
            let dtypes = vec![op.output_dtype()];

            let node_id = if op.is_input() {
                tile_graph.add_placeholder(op.name())
            } else {
                let inputs: Vec<(NodeId, EdgeId)> = dag_node
                    .parents
                    .iter()
                    .map(|&parent| (NodeId(parent), EdgeId(0)))
                    .collect();
                tile_graph
                    .add_ir_node(op.name(), &inputs)
                    .unwrap_or_else(|e| panic!("failed to wire dag node {i}: {e}"))
            };

            tile_graph.set_output_metadata(node_id, shapes, dtypes);
        }

        let sinks: Vec<usize> = (0..n)
            .filter(|i| dag.node(*i).children.is_empty())
            .collect();
        assert!(
            sinks.len() == 1,
            "from_dag expects a single sink, found {}: {:?}",
            sinks.len(),
            sinks
        );
        let sink = sinks[0];
        let sink_shape = dag.node(sink).value.output_shape().clone();
        let sink_dtype = dag.node(sink).value.output_dtype();
        let output_id = tile_graph
            .add_output((NodeId(sink), EdgeId(0)))
            .unwrap_or_else(|e| panic!("failed to add output for sink {sink}: {e}"));
        tile_graph.set_output_metadata(output_id, vec![sink_shape], vec![sink_dtype]);

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

        let mut node = Node {
            id: node_id,
            kind: NodeKind::Output,
            name: "output".to_string(),
            in_edges: Vec::with_capacity(1),
            out_edges: vec![],
            shapes: vec![],
            dtypes: vec![],
        };

        self.init_node(&mut node, &[producer])?;

        self.nodes.push(node);
        Ok(node_id)
    }

    pub fn add_ir_node(&mut self, name: &str, inputs: &[(NodeId, EdgeId)]) -> Result<NodeId> {
        let dst_node_id = NodeId(self.nodes.len());

        let mut node = Node {
            id: dst_node_id,
            kind: NodeKind::IRNode,
            name: name.to_string(),
            in_edges: Vec::with_capacity(inputs.len()),
            out_edges: vec![],
            shapes: vec![],
            dtypes: vec![],
        };

        self.init_node(&mut node, inputs)?;

        self.nodes.push(node);
        Ok(dst_node_id)
    }

    fn init_node(&mut self, node: &mut Node, inputs: &[(NodeId, EdgeId)]) -> Result<()> {
        let dst_node_id = node.id();

        for (dst_edge_id, (src_node_id, src_edge_id)) in inputs.iter().enumerate() {
            let edge = Edge {
                src_node_id: *src_node_id,
                src_edge_id: *src_edge_id,
                dst_node_id,
                dst_edge_id: EdgeId(dst_edge_id),
            };

            node.in_edges.push(edge);
            self.node_mut(*src_node_id)?.out_edges_mut().push(edge);
        }

        Ok(())
    }

    fn set_output_metadata(&mut self, node_id: NodeId, shapes: Vec<Shape>, dtypes: Vec<DtypeRepr>) {
        if let Ok(node) = self.node_mut(node_id) {
            node.shapes = shapes;
            node.dtypes = dtypes;
        }
    }
}

#[cfg(test)]
mod tests {
    use teeny_core::{
        graph::{DtypeRepr, Graph, Op, Shape},
        model::LoweringMode,
    };

    use crate::graph::TritonLowering;

    use super::TileGraph;
    use super::node::{Edge, EdgeId, NodeId, NodeKind};

    ///
    /// A simple `input -> conv2d -> batchnorm2d -> silu` graph.
    ///
    fn conv2d_bn_silu_graph() -> (Graph, Shape, Shape) {
        let in_shape: Shape = vec![Some(10), Some(3), Some(32), Some(32)];
        let out_shape: Shape = vec![Some(10), Some(8), Some(32), Some(32)];

        let mut graph = Graph::new();
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, in_shape.clone());
        let conv = graph.add_node(
            Op::Conv2d {
                in_channels: 3,
                out_channels: 8,
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                padding_h: 1,
                padding_w: 1,
                groups: 1,
                has_bias: false,
            },
            vec![input],
            DtypeRepr::F32,
            out_shape.clone(),
        );
        let bn = graph.add_node(
            Op::BatchNorm2d {
                num_features: 8,
                eps: 1e-5,
                momentum: 0.1,
                affine: true,
                track_running_stats: true,
            },
            vec![conv],
            DtypeRepr::F32,
            out_shape.clone(),
        );
        graph.add_node(Op::Silu, vec![bn], DtypeRepr::F32, out_shape.clone());

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
    fn conv2d_batchnorm_silu_to_tile_graph() {
        let (graph, in_shape, out_shape) = conv2d_bn_silu_graph();
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

        // Names come from each lowered kernel's `ExecutableOp::name()`.
        let names: Vec<&str> = tile_graph.nodes.iter().map(|node| node.name()).collect();
        assert_eq!(names[0], "input");
        assert_eq!(names[names.len() - 1], "output");
        assert!(names[1].contains("conv2d"));
        assert!(names[2].contains("batch_norm"));
        assert!(names[3].contains("silu"));

        // Node order matches the lowered DAG indices; `NodeId(i)` is dag node `i`.
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
