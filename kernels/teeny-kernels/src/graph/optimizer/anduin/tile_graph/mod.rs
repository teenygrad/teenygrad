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

use std::hash::Hash;

use crate::errors::{Error, Result};
use teeny_core::graph::{DtypeRepr, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge {
    src_node_id: NodeId,
    src_edge_id: EdgeId,
    dst_node_id: NodeId,
    dst_edge_id: EdgeId,
}

#[derive(Debug, PartialEq, Eq)]
enum NodeKind {
    Placeholder,
    Output,
    IRNode,
}

#[derive(Debug, PartialEq, Eq)]
struct Node {
    id: NodeId,
    kind: NodeKind,
    name: String,
    in_edges: Vec<Edge>,
    out_edges: Vec<Edge>,
    shapes: Vec<Shape>,
    dtypes: Vec<DtypeRepr>,
}

impl Node {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn in_edges(&self) -> &Vec<Edge> {
        &self.in_edges
    }
    pub fn out_edges(&self) -> &Vec<Edge> {
        &self.out_edges
    }
    pub fn out_edges_mut(&mut self) -> &mut Vec<Edge> {
        &mut self.out_edges
    }

    pub fn shapes(&self) -> &Vec<Shape> {
        &self.shapes
    }
    pub fn dtypes(&self) -> &Vec<DtypeRepr> {
        &self.dtypes
    }
}

pub struct TileGraph {
    nodes: Vec<Node>,
}

impl TileGraph {
    pub fn new() -> Self {
        Self { nodes: vec![] }
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
}
