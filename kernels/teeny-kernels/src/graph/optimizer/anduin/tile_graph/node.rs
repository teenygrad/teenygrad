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

use teeny_core::{
    device::hardware::MemoryLevel,
    graph::{DtypeRepr, Shape},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub id: EdgeId,
    pub src_node_id: NodeId,
    pub src_edge_id: EdgeId,
    pub dst_node_id: NodeId,
    pub dst_edge_id: EdgeId,
    pub memory_level: Option<MemoryLevel>,
}

impl Edge {
    pub fn set_connect(&mut self, memory_level: MemoryLevel) {
        self.memory_level = Some(memory_level);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum NodeKind {
    Placeholder,
    Output,
    IRNode,
}

#[derive(Debug, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub dag_node_idx: usize,
    pub kind: NodeKind,
    pub name: String,
    pub in_edges: Vec<Edge>,
    pub out_edges: Vec<Edge>,
    pub shapes: Vec<Shape>,
    pub dtypes: Vec<DtypeRepr>,
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
