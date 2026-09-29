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

use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::KernelTileSpec;

// `Ord` so a tile config keyed by these can be iterated deterministically
// rather than in `HashMap` order (teenygrad-1nr.25).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

// `Ord` so a tile config keyed by these can be iterated deterministically
// rather than in `HashMap` order (teenygrad-1nr.25).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub src_node_id: NodeId,
    pub src_edge_id: EdgeId,
    pub dst_node_id: NodeId,
    pub dst_edge_id: EdgeId,
}

#[derive(Debug, PartialEq, Eq)]
pub enum NodeKind {
    Placeholder,
    Output,
    IRNode,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub name: String,
    pub in_edges: Vec<Edge>,
    pub out_edges: Vec<Edge>,
    pub shapes: Vec<Shape>,
    pub dtypes: Vec<DtypeRepr>,
    /// The tile-shape metadata this node's kernel declares, when it declares
    /// any (teenygrad-1nr.25).
    ///
    /// `None` means the op does not describe its axes, and propagation treats
    /// it as a hard boundary rather than guessing -- the same opt-in coverage
    /// `KernelTileSpec` has always had. Note the reverse is not an excuse:
    /// `teenygrad-39jd` wired `TritonLowering` to pass the derived spec through
    /// for every flat elementwise activation, so a `None` here now means the op
    /// genuinely has nothing to say rather than that the lowering dropped it.
    pub tile_spec: Option<KernelTileSpec>,
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

    pub fn tile_spec(&self) -> Option<&KernelTileSpec> {
        self.tile_spec.as_ref()
    }
}
