/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use egg::{EGraph, Id};
use std::collections::HashMap;

pub mod analysis;
pub mod dtype;
pub mod lang;
pub mod literal;
pub mod shape;

use crate::fxgraph::analysis::TensorAnalysis;
use crate::fxgraph::dtype::DtypeValue;
use crate::fxgraph::lang::FxGraphLang;
use crate::fxgraph::shape::ShapeValue;

// Higher-level IR for mapping from FX graphs
#[derive(Debug, Clone, Default)]
pub struct FxGraph {
    pub egraph: EGraph<FxGraphLang, TensorAnalysis>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
    pub node_map: HashMap<String, Id>, // For mapping from FX node names
}

impl FxGraph {
    pub fn new() -> Self {
        Self {
            egraph: EGraph::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_tensor(&mut self, name: String, shape: ShapeValue, dtype: DtypeValue) -> Id {
        let shape_id = self.egraph.add(FxGraphLang::Shape(shape));
        let dtype_id = self.egraph.add(FxGraphLang::Dtype(dtype));
        let tensor_id = self.egraph.add(FxGraphLang::Tensor([shape_id, dtype_id]));
        self.node_map.insert(name, tensor_id);
        tensor_id
    }

    pub fn add_operation(&mut self, name: String, op: FxGraphLang) -> Id {
        let id = self.egraph.add(op);
        self.node_map.insert(name, id);
        id
    }

    pub fn get_node(&self, name: &str) -> Option<Id> {
        self.node_map.get(name).copied()
    }
}
