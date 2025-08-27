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
pub mod cat;
pub mod dtype;
pub mod item;
pub mod keyvalue;
pub mod lang;
pub mod literal;
pub mod placeholder;
pub mod shape;
pub mod value;

use crate::fxgraph::analysis::TensorAnalysis;
use crate::fxgraph::lang::FxGraphLang;

// Higher-level IR for mapping from FX graphs
#[derive(Debug, Clone, Default)]
pub struct FXGraph {
    pub egraph: EGraph<FxGraphLang, TensorAnalysis>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
    pub node_map: HashMap<String, Id>, // For mapping from FX node names
}

impl FXGraph {
    pub fn new() -> Self {
        Self {
            egraph: EGraph::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_operation(&mut self, name: &str, op: FxGraphLang) -> Id {
        let id = self.egraph.add(op);
        self.node_map.insert(name.to_string(), id);
        id
    }

    pub fn get_node(&self, name: &str) -> Option<Id> {
        self.node_map.get(name).copied()
    }

    pub fn unique_name(&self) -> String {
        let mut i = self.node_map.len();
        loop {
            let name = format!("#{i}");
            i += 1;
            if !self.node_map.contains_key(&name) {
                return name;
            }
        }
    }
}
