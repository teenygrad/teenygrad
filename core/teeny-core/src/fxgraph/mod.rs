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

use ::z3::Solver;
use egg::{EGraph, Id};
use std::collections::HashMap;

pub mod analysis;
pub mod cat;
pub mod dtype;
pub mod keyvalue;
pub mod lang;
pub mod literal;
pub mod placeholder;
pub mod shape;
pub mod tensor;
pub mod torch;
pub mod types;
pub mod value;

use crate::fxgraph::analysis::GraphAnalysis;
use crate::fxgraph::keyvalue::{KeyValue, KeyValueList};
use crate::fxgraph::lang::FxGraphLang;
use crate::fxgraph::value::Value;

// Higher-level IR for mapping from FX graphs
#[derive(Debug)]
pub struct FXGraph {
    pub egraph: EGraph<FxGraphLang, GraphAnalysis>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
    pub example_inputs: Vec<Value>,
    pub node_map: HashMap<String, Id>, // For mapping from FX node names
    pub solver: Solver,
}

impl FXGraph {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let solver = Solver::new();

        Self {
            egraph: EGraph::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            example_inputs: Vec::new(),
            node_map: HashMap::new(),
            solver,
        }
    }

    pub fn solver(&self) -> &Solver {
        &self.solver
    }

    pub fn add_operation(&mut self, name: &str, op: FxGraphLang) -> Id {
        let id = self.egraph.add(op);
        self.node_map.insert(name.to_string(), id);
        id
    }

    pub fn add_value(&mut self, value: Value) -> Id {
        self.add_operation(&self.unique_name(), FxGraphLang::Value(value))
    }

    pub fn add_args(&mut self, args: Vec<Value>) -> Id {
        let args = args
            .into_iter()
            .map(|x| self.add_value(x))
            .collect::<Vec<_>>();
        self.add_operation(&self.unique_name(), FxGraphLang::Args(args))
    }

    pub fn add_kwargs(&mut self, kvs: Vec<KeyValue>) -> Id {
        self.add_operation(
            &self.unique_name(),
            FxGraphLang::KwArgs(KeyValueList::new(kvs)),
        )
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
