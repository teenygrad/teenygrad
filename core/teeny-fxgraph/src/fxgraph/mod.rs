/*
 * Copyright (c) 2026 Teenygrad.
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

use egg::{EGraph, Id};
use std::collections::{HashMap, VecDeque};

pub mod analysis;
pub mod cat;
pub mod device;
pub mod dtype;
pub mod keyvalue;
pub mod lang;
pub mod literal;
pub mod node;
pub mod placeholder;
pub mod shape;
pub mod tensor;
pub mod torch;
pub mod types;
pub mod value;

use crate::errors::Error;
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
    pub node_map: HashMap<String, Id>, // For mapping from FX node names
}

impl FXGraph {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            egraph: EGraph::new(GraphAnalysis::new()?),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_map: HashMap::new(),
        })
    }

    pub fn verify_types(&mut self) -> Result<(), Error> {
        Ok(())
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

    pub fn depends_on(&self, id: &Id) -> Vec<Id> {
        let node = self.egraph.id_to_node(*id);

        match node {
            FxGraphLang::Value(v) => v.depends_on(),
            FxGraphLang::Placeholder(_) => vec![],
            FxGraphLang::Constant(_) => vec![],
            FxGraphLang::KwArgs(_) => vec![],
            FxGraphLang::Output(args) => vec![args[0]],
            FxGraphLang::Add(add) => vec![add[0], add[1]],
            _ => todo!("Implement depends_on: {:?}", node),
        }
    }

    pub fn topological_sort(&self) -> Result<Vec<Id>, Error> {
        // Count incoming edges for each node
        let mut in_degree: HashMap<Id, usize> = HashMap::new();

        // Initialize in-degree count
        for node_id in self.node_map.values() {
            in_degree.insert(*node_id, 0);
        }

        // Calculate in-degrees
        for id in self.node_map.values() {
            println!("Node: id={}, node={:?}", id, self.egraph.id_to_node(*id));
            let dependencies = self.depends_on(id);
            *in_degree.get_mut(id).unwrap() += dependencies.len();
        }

        println!("-----");

        // Find all nodes with no incoming edges (no dependencies)
        let mut queue = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(node_id, _)| *node_id)
            .collect::<VecDeque<_>>();

        queue.iter().for_each(|id| {
            println!(
                "Queue Node: id={}, node={:?}",
                id,
                self.egraph.id_to_node(*id)
            );
        });

        let mut result: Vec<Id> = Vec::new();
        while let Some(current) = queue.pop_front() {
            result.push(current);

            // For each node that depends on the current node
            for id in self.node_map.values() {
                let dependencies = self.depends_on(id);
                if dependencies.contains(&current) {
                    // Reduce in-degree
                    let degree = in_degree.get_mut(id).unwrap();
                    *degree -= 1;

                    // If no more dependencies, add to queue
                    if *degree == 0 {
                        queue.push_back(*id);
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != self.node_map.len() {
            println!("-----");
            self.node_map.iter().for_each(|(name, id)| {
                println!(
                    "Node: name={}, id={}, node={:?}",
                    name,
                    id,
                    self.egraph.id_to_node(*id)
                );
            });
            println!("-----");
            result.iter().for_each(|id| {
                println!(
                    "Result Node: id={}, node={:?}",
                    id,
                    self.egraph.id_to_node(*id)
                );
            });
            println!("-----");

            return Err(Error::InvalidGraph(
                "Graph contains a cycle - not a valid DAG".to_string(),
            ));
        }

        Ok(result)
    }
}
