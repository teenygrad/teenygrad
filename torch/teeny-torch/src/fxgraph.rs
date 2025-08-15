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

use crate::{error::Error, graph::Graph};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::OnceLock;

use regex::Regex;
use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang, literal::ConstantValue};

use crate::graph::{Node, OpType};

impl<'a> TryFrom<Graph<'a>> for FXGraph {
    type Error = Error;

    fn try_from(graph: Graph<'a>) -> Result<Self, Self::Error> {
        let mut fxgraph = FXGraph::new();

        // Safely access nodes with error handling
        let nodes = graph.nodes().ok_or(Error::NoGraphNodes)?;

        for node in nodes {
            let _name = node.name().ok_or(Error::NoGraphNodeName)?;
            let op = node.op();

            match op {
                OpType::placeholder => {
                    handle_placeholder(&mut fxgraph, &node)?;
                }
                OpType::call_function => {
                    call_function(&mut fxgraph, &node)?;
                }
                OpType::call_method => {
                    call_method(&mut fxgraph, &node)?;
                }
                _ => {
                    println!("Unknown op: {node:?}");
                    // return Err(Error::UnsupportedOp(op));
                }
            }
        }

        Ok(fxgraph)
    }
}

fn handle_placeholder(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node.target().ok_or(Error::NoGraphNodeTarget)?;

    fxgraph.add_operation(
        &format!("p#{target}"),
        FxGraphLang::Placeholder(target.to_string()),
    );

    Ok(())
}

// Static HashMap mapping function names to their handler functions
type ConvertFn = fn(&mut FXGraph, &Node) -> Result<(), Error>;
static FUNCS: OnceLock<HashMap<String, ConvertFn>> = OnceLock::new();

// Initialize the functions HashMap
fn get_functions() -> &'static HashMap<String, ConvertFn> {
    FUNCS.get_or_init(|| {
        HashMap::from([
            ("embedding".to_string(), function_embedding as ConvertFn),
            // ("arange", function_arange),
        ])
    })
}

fn call_function(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node.target().ok_or(Error::NoGraphNodeTarget)?;
    let op = node.op();
    let args = node.args();
    let pat =
        r#"^(?:<function\s+|<built-in\s+(?:method|function)\s+|aten\.)([a-zA-Z_][a-zA-Z0-9_]*)"#;
    let re = Regex::new(pat).unwrap();
    let func_name = re
        .captures(target)
        .ok_or(Error::NoGraphNodeTarget)?
        .get(1)
        .ok_or(Error::NoGraphNodeTarget)?
        .as_str();
    println!("Func name: {func_name:?}");

    let functions = get_functions();
    if let Some(handler) = functions.get(func_name) {
        handler(fxgraph, node)?;
    } else {
        println!("Unknown function: {func_name:?}");
    }

    Ok(())
}

fn function_getitem(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_lazy_load_decompositions(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_iadd(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_arange(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_embedding(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_add_batch_dim(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_remove_batch_dim(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_vmap_decrement_nesting(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_enter_autocast(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_matmul(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn function_vmap_increment_nesting(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!()
}

fn call_method(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let _target = node.target().ok_or(Error::NoGraphNodeTarget)?;
    let _op = node.op();
    let _args = node.args();

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::graph::deserialize_graph;

    use super::*;

    const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

    #[test]
    fn test_qwen3_conversion() {
        let qwen_files = ["qwen3_1.bin.gz", "qwen3_2.bin.gz", "qwen3_3.bin.gz"];

        for filename in qwen_files {
            let file_path = format!("{CARGO_MANIFEST_DIR}/tests/data/{filename}");
            println!("File path: {file_path}");

            // Read and decompress gzip file
            let file = std::fs::File::open(&file_path).unwrap();
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut buffer = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut buffer).unwrap();

            // Deserialize graph
            let graph = deserialize_graph(&buffer);
            assert!(graph.is_ok());

            // Convert to FXGraph
            let graph = graph.unwrap();
            let fxgraph = FXGraph::try_from(graph);
            assert!(fxgraph.is_ok());
        }
    }
}
