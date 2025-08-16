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
use teeny_core::fxgraph::{
    FXGraph,
    lang::{FxGraphLang, const_bool, const_int, const_string},
    literal::ConstantValue,
};

use crate::graph::{Node, OpType};

impl<'a> TryFrom<Graph<'a>> for FXGraph {
    type Error = Error;

    fn try_from(graph: Graph<'a>) -> Result<Self, Self::Error> {
        let mut fxgraph = FXGraph::new();

        // Safely access nodes with error handling
        let nodes = graph.nodes().ok_or(Error::NoGraphNodes)?;

        for node in nodes {
            let _name = node
                .name()
                .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;
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

// Static HashMap mapping function names to their handler functions
type NodeHandler = fn(&mut FXGraph, &Node) -> Result<(), Error>;
static FUNCS: OnceLock<HashMap<String, NodeHandler>> = OnceLock::new();

// Initialize the functions HashMap
fn get_functions() -> &'static HashMap<String, NodeHandler> {
    FUNCS.get_or_init(|| {
        HashMap::from([
            ("embedding".to_string(), embedding as NodeHandler),
            ("_enter_autocast".to_string(), enter_autocast as NodeHandler),
            ("_exit_autocast".to_string(), exit_autocast as NodeHandler),
            (
                "lazy_load_decompositions".to_string(),
                lazy_load_decompositions as NodeHandler,
            ),
            ("silu".to_string(), silu as NodeHandler),
            ("getitem".to_string(), getitem as NodeHandler),
            ("sym_sum".to_string(), sym_sum as NodeHandler),
            ("aten.index".to_string(), aten_index as NodeHandler),
            ("arange".to_string(), arange as NodeHandler),
            ("iadd".to_string(), iadd as NodeHandler),
            ("add".to_string(), add as NodeHandler),
            ("matmul".to_string(), matmul as NodeHandler),
            (
                "vmap_increment_nesting".to_string(),
                vmap_increment_nesting as NodeHandler,
            ),
            (
                "vmap_decrement_nesting".to_string(),
                vmap_decrement_nesting as NodeHandler,
            ),
            ("add_batch_dim".to_string(), add_batch_dim as NodeHandler),
            (
                "remove_batch_dim".to_string(),
                remove_batch_dim as NodeHandler,
            ),
        ])
    })
}

fn handle_placeholder(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;

    fxgraph.add_operation(
        node.name()
            .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?,
        FxGraphLang::Placeholder(target.to_string()),
    );

    println!("Placeholder added: {}", node.name().unwrap());
    Ok(())
}

fn call_function(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;
    let pat =
        r#"^(?:<function\s+|<built-in\s+(?:method|function)\s+|aten\.)([a-zA-Z_][a-zA-Z0-9_]*)"#;
    let re = Regex::new(pat).unwrap();
    let func_name = re
        .captures(target)
        .ok_or_else(|| Error::NoMatchingFunction(format!("{node:?}")))?
        .get(1)
        .ok_or_else(|| Error::NoMatchingFunction(format!("{node:?}")))?
        .as_str();
    println!("Func name: {func_name:?}");

    let functions = get_functions();
    if let Some(handler) = functions.get(func_name) {
        println!("Calling function handler: {func_name:?}");
        handler(fxgraph, node)?;
    } else {
        println!("Unknown function: {func_name:?}");
    }

    Ok(())
}

fn getitem(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn lazy_load_decompositions(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn silu(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn sym_sum(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn aten_index(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn add(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn iadd(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn arange(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let args = node
        .args()
        .ok_or_else(|| Error::NoGraphNodeArgs(format!("args: {node:?}")))?;

    let start;
    let end;
    let step;

    match args.len() {
        1 => {
            start = fxgraph.add_operation(&fxgraph.unique_name(), const_int(0));
            end = fxgraph
                .get_node(args.get(0))
                .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;
            step = fxgraph.add_operation(&fxgraph.unique_name(), const_int(1));
        }
        2 => {
            start = fxgraph
                .get_node(args.get(0))
                .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;
            end = fxgraph
                .get_node(args.get(1))
                .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;
            step = fxgraph.add_operation(&fxgraph.unique_name(), const_int(1));
        }
        3 => {
            start = fxgraph
                .get_node(args.get(0))
                .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;
            end = fxgraph
                .get_node(args.get(1))
                .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;
            step = fxgraph
                .get_node(args.get(2))
                .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;
        }
        _ => {
            return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
        }
    }

    fxgraph.add_operation(
        &fxgraph.unique_name(),
        FxGraphLang::Arange([start, end, step]),
    );

    Ok(())
}

fn embedding(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let args = node
        .args()
        .ok_or_else(|| Error::NoGraphNodeArgs(format!("args: {node:?}")))?;

    if args.len() != 5 {
        return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
    }

    let input_ids = fxgraph
        .get_node(args.get(0))
        .ok_or_else(|| Error::GraphNodeNotFound(format!("input_ids: {node:?}")))?;

    let weight = fxgraph
        .get_node(args.get(1))
        .ok_or_else(|| Error::GraphNodeNotFound(format!("weight: {node:?}")))?;

    // AXM TODO: What is this arg?
    let item = fxgraph.add_operation(&fxgraph.unique_name(), const_string(args.get(2)));

    let arg3 = fxgraph.add_operation(
        &fxgraph.unique_name(),
        const_bool(
            args.get(3)
                .to_lowercase()
                .parse::<bool>()
                .map_err(Error::ParseBoolError)?,
        ),
    );

    let arg4 = fxgraph.add_operation(
        &fxgraph.unique_name(),
        const_bool(
            args.get(4)
                .to_lowercase()
                .parse::<bool>()
                .map_err(Error::ParseBoolError)?,
        ),
    );

    fxgraph.add_operation(
        &fxgraph.unique_name(),
        FxGraphLang::Embedding([input_ids, weight, item, arg3, arg4]),
    );

    println!("Embedding added: {node:?}");
    Ok(())
}

fn add_batch_dim(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn remove_batch_dim(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn vmap_decrement_nesting(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn enter_autocast(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn exit_autocast(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn matmul(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn vmap_increment_nesting(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    todo!("{node:?}")
}

fn call_method(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let _target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;
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
    fn test_qwen3_conversion() -> Result<(), Error> {
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
            let graph = deserialize_graph(&buffer)?;

            // Convert to FXGraph
            let fxgraph = FXGraph::try_from(graph)?;

            println!("FXGraph: {fxgraph:?}");
        }

        Ok(())
    }
}
