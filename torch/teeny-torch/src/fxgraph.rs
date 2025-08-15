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
use std::{collections::HashSet, convert::TryFrom};

use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang, literal::ConstantValue};

use crate::graph::{Node, OpType};

impl<'a> TryFrom<Graph<'a>> for FXGraph {
    type Error = Error;

    fn try_from(graph: Graph<'a>) -> Result<Self, Self::Error> {
        let mut fxgraph = FXGraph::new();
        let mut funcs = HashSet::new();
        let mut built_ins = HashSet::new();
        let mut modules = HashSet::new();

        // Safely access nodes with error handling
        let nodes = graph.nodes().ok_or(Error::NoGraphNodes)?;

        for node in nodes {
            let _name = node.name().ok_or(Error::NoGraphNodeName)?;
            let op = node.op();
            let target = node.target().ok_or(Error::NoGraphNodeTarget)?;

            if op == OpType::call_function {
                funcs.insert(target);
            } else if op == OpType::call_method {
                built_ins.insert(target);
            } else if op == OpType::call_module {
                modules.insert(target);
            }
            // match op {
            //     OpType::placeholder => {
            //         handle_placeholder(&mut fxgraph, &node)?;
            //     }
            //     OpType::call_function => {
            //         handle_call_function(&mut fxgraph, &node)?;
            //     }
            //     OpType::call_method => {
            //         handle_call_method(&mut fxgraph, &node)?;
            //     }
            //     _ => {
            //         println!("{op:?} {_target:?}");
            //         // return Err(Error::UnsupportedOp(op));
            //     }
            // }
        }

        println!("================");
        for func in funcs {
            println!("{func:?}");
        }
        for built_in in built_ins {
            println!("{built_in:?}");
        }
        for module in modules {
            println!("{module:?}");
        }
        println!("================");
        Ok(fxgraph)
    }
}

fn handle_placeholder(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node.target().ok_or(Error::NoGraphNodeTarget)?;
    let id = fxgraph.add_operation(
        target,
        FxGraphLang::Constant(ConstantValue::String(target.to_string())),
    );

    fxgraph.add_operation(
        &format!("placeholder#{target}"),
        FxGraphLang::Placeholder([id]),
    );

    Ok(())
}

fn handle_call_function(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node.target().ok_or(Error::NoGraphNodeTarget)?;
    let op = node.op();
    let args = node.args();

    let functions = [
        "<function embedding",
        "<function lazy_load_decompositions",
        "<function _enter_autocast",
        "aten.index",
    ];
    let built_ins = [
        "<built-in method arange",
        "<built-in function iadd",
        "<built-in method _vmap_increment_nesting",
        "<built-in method _add_batch_dim",
        "<built-in method _remove_batch_dim",
        "<built-in method _vmap_decrement_nesting",
        "<built-in function getitem>",
    ];

    if functions.iter().any(|f| target.starts_with(f)) {
        // todo handle functions
    } else if built_ins.iter().any(|f| target.starts_with(f)) {
        // todo handle built-ins
    } else {
        println!("{target:?} {op:?} {args:?}");
        return Err(Error::UnsupportedOp(op));
    }

    Ok(())
}

fn handle_call_method(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
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
