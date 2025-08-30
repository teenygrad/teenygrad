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

use crate::fxgraph::call_function::call_function;
use crate::fxgraph::call_method::call_method;
use crate::fxgraph::output::output;
use crate::fxgraph::placeholder::handle_placeholder;
use crate::{error::Error, torch::Graph};
use std::convert::TryFrom;

use teeny_core::fxgraph::FXGraph;

use crate::torch::Node;

mod call_function;
mod call_method;
mod dtype;
mod output;
mod placeholder;
mod shape;
mod symint;
mod util;
mod value;

impl<'a> TryFrom<Graph<'a>> for FXGraph {
    type Error = Error;

    fn try_from(graph: Graph<'a>) -> Result<Self, Self::Error> {
        let mut fxgraph = FXGraph::new();

        // Safely access nodes with error handling
        let nodes = graph.nodes().ok_or(Error::NoGraphNodes)?;

        for node in nodes {
            let node_type = node.node_type();
            match node_type {
                Node::placeholder => {
                    let placeholder = node
                        .node_as_placeholder()
                        .ok_or(Error::InvalidBuffer(format!("{node:?}")))?;
                    handle_placeholder(&mut fxgraph, &placeholder)?;
                }
                Node::call_function => {
                    let node = node
                        .node_as_call_function()
                        .ok_or(Error::InvalidBuffer(format!("{node:?}")))?;
                    call_function(&mut fxgraph, &node)?;
                }
                Node::call_method => {
                    let node = node
                        .node_as_call_method()
                        .ok_or(Error::InvalidBuffer(format!("{node:?}")))?;
                    call_method(&mut fxgraph, &node)?;
                }
                Node::output => {
                    let node = node
                        .node_as_output()
                        .ok_or(Error::InvalidBuffer(format!("{node:?}")))?;
                    output(&mut fxgraph, &node)?;
                }
                _ => {
                    unimplemented!("Unknown node type: {node_type:?}");
                }
            }
        }

        fxgraph.egraph.rebuild();
        Ok(fxgraph)
    }
}

// Static HashMap mapping function names to their handler functions

#[cfg(test)]
mod tests {
    use crate::torch::deserialize_graph;

    use super::*;

    const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

    #[test]
    fn test_qwen3_conversion() -> Result<(), Error> {
        let qwen_files = ["qwen3_1.bin.gz", "qwen3_2.bin.gz", "qwen3_3.bin.gz"];

        for filename in qwen_files {
            let file_path = format!("{CARGO_MANIFEST_DIR}/tests/data/{filename}");

            // Read and decompress gzip file
            let file = std::fs::File::open(&file_path).unwrap();
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut buffer = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut buffer).unwrap();

            // Deserialize graph
            let graph = deserialize_graph(&buffer)?;

            // Convert to FXGraph
            let fxgraph = FXGraph::try_from(graph)?;
            println!("FXGraph: #nodes {}", fxgraph.node_map.len());
        }

        Ok(())
    }
}
