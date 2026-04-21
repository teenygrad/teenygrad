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

use crate::graph::call_function::call_function;
use crate::graph::call_method::call_method;
use crate::graph::example_input::into_example_input;
use crate::graph::output::output;
use crate::graph::placeholder::handle_placeholder;
use crate::{error::Error, torch::Graph};
use std::convert::TryFrom;

use teeny_fxgraph::fxgraph::FXGraph;

use crate::torch::Node;

mod call_function;
mod call_method;
mod dtype;
mod example_input;
mod keyvalue;
mod output;
mod placeholder;
mod shape;
mod symint;
mod value;

impl<'a> TryFrom<Graph<'a>> for FXGraph {
    type Error = Error;

    fn try_from(graph: Graph<'a>) -> Result<Self, Self::Error> {
        let mut fxgraph = FXGraph::new().map_err(Error::FxGraph)?;

        println!("Graph: {:?}", graph);

        let mut example_inputs = vec![];
        if let Some(example_input) = graph.example_inputs() {
            let inputs = example_input.inputs();
            if let Some(inputs) = inputs {
                for input in inputs {
                    let value = into_example_input(input)?;
                    example_inputs.push(value);
                }
            }
        }

        // Safely access nodes with error handling
        let nodes = graph.nodes().ok_or(Error::NoGraphNodes)?;

        for node in nodes {
            let node_type = node.node_type();
            match node_type {
                Node::placeholder => {
                    let placeholder = node
                        .node_as_placeholder()
                        .ok_or(Error::InvalidBuffer(format!("{node:?}")))?;
                    handle_placeholder(&mut fxgraph, &placeholder, &example_inputs)?;
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
        fxgraph.verify_types().map_err(Error::FxGraph)?;

        Ok(fxgraph)
    }
}

// Static HashMap mapping function names to their handler functions

#[cfg(test)]
mod tests {
    use crate::torch::deserialize_graph;

    use super::*;

    const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

    // #[test]
    #[allow(unused)]
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

    #[test]
    fn test_vector_add_conversion() -> Result<(), Error> {
        let vector_add_files = ["vector_add_1.bin.gz"];

        for filename in vector_add_files {
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
