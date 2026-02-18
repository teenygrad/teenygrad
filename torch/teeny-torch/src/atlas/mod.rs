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

use pyo3::prelude::*;
use std::{fs::File, io::Write};

use crate::torch::*;

#[pyfunction]
pub fn atlas_compile(buffer: &[u8]) -> pyo3::PyResult<String> {
    println!("Writing file to /tmp/model");

    write_file(buffer).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write file: {e}"))
    })?;

    let graph = deserialize_graph(buffer).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to deserialize graph: {e}"))
    })?;

    println!("Graph: {:?}", graph);
    // Safely access nodes with error handling
    let _nodes = graph
        .nodes()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Graph has no nodes"))?;

    // for i in 0..nodes.len() {
    //     let node = nodes.get(i);

    //     // Safely access node properties with null checks
    //     let node_name = node.name().ok_or_else(|| {
    //         pyo3::exceptions::PyRuntimeError::new_err(format!("Node {i} has no name"))
    //     })?;

    //     let op = node.op();
    //     let target = node.target().ok_or_else(|| {
    //         pyo3::exceptions::PyRuntimeError::new_err(format!("Node '{node_name}' has no target"))
    //     })?;

    //     println!("Node name: {node_name}");
    //     println!("Op: {op:?}");
    //     println!("Target: {target}");

    //     // Safely process args
    //     if let Some(args) = node.args() {
    //         for j in 0..args.len() {
    //             let arg = args.get(j);
    //             println!("Arg {j}: {arg:?}");
    //         }
    //     } else {
    //         println!("Args: <none>");
    //     }

    //     // Safely process users
    //     if let Some(users) = node.users() {
    //         for user in users.iter() {
    //             println!("-> User: {user}");
    //         }
    //     } else {
    //         println!("Users: <none>");
    //     }
    // }

    println!("Example inputs:");
    for example in graph.example_inputs().iter() {
        println!("Example: {example:?}");
    }

    Ok("Graph deserialized successfully".to_string())
}

fn write_file(data: &[u8]) -> PyResult<()> {
    // Generate a unique filename by adding a digit suffix to "qwen_[digit].bin"
    use std::path::Path;
    let mut unique_path = String::new();
    for i in 1..1000 {
        // Create /tmp/model directory if it doesn't exist
        let model_dir = "/tmp/model";
        if !Path::new(model_dir).exists() {
            std::fs::create_dir_all(model_dir).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create directory {model_dir}: {e}"
                ))
            })?;
        }
        let candidate = format!("{model_dir}/model_{i}.bin");
        if !Path::new(&candidate).exists() {
            unique_path = candidate;
            break;
        }
    }
    let path = &unique_path;

    println!("Writing file to {path}-{}", data.len());
    let mut file = File::create(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create file: {e}"))
    })?;
    file.write_all(data).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write to file: {e}"))
    })?;
    Ok(())
}
