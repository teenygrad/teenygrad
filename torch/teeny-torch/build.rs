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

use std::{
    env,
    path::{Path, PathBuf},
};

use cargo_metadata::MetadataCommand;

fn main() {
    let metadata = MetadataCommand::new().exec().unwrap();

    let package = metadata
        .packages
        .iter()
        .find(|p| p.name.as_str() == "teeny-torch")
        .unwrap();
    let project_dir: PathBuf = package.manifest_path.parent().unwrap().into();

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=graph.fbs");
    println!("cargo:rerun-if-changed=src/");

    flatc_rust::run(flatc_rust::Args {
        inputs: &[Path::new("graph.fbs")],
        out_dir: out_path.join("flatbuffers").as_path(),
        ..Default::default()
    })
    .expect("flatc-rust");

    // Remove the folder python/teenygrad/graph/FXGraph if it exists
    let fxgraph_dir = project_dir.join("python/teenygrad/graph/FXGraph");
    if fxgraph_dir.exists()
        && let Err(e) = std::fs::remove_dir_all(&fxgraph_dir)
    {
        eprintln!("Warning: Failed to remove FXGraph directory: {e}");
    }

    flatc_rust::run(flatc_rust::Args {
        lang: "python",
        inputs: &[Path::new("graph.fbs")],
        out_dir: project_dir.join("python/teenygrad/graph").as_path(),
        ..Default::default()
    })
    .expect("flatc-python");

    pyo3_build_config::add_extension_module_link_args();

    // Run the import fix script after generating Python files
    let status = std::process::Command::new("python3")
        .arg("fix_imports.py")
        .current_dir(&project_dir)
        .status();

    if let Err(e) = status {
        eprintln!("Warning: Failed to run import fix script: {}", e);
    }
}
