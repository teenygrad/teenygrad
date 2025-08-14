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
    if fxgraph_dir.exists() {
        if let Err(e) = std::fs::remove_dir_all(&fxgraph_dir) {
            eprintln!("Warning: Failed to remove FXGraph directory: {e}");
        }
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
