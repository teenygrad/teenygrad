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

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use cargo_metadata::MetadataCommand;

fn main() {
    let metadata = MetadataCommand::new().exec().unwrap();

    let package = metadata
        .packages
        .iter()
        .find(|p| p.name.as_str() == "teeny-triton")
        .unwrap();
    let project_dir: PathBuf = package.manifest_path.parent().unwrap().into();
    let triton_path = Path::new(&project_dir).join("src/triton");
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("triton_lang.rs");

    // Tell Cargo to rerun if the triton directory changes
    println!("cargo:rerun-if-changed={}", triton_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    // Load teeny-core's dtype/mod.rs and transform it for the no_core DSL context.
    // In the DSL, core.rs defines a fake `std::ops`, but `core::ops` doesn't exist.
    // We replace `core::ops` with `std::ops` so the generated DSL string compiles.
    let core_pkg = metadata
        .packages
        .iter()
        .find(|p| p.name.as_str() == "teeny-core")
        .unwrap();
    let dtype_path = core_pkg
        .manifest_path
        .parent()
        .unwrap()
        .join("src/dtype/mod.rs");
    println!("cargo:rerun-if-changed={}", dtype_path);

    let dtype_for_dsl = fs::read_to_string(&dtype_path)
        .unwrap_or_else(|e| panic!("Failed to read teeny-core dtype: {}", e));

    let mut result = String::new();

    // Process the triton module
    if let Err(e) = process_module(&triton_path, &triton_path, 1, &mut result, &dtype_for_dsl) {
        panic!("Failed to process triton module: {}", e);
    }

    result.push_str("pub use triton::*;\n");

    // Write the output file
    let output = format!("pub const TRITON: &str = r#\"{}\"#;", result);
    let mut file = fs::File::create(&dest_path).expect("Failed to create output file");
    file.write_all(output.as_bytes())
        .expect("Failed to write output file");
}

fn process_module(
    base_path: &Path,
    current_path: &Path,
    depth: usize,
    result: &mut String,
    dtype_for_dsl: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = fs::read_dir(current_path)?;

    let module_name = current_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Invalid file name")?;

    process_file(
        base_path,
        current_path.join("mod.rs").as_path(),
        module_name,
        false,
        depth,
        result,
        dtype_for_dsl,
    )?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // if the path ends in dummy then ignore
            if let Some(name) = path.file_name().and_then(|s| s.to_str())
                && name == "dummy"
            {
                continue;
            }
            // Recursively process subdirectories
            process_module(base_path, &path, depth + 1, result, dtype_for_dsl)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let file_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("Invalid file name")?;

            if file_name == "mod" {
                // module is handled differently
                continue;
            }

            process_file(base_path, &path, file_name, true, depth, result, dtype_for_dsl)?;
        }
    }

    // Add the module suffix
    result.push_str("}\n");

    Ok(())
}

fn process_file(
    base_path: &Path,
    path: &Path,
    file_name: &str,
    add_suffix: bool,
    depth: usize,
    result: &mut String,
    dtype_for_dsl: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // For the `types` module, use the transformed teeny-core dtype content
    // instead of the file's actual content (which is a re-export, unusable in no_core).
    let contents = if file_name == "types" {
        dtype_for_dsl.to_string()
    } else {
        fs::read_to_string(path)?
    };

    // Special handling for core.rs - emit at top level
    if file_name == "core" {
        let filtered = apply_filter(base_path, file_name, &contents)?;
        result.insert_str(0, &filtered);
    } else {
        // Apply the filter
        let filtered = apply_filter(base_path, file_name, &contents)?;

        // Wrap in module
        let core = format!("pub use {}*;", "super::".repeat(depth));
        result.push_str(&format!(
            "pub mod {} {{\n{}\n{}\n",
            file_name, core, filtered
        ));
        if add_suffix {
            result.push_str("}\n");
        }
    }

    Ok(())
}

fn apply_filter(
    _module_path: &Path,
    _module_name: &str,
    contents: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // TODO: Implement your actual filter logic here
    // This is a placeholder that just returns the contents as-is
    // Replace this with your actual filtering logic

    let filtered = contents
        .lines()
        .filter(|line| !(line.starts_with("pub mod") && line.ends_with(";")))
        .collect::<Vec<&str>>()
        .join("\n")
        .to_string();

    Ok(filtered)
}
