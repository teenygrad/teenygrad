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

    let mut result = String::new();

    // Process the triton module
    if let Err(e) = process_directory(&triton_path, &triton_path, &mut result) {
        panic!("Failed to process triton module: {}", e);
    }

    // Write the output file
    let output = format!("pub const TRITON: &str = r#\"{}\"#;", result);
    let mut file = fs::File::create(&dest_path).expect("Failed to create output file");
    file.write_all(output.as_bytes())
        .expect("Failed to write output file");
}

fn process_directory(
    base_path: &Path,
    current_path: &Path,
    result: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = fs::read_dir(current_path)?;

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
            process_directory(base_path, &path, result)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let file_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("Invalid file name")?;

            let contents = fs::read_to_string(&path)?;

            // Special handling for core.rs - emit at top level
            if file_name == "core" {
                let filtered = apply_filter(base_path, file_name, &contents)?;
                result.push_str(&filtered);
                result.push('\n');
            } else {
                // Apply the filter
                let filtered = apply_filter(base_path, file_name, &contents)?;
                let mut module_name = file_name;

                if file_name == "mod" {
                    module_name = current_path.file_stem().and_then(|s| s.to_str()).unwrap();
                }

                // Wrap in module
                result.push_str(&format!("mod {} {{\n{}\n}}\n", module_name, filtered));
            }
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

    // Example filter that just returns the contents:
    Ok(contents.to_string())

    // If you need to use the path and name in your filter:
    // let _ = module_path;
    // let _ = module_name;
    // ... your logic here ...
}
