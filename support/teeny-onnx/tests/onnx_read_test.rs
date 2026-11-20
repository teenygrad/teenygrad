/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

use std::path::PathBuf;

use teeny_onnx::{Onnx, errors::Error};

/// Every model in the ONNX backend test data must parse. A model the importer deliberately
/// refuses (`Error::Unsupported`: an attribute it would otherwise compute wrongly, say) is
/// counted but allowed; any other error fails the test, listing every such model.
#[test]
fn test_read_onnx_files() {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("onnx/onnx/backend/test/data");

    fn visit(dir: &PathBuf, files: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for path in entries.flatten().map(|e| e.path()) {
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "onnx") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    visit(&root, &mut files);
    assert!(
        !files.is_empty(),
        "no .onnx files under {root:?}; is the onnx submodule initialised?"
    );

    let (mut refused, mut failed) = (0usize, Vec::new());
    for path in &files {
        if let Err(e) = Onnx::from_path(path) {
            match e.downcast_ref::<Error>() {
                Some(Error::Unsupported(_)) => refused += 1,
                _ => failed.push(format!("{}: {e}", path.display())),
            }
        }
    }
    println!(
        "{} models: {} parsed, {refused} refused as unsupported",
        files.len(),
        files.len() - refused - failed.len()
    );
    assert!(
        failed.is_empty(),
        "{} models failed to parse:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
