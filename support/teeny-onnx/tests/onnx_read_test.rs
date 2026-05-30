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

use std::path::PathBuf;

use teeny_onnx::Onnx;

#[test]
fn test_read_onnx_files() {
    let mut resource_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    resource_path.push("onnx/onnx/backend/test/data");

    // Recursively search for .onnx files and test parsing them
    fn visit_dirs_and_test_onnx<P: Into<PathBuf>>(dir: P) {
        fn visit_dir(dir: &PathBuf) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        visit_dir(&path);
                    } else if let Some(ext) = path.extension()
                        && ext == "onnx"
                    {
                        // Try reading and parsing the file through the public ONNX API.
                        let result = Onnx::from_path(&path);

                        assert!(
                            result.is_ok(),
                            "Failed to parse graph: {:?} - {:?}",
                            path,
                            result.err()
                        );
                    }
                }
            }
        }

        let root = dir.into();
        visit_dir(&root);
    }

    visit_dirs_and_test_onnx(&resource_path);
}
