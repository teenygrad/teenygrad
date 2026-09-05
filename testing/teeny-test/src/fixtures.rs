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

/// Load a little-endian `f32` fixture under `tests/fixtures/{rel}` of the calling crate
/// (resolved via `$CARGO_MANIFEST_DIR` at the call site, not this crate's).
pub fn load_fixture(manifest_dir: &str, rel: &str) -> Vec<f32> {
    let path = format!("{manifest_dir}/tests/fixtures/{rel}");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Load a little-endian `i32` fixture under `tests/fixtures/{rel}` of the calling crate
/// (resolved via `$CARGO_MANIFEST_DIR` at the call site, not this crate's).
pub fn load_fixture_i32(manifest_dir: &str, rel: &str) -> Vec<i32> {
    let path = format!("{manifest_dir}/tests/fixtures/{rel}");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}
