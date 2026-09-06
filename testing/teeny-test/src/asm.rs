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

use std::path::Path;

/// Reads a `compile_kernel`-produced artifact at `path` and returns it in the form compile-only
/// snapshot tests want: lossily decoded as UTF-8, and truncated at the first `.file` debug
/// directive.
///
/// Two backend-specific quirks this papers over so call sites don't have to:
/// - CUDA's artifact is real PTX text, but from the `.file` directive onward it embeds an
///   absolute, environment-specific cache-dir path, followed by a DWARF `.debug_info` dump that
///   re-encodes that same path byte-by-byte -- roughly doubling the snapshot's size with content
///   that isn't the actual generated instructions and isn't portable across machines/CI with a
///   different `TEENYC_CACHE_DIR`. Truncating there keeps the snapshot to just the instructions.
/// - RISC-V's artifact is today a real ELF binary (not text) -- see `teenygrad-1zd`, the
///   backend is still a placeholder-codegen stub -- so a plain `read_to_string` would fail with
///   "stream did not contain valid UTF-8". The lossy decode never fails; once real RISC-V codegen
///   lands this will presumably also become readable assembly and the lossy step becomes a no-op.
pub fn read_compiled_asm(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("missing compiled artifact {path:?}: {e}"));
    let text = String::from_utf8_lossy(&bytes);
    text.split("\t.file\t1")
        .next()
        .unwrap_or(&text)
        .trim()
        .to_string()
}
