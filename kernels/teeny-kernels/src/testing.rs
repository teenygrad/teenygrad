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

//! Shared helpers for `teeny-kernels` integration tests.

use teeny_core::device::program::Kernel;
use teeny_core::model::ExecutableOp;

/// Adapts a lowered [`ExecutableOp`] to [`Kernel`] for [`teeny_cuda::compiler::compile_kernel`].
pub struct ExecKernel<'a>(pub &'a dyn ExecutableOp);

impl Kernel for ExecKernel<'_> {
    type Args<'b> = ();

    fn name(&self) -> &str {
        self.0.name()
    }

    fn source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn kernel_source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn entry_point_source(&self) -> &str {
        ""
    }

    fn entry_point_name(&self) -> String {
        self.0.forward_kernel_entry_point().to_string()
    }
}

/// Load a little-endian `f32` fixture under `tests/fixtures/{rel}`.
pub fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Load a little-endian `i32` fixture under `tests/fixtures/{rel}`.
pub fn load_fixture_i32(rel: &str) -> Vec<i32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

/// Resolve the `teenyc` scratch/cache directory (`TEENYC_CACHE_DIR`, else `/tmp/teenyc_cache`).
pub fn teenyc_cache_dir() -> String {
    std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string())
}
