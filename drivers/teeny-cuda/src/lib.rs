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

//! The CUDA device backend for [teenygrad](https://teenygrad.org) — driver bindings (via
//! `bindgen` against the CUDA headers), device/runtime abstraction ([`device`], [`runtime`]),
//! and the AOT/JIT kernel-compilation path ([`compiler`]) used by `teeny-kernels`' `cuda`
//! feature.
//!
//! Building this crate requires the CUDA toolkit headers/libs on the host (`build.rs` links
//! against them unconditionally); see the crate README for `CUDA_INCLUDE_DIR`/`CUDA_LIB_DIR` and
//! the separate, runtime-only `teenyc`/`TEENYC_PATH` requirement for actually compiling kernels.

#![warn(missing_docs)]

/// AOT/JIT kernel compilation.
pub mod compiler;
/// Device and context management.
pub mod device;
/// Error types.
pub mod errors;
/// Loaded-model execution.
pub mod model;
/// CUDA runtime abstraction (streams, memory, launches).
pub mod runtime;
/// Test helpers for `teeny-cuda`'s own test suite.
pub mod testing;

mod cuda;

/// Signal nsys (or any CUDA profiler) to start capturing.
///
/// # Safety
/// Calls `cudaProfilerStart` via the CUDA runtime. Safe to call multiple times;
/// has no effect if no profiler is attached.
pub unsafe fn cuda_profiler_start() {
    unsafe { cuda::cudaProfilerStart() };
}

/// Signal nsys (or any CUDA profiler) to stop capturing.
///
/// # Safety
/// Calls `cudaProfilerStop` via the CUDA runtime.
pub unsafe fn cuda_profiler_stop() {
    unsafe { cuda::cudaProfilerStop() };
}
