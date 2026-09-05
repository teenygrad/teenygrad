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

//! The RISC-V device backend for [teenygrad](https://teenygrad.org) — `Target`/`Capability`
//! types for the `mlir`/Triton compiler backend's `riscv64-generic` path ([`compiler`]), and a
//! `libloading`-based runtime for loading a compiled kernel's shared library and calling its
//! exported symbol ([`runtime`]).
//!
//! See the crate README for current status: the underlying compiler backend is an early stub, so
//! only the placeholder no-argument kernel it always produces can be compiled/loaded today.

#![warn(missing_docs)]

/// `Target`/`Capability` types for compiling kernels via `LlvmCompiler`.
pub mod compiler;
/// Error types.
pub mod errors;
/// Loading and calling a compiled kernel's shared library.
pub mod runtime;
