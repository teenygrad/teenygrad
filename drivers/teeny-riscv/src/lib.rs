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

//! The RISC-V device backend for [teenygrad](https://teenygrad.org) — `Target`/`Capability`
//! types for the `mlir`/Triton compiler backend's `riscv64-generic` path ([`compiler`]),
//! `teeny_core::device` implementations ([`device`]), and a `libloading` wrapper for calling a
//! kernel's exported symbol on a native RISC-V host ([`runtime`]).
//!
//! There is no RISC-V hardware support yet. With the `qemu` feature, [`device::RiscvDevice`]
//! runs launched kernels under `qemu-riscv64`; without it, launching a kernel panics.

#![warn(missing_docs)]

/// `Target`/`Capability` types for compiling kernels via `LlvmCompiler`.
pub mod compiler;
/// `teeny_core::device` trait implementations -- a "device" here is just the local machine or
/// the `qemu-riscv64` environment kernels run under, not a discrete accelerator.
pub mod device;
mod elf;
/// Error types.
pub mod errors;
#[cfg(feature = "qemu")]
mod qemu;
/// Loading and calling a compiled kernel's shared library.
pub mod runtime;
