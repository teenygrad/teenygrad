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
//! Built for a `riscv64` host, [`device::RiscvDevice`] runs launched kernels natively, in-process.
//! Built for any other host, it runs them under `qemu-riscv64` with the `qemu` feature, and
//! launching a kernel panics without it.

#![warn(missing_docs)]

/// `Target`/`Capability` types for compiling kernels via `LlvmCompiler`.
pub mod compiler;
/// `teeny_core::device` trait implementations -- a "device" here is just the local machine or
/// the `qemu-riscv64` environment kernels run under, not a discrete accelerator.
pub mod device;
mod elf;
/// Error types.
pub mod errors;
#[cfg(any(target_arch = "riscv64", test))]
mod native;
#[cfg(all(feature = "qemu", not(target_arch = "riscv64")))]
mod qemu;
/// Loading and calling a compiled kernel's shared library.
pub mod runtime;
