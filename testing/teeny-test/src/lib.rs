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

//! Shared test support for [teenygrad](https://teenygrad.org)'s integration tests, factored out
//! so the same fixture-loading / kernel-compiling helpers work across every driver backend
//! instead of each crate's tests hardcoding one (historically `teeny-cuda`) directly.
//!
//! - [`ExecKernel`], [`load_fixture`], [`load_fixture_i32`], [`teenyc_cache_dir`] are
//!   driver-agnostic and always available.
//! - [`cuda`] (feature `cuda`) wraps `teeny-cuda`'s device/program types for test setup.
//! - [`riscv`] (feature `riscv`) wraps `teeny-riscv`'s host-tool discovery; its [`riscv::qemu`]
//!   submodule (feature `qemu`, additionally) can actually execute a compiled kernel under
//!   `qemu-riscv64`.

mod cache;
mod exec_kernel;
mod fixtures;

pub use cache::teenyc_cache_dir;
pub use exec_kernel::ExecKernel;
pub use fixtures::{load_fixture, load_fixture_i32};

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "riscv")]
pub mod riscv;
