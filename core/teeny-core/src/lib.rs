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

//! Foundation crate of [teenygrad](https://teenygrad.org): tensor/graph types, the computational
//! graph ([`graph::Graph`]/[`graph::Op`]/[`graph::Shape`]), neural network layers ([`nn`]), the
//! dtype system ([`dtype`]), device abstraction ([`device`]), and name-scoping used by every
//! other `teeny-*` crate.
//!
//! `no_std` by default (the `std` feature is not in the default feature set) — enable `std` if
//! you need it. See the crate README for the full feature list.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
extern crate alloc;

/// Compiler-facing traits (targets, compiled kernels).
pub mod compiler;
/// Device abstraction (buffers, contexts, launch configuration).
pub mod device;
/// The dtype system: `Dtype`/`Float`/tensor traits kernel code is generic over.
pub mod dtype;
/// Error types.
pub mod errors;
/// The computational graph: `Graph`, `Op`, `Shape`, `SymTensor`.
pub mod graph;
/// Internal macro helpers.
pub mod macros;
/// Model execution traits (`Layer`, lowering, kernel-launch argument packing).
pub mod model;
/// Scoped naming for graph nodes/parameters (requires the `std` feature).
#[cfg(feature = "std")]
pub mod name_scope;
/// Standard neural network layers.
pub mod nn;
/// Runtime execution traits.
pub mod runtime;
/// Miscellaneous utilities (e.g. the `Dag` type).
pub mod utils;
