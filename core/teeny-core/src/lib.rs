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
extern crate alloc;

pub mod compiler;
pub mod device;
pub mod dtype;
pub mod errors;
pub mod graph;
pub mod macros;
pub mod model;
#[cfg(feature = "std")]
pub mod name_scope;
pub mod nn;
pub mod runtime;
pub mod utils;
