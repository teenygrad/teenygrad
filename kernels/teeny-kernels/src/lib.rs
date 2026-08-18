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

//! CPU/GPU kernel implementations of [teenygrad](https://teenygrad.org)'s `nn` layers
//! ([`nn`] — including a Flash Attention 2 forward/backward implementation, [`math`] ops, and
//! [`graph`] lowering), written against the `teeny-triton` DSL and compiled via `teeny-compiler`.
//!
//! The `cuda` feature (on by default) enables the `teeny-cuda` backend, which requires the CUDA
//! toolkit to build — see its crate README. Running/compiling kernels additionally needs the
//! custom `teenyc` compiler at runtime; see `teeny-compiler`'s README.

// `#[kernel]`-annotated functions naturally take many parameters (pointers, strides, dims,
// block-size const-generics) matching the CUDA/Triton kernel calling convention -- inherent to
// the domain, not something to refactor away. `Dag<Box<dyn ExecutableOp>>`-style return types
// (graph::lower) are similarly intentional, not accidental complexity.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod errors;
pub mod graph;
pub mod math;
pub mod nn;
pub mod testing;
