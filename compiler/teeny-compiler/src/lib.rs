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

//! Compiles a [teenygrad](https://teenygrad.org) computational graph (FXGraph, traced from
//! `teeny-core`) down to a target backend — LLVM/MLIR object code today ([`compiler::backend`]),
//! with an `ndarray`-backed CPU path behind the `ndarray` feature.
//!
//! Compiling kernels through the LLVM backend shells out to the custom `teenyc` compiler at
//! *runtime* (via `TEENYC_PATH`) — not needed to build this crate itself. See the crate README
//! for the `cargo-teeny`/`TEENYC_PATH` setup.

pub mod compiler;
pub mod errors;
pub mod fxgraph;

/// Initialize logging for the teeny-compiler
pub fn init_logging() {
    use tracing_subscriber::{EnvFilter, fmt};

    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();
}
