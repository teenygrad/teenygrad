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

//! A chain member's per-element compute, with its own load/store boilerplate
//! stripped, ready to splice into a foreign kernel's grid shape
//! (teenygrad-3w0.9's reduction-terminated fusion).
//!
//! Derived by `teeny_macros::tiled_kernel` at proc-macro time -- not at fusion-pass
//! runtime -- since only the macro has `syn` AST access to the kernel
//! author's original (pre-prelude) statements. Only available for
//! single-input, single-axis `#[tile(...)]` kernels whose hand-written body
//! ends in a plain `T::store(...)` call (the macro's auto-prelude only
//! injects the *load* side; the trailing store stays author-written, so
//! extracting a splice-ready "core" means stripping that trailing statement
//! and capturing the last computed identifier).

/// See the module docs. `input_ident` is the name the auto-prelude binds the
/// loaded input tensor to (e.g. `"x"`); `output_ident` is the identifier
/// bound by the body's last `let` statement before the trailing store (e.g.
/// `elu_forward`'s `"y"`); `body_source` is every statement before that
/// trailing store, as Rust/DSL source text; `extra_params` are this kernel's
/// non-pointer, non-block, non-extent scalar parameters (name, type), e.g.
/// `elu_forward`'s `[("alpha", "f32")]`, needed by a caller re-threading this
/// core into a synthesized kernel with its own parameter list.
#[derive(Debug, Clone, Copy)]
pub struct FusionCore {
    /// Identifier the auto-prelude binds the loaded input to.
    pub input_ident: &'static str,
    /// Identifier bound by the last statement before the trailing store.
    pub output_ident: &'static str,
    /// Source text of every statement before the trailing store.
    pub body_source: &'static str,
    /// This kernel's extra scalar parameters, as `(name, type)` pairs.
    pub extra_params: &'static [(&'static str, &'static str)],
}
