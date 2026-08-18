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

//! Procedural macros for [teenygrad](https://teenygrad.org). Provides the
//! [`macro@kernel`] attribute macro, used to mark functions as GPU/CPU kernel definitions
//! consumed by `teeny-triton`/`teeny-kernels`, and the more opinionated [`macro@tiled_kernel`]
//! variant, which additionally understands the `#[tile(...)]` tile-shape DSL.

#![warn(missing_docs)]

use proc_macro::TokenStream;

mod macros;

/// Marks a function as a kernel definition for `teeny-triton`/`teeny-kernels`.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    macros::kernel::kernel(attr, item)
}

/// Marks a function as a kernel definition, like [`macro@kernel`], but additionally
/// opts into the `#[tile(...)]`/`#[tile_loop(...)]`/`#[tile_pid_swizzle(...)]` tile-shape
/// DSL (teenygrad-3w0): auto-generated load preludes, GEMM's swizzled `pid` decode,
/// `KernelTileSpec` for the cost model, and splice-ready fusion cores. Use this instead
/// of [`macro@kernel`] when the kernel's tile shape should be declared, not hand-rolled.
#[proc_macro_attribute]
pub fn tiled_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    macros::tiled_kernel::tiled_kernel(attr, item)
}
