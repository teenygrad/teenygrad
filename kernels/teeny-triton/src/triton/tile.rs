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

// A tensor plus an optional boundary mask (teenygrad-1nr.1). `#[tiled_kernel]`
// functions declare it directly as an entry-parameter type, wrapped in the
// same `In`/`Out`/`InOut` ABI markers used for raw pointers today (e.g.
// `x: In<Tile<T, D>>`) -- but `Tile` itself never crosses the *real*
// device/host ABI: the macro rewrites `In<Tile<HW,D>>`/`Out<Tile<HW,D>>`
// back to `In<HW::Pointer<D>>`/`Out<HW::Pointer<D>>` in both the generated
// device source and the real compiled host function
// (`common::rewrite_tile_param_to_pointer`/`unwrap_pointer_marker`). An
// `In<Tile<..>>` parameter additionally gets an auto-generated load prelude
// (pid decode + masked `T::load`) spliced ahead of the kernel author's body,
// shadowing the parameter name with the loaded `Tile` value; `Out<Tile<..>>`
// parameters get no such treatment and stay a plain pointer in the body --
// `T::store` is still explicit.
//
// Keep this file free of derive, //! docs, and host-only APIs — same
// constraint as `ptr.rs`, since kernel bodies are spliced into the
// no_core device-side source string.

use super::Triton;
use super::types::{self as ty};

/// A tensor value plus an optional boundary mask — the pair every kernel
/// already threads through `T::load`/`T::store` by convention
/// (`T::load(ptr, Some(mask), ...)`), collapsed into one type so kernel
/// bodies can operate on it as a single value instead of managing the mask
/// separately.
///
/// See this module's doc comment for how `Tile` relates to the real
/// device/host ABI when used as an `In`/`Out`/`InOut`-wrapped entry
/// parameter.
///
/// Deliberately no `Clone`/`Copy` impl: `teenyc`'s no_core device
/// environment doesn't resolve `Option<T::BoolTensor>: Copy` even though
/// `T::BoolTensor: Copy` holds (teenyc-3af.1's own test fixture defines its
/// `Tile` with no derives either, move-only). Read `.tensor`/`.mask` once
/// each and move them onward rather than reusing `tile` itself.
pub struct Tile<T: Triton, D: ty::Dtype> {
    /// The tile's data.
    pub tensor: T::Tensor<D>,
    /// Boundary mask; `None` when every lane is known in-bounds.
    pub mask: Option<T::BoolTensor>,
}
