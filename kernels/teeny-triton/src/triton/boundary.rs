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

// Per-axis boundary-fold mode, selected as a *type* parameter rather than a
// runtime/const enum value (teenyc-3af.3: this backend has no support for
// reading a custom enum's discriminant at all -- see that issue). Each mode
// is a distinct zero-field marker type implementing `BoundaryFold`, so a
// kernel picks its mode by naming the type at the call site (a bare
// `B: BoundaryFold<D>` generic on the kernel function) -- fully resolved at
// monomorphization, no enum/branch/discriminant of any kind involved.
//
// `T: Triton` is a generic on `default_value` itself, not on `BoundaryFold`
// or its impls: `#[kernel]`/`#[tiled_kernel]` generates the host-callable
// struct parameterized only over the kernel fn's *non-hardware* type params
// (see `macros/teeny-macros/src/macros/{kernel,tiled_kernel}.rs`'s
// `struct_gen_params`, which filters the `Triton`-bound type param out) --
// a trait bound written as `BoundaryFold<T, D>` would reference `T` in a
// scope where the generated struct definition never brings it into scope.
// Keeping `T` off the trait/impls entirely sidesteps that.
//
// Keep this file free of derive, //! docs, and host-only APIs -- same
// constraint as `tile.rs`/`ptr.rs`, since kernel bodies (and, for a type
// parameter like this, the generated entry point's monomorphizing call
// site) are spliced into the no_core device-side source string built from
// this whole `triton/` directory (see `teeny-triton/build.rs`).

use super::Triton;
use super::types::{self as ty};

/// How an out-of-range coordinate on one axis is resolved. Only the
/// *default value* substituted for an out-of-bounds load is modeled here --
/// not index-folding (a real `Clamp`/`Reflect`/`Wrap` would also need to
/// remap the load's address, not just its default), which is future work.
/// Bound by `Num`, not the more general `Dtype`, purely so `Clamp` below can
/// use `+` on the resulting tensor -- narrow the bound further per-impl if a
/// future mode needs it.
pub trait BoundaryFold<D: ty::Num> {
    /// The value substituted for a masked-out (out-of-range) lane of a
    /// `shape`-shaped tile.
    fn default_value<T: Triton>(shape: &[i32]) -> T::Tensor<D>;
}

/// Out-of-range reads contribute nothing (today's conv2d/pool zero-padding).
pub struct Zero;

impl<D: ty::Num> BoundaryFold<D> for Zero {
    #[inline(always)]
    fn default_value<T: Triton>(shape: &[i32]) -> T::Tensor<D> {
        T::zeros::<D>(shape)
    }
}

/// Placeholder second mode, structurally distinct from `Zero` (not a
/// semantically faithful `Clamp` -- real coordinate-clamping needs to fold
/// the load address too) -- enough to prove two different `BoundaryFold`
/// impls produce two genuinely different monomorphized kernel bodies.
pub struct Clamp;

impl<D: ty::Num> BoundaryFold<D> for Clamp {
    #[inline(always)]
    fn default_value<T: Triton>(shape: &[i32]) -> T::Tensor<D> {
        T::zeros::<D>(shape) + T::zeros::<D>(shape)
    }
}
