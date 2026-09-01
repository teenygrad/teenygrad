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

//! [`GridSpec`] — declarative metadata for a kernel's launch-grid
//! decomposition (teenygrad-1nr.17/teenygrad-1nr.18): which real
//! hardware grid dimension each logical axis reads from, and whether
//! that axis is block-tiled or "one CTA per index".
//!
//! Sibling to [`super::tile_spec::KernelTileSpec`], not a field on it:
//! tile shape and grid decomposition are different concerns (tensor
//! tiling vs. launch configuration) that happen to be generated from the
//! same `#[tile(...)]`-tagged parameter by `#[tiled_kernel]` — see that
//! macro's own doc comment for why keeping them as two independently
//! consumed types, generated from one declaration, is deliberate: it
//! keeps them provably consistent without coupling their consumers.
//!
//! Exists to answer a narrower question than Welder's own mechanism does.
//! Welder derives a fused kernel's blockIdx/threadIdx remapping by
//! walking each op's already-lowered TIR ("deduced from their tensor
//! expressions") — a symbolic-IR capability this codebase doesn't have.
//! [`GridSpec`] instead lets that remapping be *checked* mechanically for
//! the case that's actually common and was already verified by hand
//! (teenygrad-1nr.17): one fused op has a real grid of its own, and every
//! other fused op's grid axes are a name-matching subset of it (so it
//! can ride along on the anchor's already-decoded values, needing no
//! grid of its own — the "register-level fusion" case). Two grids that
//! *don't* share axes this way still need real remapping or must be
//! rejected; nothing here solves that harder case.

/// Which real hardware grid dimension (CUDA `blockIdx.x`/`.y`/`.z`) one
/// [`GridAxisBinding`] reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GridDim {
    /// `blockIdx.x`.
    X,
    /// `blockIdx.y`.
    Y,
    /// `blockIdx.z`.
    Z,
}

/// One logical axis of a kernel's launch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridAxisBinding {
    /// This axis's identity, shared across kernels that iterate "the
    /// same" logical axis (e.g. `"B"`, `"C_OUT"`) — two kernels'
    /// [`GridAxisBinding`]s sharing a name are the same free variable,
    /// mirroring [`super::TileAxisBinding::extent_param`]'s own
    /// name-matching convention.
    pub name: &'static str,
    /// Names of the `{NAME}: i32` params / `const {NAME}: i32` generics
    /// this axis's extent depends on — documentation only, like
    /// [`super::TileLoopSpec::trip_count_factors`], not a literal
    /// formula: a tiled axis's real extent is `cdiv(extent, block)`, so
    /// this names both; an untiled axis names just its own full-extent
    /// param.
    pub extent_factors: &'static [&'static str],
    /// Which real hardware grid dimension this axis is read from.
    /// Multiple axes sharing a `dim` are packed via mixed-radix decode
    /// (`pid % extent`, `pid / extent`, repeat) in the order they're
    /// declared in [`GridSpec::axes`] — outermost (divided last) first.
    pub dim: GridDim,
    /// `Some(block_const)` when this axis is block-tiled: one CTA covers
    /// `block_const` elements, and the axis's real grid extent is
    /// `cdiv(full_extent, block_const)`. `None` for an axis with no
    /// tiling of its own — one CTA per index (e.g. a batch/channel
    /// dimension a kernel doesn't subdivide).
    pub block_const: Option<&'static str>,
}

/// A kernel's launch-grid decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridSpec {
    /// This kernel's grid axes. Order matters: outermost to innermost
    /// *within each [`GridDim`]* — see [`GridAxisBinding::dim`].
    pub axes: &'static [GridAxisBinding],
}
