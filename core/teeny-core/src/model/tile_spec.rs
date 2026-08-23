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

//! [`KernelTileSpec`] — declarative, per-kernel tile-shape metadata consumed
//! by `teeny-kernels`' `TileGraph::propagate` (Welder §3.1's `Propagate`,
//! OSDI'23).
//!
//! This is a revival of `kernels/teeny-triton/src/tile.rs`'s
//! `KernelTileSpec`/`TensorTileSpec`/`TileAxisBinding` (deleted at
//! `84ca6eedf^` alongside the separate `#[tile(...)]` attribute macro that
//! *also* auto-generated index arithmetic into the compiled kernel body —
//! that codegen coupling, not this metadata, is what made the original hard
//! to keep: it broke composability when a kernel is called as a tile-op
//! from inside another kernel's body. This revival is deliberately
//! metadata-only: a spec is hand-authored `const` data describing a
//! kernel's tensors and axes, consumed purely for scheduling analysis
//! (`TileGraph::propagate`/`mem_traffic`/`mem_footprint`), and never drives
//! what gets generated into a kernel's source.
//!
//! Coverage is opt-in per kernel, same as the original — most ops simply
//! have no [`KernelTileSpec`] ([`ExecutableOp::tile_spec`] defaults to
//! `None`), and `TileGraph::propagate` treats that as a hard boundary
//! rather than guessing.
//!
//! ## Propagation is name-matching, not expression evaluation
//!
//! Every axis declares an `extent_param` name. Two axes anywhere — on the
//! same tensor, on different tensors, on an input or an output — that
//! declare the same `extent_param` name are the same free variable. Seeding
//! one node's *output* tile axis values by name resolves every other axis
//! sharing those names for free: e.g. a GEMM kernel's `a_ptr` and `c_ptr`
//! both declaring an axis named `"M"` means propagating `c_ptr`'s chosen
//! `M` automatically resolves `a_ptr`'s `M` too, with no arithmetic
//! involved. An axis whose name never appears in the output (GEMM's
//! reduction axis `"K"`) is *correctly* left unresolved by this mechanism —
//! that's not a gap, it matches Welder's own model where a reduction axis's
//! tile size is chosen independently by the tiling search, not propagated
//! top-down from the output shape.
//!
//! [`TileWindow`] is carried over from the original for fidelity but is
//! **not yet consumed by `TileGraph::propagate`** — the original never
//! actually wired it into `propagate_within_kernel` either (it was only
//! read by the separate `mem_traffic` estimator, given an already-resolved
//! `resolve: impl FnMut(&str) -> i64` supplied entirely by the caller).
//! Teaching `propagate` to invert a windowed axis's extent from its driving
//! output axis is a real follow-up, not a claimed capability here.
//!
//! [`TensorTileSpec::untiled_dims`] is, by contrast, already effectively
//! honored: `TileGraph::propagate` builds every input/output tile at that
//! tensor's full `rank`, not `axes.len()`, so a dim named in
//! `untiled_dims` (no [`TileAxisBinding`] at all) falls back to its real
//! full extent rather than being dropped — the same fallback an axis with
//! no output-side name match (e.g. a reduction axis) already got. The
//! string names in `untiled_dims` itself still aren't read; the effect
//! comes from simply omitting a dim from `axes` (teenygrad-1nr.7).

/// Strided/padded window relating an axis's *output* tile to the actual
/// *input* positions it reads — e.g. a conv kernel's `x_ptr`, whose
/// per-output-tile input region is `(block-1)*stride + kernel_size`
/// elements wide, not simply `block`. See the module doc comment for why
/// `TileGraph::propagate` doesn't resolve this yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileWindow {
    /// Name of the `const {NAME}: i32` generic giving this axis's stride.
    pub stride_const: &'static str,
    /// Name of the `const {NAME}: i32` generic giving this axis's symmetric
    /// padding (applied equally on both sides).
    pub pad_const: &'static str,
    /// Name of the `const {NAME}: i32` generic giving this axis's kernel
    /// (receptive-field) size.
    pub kernel_size_const: &'static str,
}

/// One tensor axis's tile binding: which compile-time block size and
/// runtime extent parameter it's sliced by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileAxisBinding {
    /// Axis indices within the tensor's total shape that this one
    /// binding's block spans, outermost to innermost. `TileGraph::propagate`
    /// indexes by these directly (not by an axis's position within
    /// `axes`), so entries needn't be listed in tensor-dim order — but
    /// every index must be `< rank`, and no index may repeat across one
    /// `TensorTileSpec`'s `axes` (an out-of-range index is silently
    /// skipped; a repeat is last-write-wins; both are a spec-authoring
    /// bug, not something `propagate` tries to normalize).
    ///
    /// Almost always one entry (`&[dim]`) — the ordinary case of one
    /// block const tiling one real axis, e.g. GEMM's `BLOCK_M` tiling
    /// `a_ptr`'s axis 0. More than one entry means this binding's block
    /// const spans a *flattened* combination of several real axes, e.g.
    /// a kernel that iterates a combined `H*W` range in one loop with one
    /// `BLOCK_HW`: `dims: &[h_axis, w_axis]`. `propagate` resolves the
    /// actual block-sized value onto the *last* (innermost) entry and
    /// sets every other entry to a bare `1` — product-preserving (the
    /// tile's total element count still matches `BLOCK_HW`), not a
    /// literal axis-aligned subregion once the block size doesn't evenly
    /// divide the innermost axis's extent, matching this codebase's
    /// existing masked/partial-last-tile simplifications elsewhere (see
    /// `TileGraph::enumerate_subtiles`'s doc comment).
    pub dims: &'static [usize],
    /// Name of the `const {NAME}: i32` generic providing this axis's tile
    /// size.
    pub block_const: &'static str,
    /// Name of the `{NAME}: i32` kernel parameter providing this axis's
    /// total extent. Shared across tensors/ops to mean "the same free
    /// variable" — see the module doc comment.
    pub extent_param: &'static str,
    /// `Some` when this axis is read through a strided/padded sliding
    /// window rather than a plain contiguous slice.
    pub window: Option<TileWindow>,
}

/// Tile-shape metadata for one tensor (a pointer parameter, in the original
/// Triton-kernel sense).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorTileSpec {
    /// The parameter's name, e.g. `"x_ptr"`.
    pub param: &'static str,
    /// The tensor's real, full rank -- how many dims `TileGraph::propagate`
    /// expects on the actual graph edge/output tile for this tensor.
    /// `axes` may cover *fewer* dims than this (see `untiled_dims`): any
    /// dim with no `TileAxisBinding` keeps its full extent when
    /// `propagate` resolves this tensor's tile, the same fallback an axis
    /// with no output-side counterpart (e.g. a reduction axis) already
    /// gets.
    pub rank: usize,
    /// Per-axis tile bindings. Order doesn't matter -- each binding names
    /// its own `dim` -- and this may have fewer entries than `rank` (see
    /// `untiled_dims`).
    pub axes: &'static [TileAxisBinding],
    /// Axis index this tensor is reduced/accumulated over, if any.
    pub reduction_axis: Option<usize>,
    /// Names of `{NAME}: i32` params giving the sizes of this tensor's
    /// *other* real dimensions — present in memory, but not individually
    /// tiled by an axis binding (e.g. a conv kernel's output is tagged on
    /// its width axis alone; batch/channels/height are real but
    /// grid-driven, so they belong here, not in `axes`). Purely
    /// documentation today -- not read by `TileGraph::propagate` -- but
    /// the *effect* it describes (an untiled dim keeps its full extent
    /// rather than being dropped) is what `propagate` actually does,
    /// structurally, via `rank`/`axes` above; naming a dim here vs. simply
    /// omitting it from `axes` has no functional difference yet.
    pub untiled_dims: &'static [&'static str],
}

/// Tile-shape metadata for a kernel's full tensor set — the queryable
/// metadata `TileGraph::propagate` (and, for windowed tensors once that's
/// wired up, `mem_traffic`-style cost estimation) consumes.
///
/// `TileLoopSpec`/`TileCarryBinding` (flash-attention-style loop-carried
/// accumulator state, present in the original) are deliberately not
/// revived here — no consumer needs them yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelTileSpec {
    /// Tile specs for this kernel's input tensors, in the same order as
    /// its declared operands (positional correspondence with a
    /// `TileGraph` node's own parent edges — see
    /// `TileGraph::propagate`'s doc comment for the same limitation the
    /// original `propagate_graph` had).
    pub inputs: &'static [TensorTileSpec],
    /// Tile specs for this kernel's output tensor(s). An in-place
    /// (input-and-output) tensor appears in both `inputs` and `outputs`.
    /// `TileGraph::propagate` uses `outputs[0]` — every `TileOp` is
    /// single-output today.
    pub outputs: &'static [TensorTileSpec],
}
