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
//! metadata-only: a spec is data describing a kernel's tensors and axes —
//! either hand-authored `const`s at the `TritonLowering` construction site
//! (most kernels today), or, for a kernel whose `In<Tile<..>>`/
//! `Out<Tile<..>>` parameters all carry an explicit
//! `#[tile(block=..,extent=..)]` (teenygrad-1nr.18), derived by
//! `#[tiled_kernel]`'s generated `tile_spec()` method from that same
//! attribute instead — consumed purely for scheduling analysis
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

use alloc::format;
use alloc::string::ToString;

use crate::errors::{Error, Result};

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
    /// `Some(g)` when this axis's real, usable per-tile extent is its
    /// full extent divided evenly by a compile-time constant `g` not
    /// otherwise named anywhere (e.g. a GroupNorm-style kernel whose
    /// per-CTA channel slice is `channels_per_group = C / G` -- `G` isn't
    /// a graph-level shape dimension or an `extent_param` any other axis
    /// shares, just an internal kernel constant). `TileGraph::propagate`
    /// applies this uniformly wherever it would otherwise use this
    /// axis's raw full extent -- both resolving `resolved` from the
    /// output side and falling back on the input side when this axis's
    /// `extent_param` isn't otherwise resolved -- so every consumer of
    /// this axis's name sees the already-divided value. `None` (the
    /// ordinary case) leaves the axis's extent as-is.
    ///
    /// A plain numeric divisor, not a named parameter to look up: unlike
    /// `extent_param`/`block_const` (names resolved by matching, or left
    /// as documentation), the divisor's actual value has to be known at
    /// `KernelTileSpec`-construction time -- there's no per-node
    /// parameter-value lookup elsewhere in this metadata. A real op
    /// whose divisor varies per instance (GroupNorm's `num_groups`) must
    /// build its `tile_spec()` fresh per call using its own instance
    /// data, not return one shared `const`.
    pub divide_by: Option<usize>,
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

/// One loop-carried accumulator variable threaded through a kernel's
/// reduction/accumulation loop (e.g. `conv2d_forward`'s `acc: [BLOCK_OW]`,
/// accumulated over its `(C_IN/G)*KH*KW`-iteration receptive-field loop;
/// flash-attention's online-softmax `acc`/`m_i`/`l_i` is the same shape).
/// Not representable as a [`TensorTileSpec`]: these aren't tiles of a
/// pointer parameter sliced by `(block, extent)`, they're kernel-body-local
/// tensors whose *shape* is fixed by const generics (not grid-varying) and
/// whose *value* the loop updates in place.
///
/// Revived (teenygrad-1nr.18) from the pre-revival design this module is
/// already a revival of (`kernels/teeny-triton/src/tile.rs` at
/// `84ca6eedf^`) — see teenygrad-1nr.12, whose investigation found the
/// original design declared this metadata but never grew a consumer for
/// it. That remains true here: this records shape and identity only, not
/// the per-iteration combine expression, and `TileGraph::propagate`/
/// `mem_traffic`/`mem_footprint` still treat a loop-carried kernel's real
/// inputs as fully materialized, exactly like any other `tile_spec`-less
/// op — declaring `loop_spec` does not by itself fix that; it only makes
/// the loop's existence and carried shape queryable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCarryBinding {
    /// The variable's name in the kernel body, e.g. `"acc"`.
    pub name: &'static str,
    /// Names of the `const {NAME}: i32` generics giving the carried
    /// tensor's shape, in dimension order (e.g. conv2d's `["BLOCK_OW"]`).
    pub shape_consts: &'static [&'static str],
}

/// Loop-carry metadata for a kernel's reduction/accumulation loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileLoopSpec {
    /// Variables carried across the loop's iterations.
    pub carries: &'static [TileCarryBinding],
    /// Names of the `{NAME}: i32` params / `const {NAME}: i32` generics
    /// whose values determine this loop's trip count. Documentation only,
    /// like [`TensorTileSpec::untiled_dims`] — not a literal formula to
    /// evaluate: e.g. conv2d's real trip count is `(C_IN/G)*KH*KW`, a
    /// runtime param (`C_IN`) combined with three consts, so this names
    /// `["C_IN", "G", "KH", "KW"]` rather than one single param the way a
    /// flash-attention-style fixed-trip-count loop's would be `["n_ctx_k"]`
    /// alone. No consumer computes the actual trip count from these yet.
    pub trip_count_factors: &'static [&'static str],
}

/// Tile-shape metadata for a kernel's full tensor set — the queryable
/// metadata `TileGraph::propagate` (and, for windowed tensors once that's
/// wired up, `mem_traffic`-style cost estimation) consumes.
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
    /// `Some` when this kernel's body loops with carried accumulator
    /// state (see [`TileLoopSpec`]) instead of resolving its whole output
    /// in one shot. `None` (the ordinary case) for every non-looping spec.
    pub loop_spec: Option<TileLoopSpec>,
}

impl TileAxisBinding {
    /// Checks this binding in isolation, for a tensor of rank `rank`.
    ///
    /// `param` names the tensor only so the error can say which one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTileSpec`] for an empty `dims`, an index at
    /// or beyond `rank`, an empty name, or a zero `divide_by`.
    fn validate(&self, param: &str, rank: usize) -> Result<()> {
        let bad = |problem: alloc::string::String| Error::InvalidTileSpec {
            param: param.to_string(),
            problem,
        };

        if self.dims.is_empty() {
            return Err(bad(format!(
                "axis `{}` binds no dims; a binding must cover at least one",
                self.block_const
            ))
            .into());
        }
        for &dim in self.dims {
            if dim >= rank {
                return Err(bad(format!(
                    "axis `{}` binds dim {dim}, but the tensor is rank {rank}",
                    self.block_const
                ))
                .into());
            }
        }
        if self.block_const.is_empty() || self.extent_param.is_empty() {
            return Err(bad(format!(
                "axis binding on dims {:?} has an empty block const or extent param",
                self.dims
            ))
            .into());
        }
        if let Some(window) = self.window
            && (window.stride_const.is_empty()
                || window.pad_const.is_empty()
                || window.kernel_size_const.is_empty())
        {
            return Err(bad(format!(
                "axis `{}` has a window with an empty const name",
                self.block_const
            ))
            .into());
        }
        if self.divide_by == Some(0) {
            return Err(bad(format!(
                "axis `{}` has `divide_by: Some(0)`, which would divide its extent by zero",
                self.block_const
            ))
            .into());
        }
        Ok(())
    }
}

impl TensorTileSpec {
    /// Checks this tensor's bindings against its own `rank`.
    ///
    /// The invariants are the ones [`TileAxisBinding::dims`] states in
    /// prose: every index below `rank`, and no index repeated across this
    /// tensor's `axes`. Both are documented as authoring bugs that
    /// `propagate` does not normalise — an out-of-range index is skipped
    /// and a repeat is last-write-wins — so neither shows up at the point
    /// the spec is written.
    ///
    /// `untiled_dims` is documentation, and deliberately need not be
    /// complete: `CONV1D_TILE_SPEC`'s input leaves every dim out of both
    /// lists. So this checks only that the two do not *contradict* each
    /// other, and that together they do not describe more dims than exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTileSpec`] naming the tensor and the
    /// offending index or name.
    pub fn validate(&self) -> Result<()> {
        let bad = |problem: alloc::string::String| Error::InvalidTileSpec {
            param: self.param.to_string(),
            problem,
        };

        if self.param.is_empty() {
            return Err(bad("tensor has an empty param name".to_string()).into());
        }

        let mut seen: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
        for axis in self.axes {
            axis.validate(self.param, self.rank)?;
            for &dim in axis.dims {
                if seen.contains(&dim) {
                    return Err(bad(format!(
                        "dim {dim} is bound twice; `propagate` resolves repeats \
                         last-write-wins, so one of the two bindings is dead"
                    ))
                    .into());
                }
                seen.push(dim);
            }
        }

        if let Some(axis) = self.reduction_axis
            && axis >= self.rank
        {
            return Err(bad(format!(
                "reduction_axis is {axis}, but the tensor is rank {}",
                self.rank
            ))
            .into());
        }

        for name in self.untiled_dims {
            if let Some(axis) = self.axes.iter().find(|a| a.extent_param == *name) {
                return Err(bad(format!(
                    "`{name}` is listed in untiled_dims but is also the extent param \
                     of the axis bound to dims {:?}",
                    axis.dims
                ))
                .into());
            }
        }

        let described = seen.len() + self.untiled_dims.len();
        if described > self.rank {
            return Err(bad(format!(
                "{} tiled dims plus {} untiled names describe {described} dims, \
                 but the tensor is rank {}",
                seen.len(),
                self.untiled_dims.len(),
                self.rank
            ))
            .into());
        }

        Ok(())
    }
}

impl KernelTileSpec {
    /// Checks every tensor in this spec, and the loop metadata.
    ///
    /// Written to be called from a test that enumerates the registered
    /// specs. A `const fn` version would have to panic with a static
    /// string rather than name the offending tensor and index, since
    /// formatting needs an allocator — the diagnosis is worth more here
    /// than the compile-time check.
    ///
    /// # Errors
    ///
    /// Returns the first [`Error::InvalidTileSpec`] found, naming the
    /// tensor it came from.
    pub fn validate(&self) -> Result<()> {
        if self.outputs.is_empty() {
            return Err(Error::InvalidTileSpec {
                param: "<kernel>".to_string(),
                problem: "spec declares no outputs, but `propagate` reads `outputs[0]`".to_string(),
            }
            .into());
        }

        for tensor in self.inputs.iter().chain(self.outputs) {
            tensor.validate()?;
        }

        if let Some(loop_spec) = self.loop_spec {
            for carry in loop_spec.carries {
                if carry.name.is_empty() || carry.shape_consts.is_empty() {
                    return Err(Error::InvalidTileSpec {
                        param: "<loop_spec>".to_string(),
                        problem: format!(
                            "carry `{}` has an empty name or no shape consts",
                            carry.name
                        ),
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    const fn axis(dims: &'static [usize], block: &'static str) -> TileAxisBinding {
        TileAxisBinding {
            dims,
            block_const: block,
            extent_param: "N",
            window: None,
            divide_by: None,
        }
    }

    fn tensor(rank: usize, axes: &'static [TileAxisBinding]) -> TensorTileSpec {
        TensorTileSpec {
            param: "x_ptr",
            rank,
            axes,
            reduction_axis: None,
            untiled_dims: &[],
        }
    }

    fn err(spec: &TensorTileSpec) -> String {
        alloc::format!("{}", spec.validate().expect_err("spec should be rejected"))
    }

    #[test]
    fn a_well_formed_tensor_validates() {
        const AXES: &[TileAxisBinding] = &[axis(&[0], "BLOCK_M"), axis(&[1], "BLOCK_K")];
        assert!(tensor(2, AXES).validate().is_ok());
    }

    /// The flattened multi-dim case (`BATCHNORM2D_TILE_SPEC`'s `HW`).
    #[test]
    fn one_binding_may_span_several_dims() {
        const AXES: &[TileAxisBinding] = &[axis(&[2, 3], "BLOCK_HW")];
        let spec = TensorTileSpec {
            param: "x_ptr",
            rank: 4,
            axes: AXES,
            reduction_axis: None,
            untiled_dims: &["B", "C"],
        };
        assert!(spec.validate().is_ok());
    }

    /// `CONV1D_TILE_SPEC`'s input describes no dims at all — untiled_dims is
    /// documentation and need not be complete.
    #[test]
    fn a_tensor_describing_none_of_its_dims_is_allowed() {
        assert!(tensor(3, &[]).validate().is_ok());
    }

    #[test]
    fn a_dim_at_or_beyond_rank_is_rejected() {
        const AXES: &[TileAxisBinding] = &[axis(&[2], "BLOCK_OOB")];
        let msg = err(&tensor(2, AXES));
        assert!(msg.contains("binds dim 2"), "{msg}");
        assert!(msg.contains("rank 2"), "{msg}");
        assert!(msg.contains("x_ptr"), "{msg}");
    }

    #[test]
    fn binding_the_same_dim_twice_is_rejected() {
        const AXES: &[TileAxisBinding] = &[axis(&[0], "BLOCK_A"), axis(&[0], "BLOCK_B")];
        let msg = err(&tensor(2, AXES));
        assert!(msg.contains("dim 0 is bound twice"), "{msg}");
    }

    /// Across bindings, not just within one.
    #[test]
    fn a_dim_repeated_across_a_flattened_binding_is_rejected() {
        const AXES: &[TileAxisBinding] = &[axis(&[1, 2], "BLOCK_HW"), axis(&[2], "BLOCK_W")];
        let msg = err(&tensor(4, AXES));
        assert!(msg.contains("dim 2 is bound twice"), "{msg}");
    }

    #[test]
    fn a_binding_with_no_dims_is_rejected() {
        const AXES: &[TileAxisBinding] = &[axis(&[], "BLOCK_NONE")];
        let msg = err(&tensor(2, AXES));
        assert!(msg.contains("binds no dims"), "{msg}");
    }

    #[test]
    fn an_out_of_range_reduction_axis_is_rejected() {
        let spec = TensorTileSpec {
            param: "a_ptr",
            rank: 2,
            axes: &[],
            reduction_axis: Some(5),
            untiled_dims: &[],
        };
        let msg = alloc::format!("{}", spec.validate().expect_err("out of range"));
        assert!(msg.contains("reduction_axis is 5"), "{msg}");
    }

    #[test]
    fn divide_by_zero_is_rejected() {
        const AXES: &[TileAxisBinding] = &[TileAxisBinding {
            dims: &[0],
            block_const: "BLOCK_C",
            extent_param: "C",
            window: None,
            divide_by: Some(0),
        }];
        let msg = err(&tensor(1, AXES));
        assert!(msg.contains("divide_by"), "{msg}");
    }

    /// A dim cannot be both tiled and declared untiled.
    #[test]
    fn an_extent_param_in_untiled_dims_is_rejected() {
        const AXES: &[TileAxisBinding] = &[axis(&[0], "BLOCK_N")];
        let spec = TensorTileSpec {
            param: "x_ptr",
            rank: 2,
            axes: AXES,
            reduction_axis: None,
            untiled_dims: &["N"],
        };
        let msg = alloc::format!("{}", spec.validate().expect_err("contradiction"));
        assert!(msg.contains("untiled_dims"), "{msg}");
    }

    #[test]
    fn describing_more_dims_than_the_rank_is_rejected() {
        const AXES: &[TileAxisBinding] = &[axis(&[0], "BLOCK_N")];
        let spec = TensorTileSpec {
            param: "x_ptr",
            rank: 2,
            axes: AXES,
            reduction_axis: None,
            untiled_dims: &["B", "C"],
        };
        let msg = alloc::format!("{}", spec.validate().expect_err("over-described"));
        assert!(msg.contains("rank 2"), "{msg}");
    }

    #[test]
    fn a_kernel_spec_with_no_outputs_is_rejected() {
        const SPEC: KernelTileSpec = KernelTileSpec {
            inputs: &[],
            outputs: &[],
            loop_spec: None,
        };
        let msg = alloc::format!("{}", SPEC.validate().expect_err("no outputs"));
        assert!(msg.contains("outputs[0]"), "{msg}");
    }

    #[test]
    fn a_kernel_spec_checks_every_tensor_not_just_the_first() {
        const GOOD: &[TileAxisBinding] = &[axis(&[0], "BLOCK_M")];
        const BAD: &[TileAxisBinding] = &[axis(&[9], "BLOCK_OOB")];
        const SPEC: KernelTileSpec = KernelTileSpec {
            inputs: &[TensorTileSpec {
                param: "a_ptr",
                rank: 2,
                axes: GOOD,
                reduction_axis: None,
                untiled_dims: &[],
            }],
            outputs: &[TensorTileSpec {
                param: "c_ptr",
                rank: 2,
                axes: BAD,
                reduction_axis: None,
                untiled_dims: &[],
            }],
            loop_spec: None,
        };
        let msg = alloc::format!("{}", SPEC.validate().expect_err("bad output tensor"));
        assert!(msg.contains("c_ptr"), "{msg}");
    }

    #[test]
    fn a_carry_with_no_shape_consts_is_rejected() {
        const SPEC: KernelTileSpec = KernelTileSpec {
            inputs: &[],
            outputs: &[TensorTileSpec {
                param: "y_ptr",
                rank: 1,
                axes: &[],
                reduction_axis: None,
                untiled_dims: &[],
            }],
            loop_spec: Some(TileLoopSpec {
                carries: &[TileCarryBinding {
                    name: "acc",
                    shape_consts: &[],
                }],
                trip_count_factors: &["N"],
            }),
        };
        let msg = alloc::format!("{}", SPEC.validate().expect_err("empty carry"));
        assert!(msg.contains("acc"), "{msg}");
    }
}
