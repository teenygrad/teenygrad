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

//! The per-node half of Welder's `Propagate`, as a pure function.
//!
//! `TileGraph::propagate` is two things welded together: a walk over the
//! graph, and — at each node — the resolution of one [`KernelTileSpec`]
//! against one output tile. This module is the second half. It needs no
//! graph, no scheduler and no hardware, so a spec is testable the moment it
//! is written rather than only once the scheduler is re-landed
//! (`teenygrad-1nr.25`).
//!
//! It deliberately lives outside `teeny-core`, where the spec *types* live:
//! those have to be there because `ExecutableOp::tile_spec` is a trait
//! method and `teeny-core` holds `Dag<Box<dyn ExecutableOp>>`, but
//! tile-resolution arithmetic is scheduler semantics and belongs beside the
//! scheduler. It also lives outside `graph::optimizer::anduin`, so that
//! testing it does not drag in the optimizer.
//!
//! ## Resolution is name-matching
//!
//! Per [`KernelTileSpec`]'s own contract, two axes declaring the same
//! `extent_param` are the same free variable. Resolving an output tile
//! therefore means reading the blocks the output's axes were given, keying
//! them by name, and looking every input axis up in that map. An axis whose
//! name never appears in the output — a GEMM's reduction axis `K` — is
//! correctly left unresolved; that is not a gap.
//!
//! ## Unknown means full extent
//!
//! A `None` in a [`TileShape`] means "not resolved here", which every
//! consumer reads as the axis's full extent. That is the conservative
//! direction: an over-estimate is safe for a footprint checked against a
//! hard capacity, an under-estimate is not. Every case this module cannot
//! compute — an unmatched name, a window whose consts the caller did not
//! supply — yields `None` rather than a guess.

use teeny_core::model::{KernelTileSpec, TensorTileSpec};

use crate::errors::{Error, Result};

/// One tensor's per-axis tile extents, `None` where an axis is unresolved
/// and therefore at its full extent.
///
/// Mirrors `teeny_core::graph::Shape`'s idiom rather than the scheduler's
/// own `TileDim`: `TileDim` returns with `teenygrad-1nr.25`, and depending
/// on it would block this on the work it is meant to run ahead of.
/// Reconciling the two is a design point for that issue.
pub type TileShape = Vec<Option<usize>>;

/// Resolves the constant-valued names a spec refers to — block sizes,
/// extents, and a window's stride/padding/kernel consts.
///
/// A spec names these rather than carrying their values, because the
/// metadata has no per-node parameter lookup of its own. The caller owns
/// that lookup, the same way v1's `mem_traffic` took a caller-supplied
/// resolver. Returning `None` for a name is always allowed and always
/// safe — the affected axis falls back to its full extent.
pub trait ConstLookup {
    /// The value of `name`, or `None` if the caller cannot supply it.
    fn get(&self, name: &str) -> Option<usize>;
}

impl<F: Fn(&str) -> Option<usize>> ConstLookup for F {
    fn get(&self, name: &str) -> Option<usize> {
        self(name)
    }
}

/// A [`ConstLookup`] that knows nothing, so every axis it is asked about
/// falls back to its full extent.
pub struct NoConsts;

impl ConstLookup for NoConsts {
    fn get(&self, _name: &str) -> Option<usize> {
        None
    }
}

/// Recovers the input tile each operand must supply, given the tile this
/// kernel's first output was assigned.
///
/// Returns one [`TileShape`] per entry in `spec.inputs`, in declaration
/// order, each at that tensor's declared `rank`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when `spec` declares no outputs, when
/// `output_tile`'s rank disagrees with the spec's first output, when a
/// binding indexes past its tensor's rank, or when a windowed axis is given
/// a zero block.
pub fn resolve_inputs(
    spec: &KernelTileSpec,
    output_tile: &TileShape,
    consts: &impl ConstLookup,
) -> Result<Vec<TileShape>> {
    let output = *spec.outputs.first().ok_or_else(|| {
        Error::InvalidArgument(
            "tile spec declares no outputs, so there is no output tile to resolve against"
                .to_string(),
        )
    })?;

    if output_tile.len() != output.rank {
        return Err(Error::InvalidArgument(format!(
            "output tile has rank {}, but `{}` declares rank {}",
            output_tile.len(),
            output.param,
            output.rank
        ))
        .into());
    }

    // Blocks the output was given, keyed by the free variable they name.
    let mut resolved: Vec<(&'static str, usize)> = Vec::new();
    for axis in output.axes {
        let mut block = Some(1usize);
        for &dim in axis.dims {
            let extent = *output_tile.get(dim).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "`{}` axis `{}` binds dim {dim}, past its rank {}",
                    output.param, axis.block_const, output.rank
                ))
            })?;
            // A flattened binding spreads its block across several dims,
            // all but the innermost being 1, so the product is the block.
            block = match (block, extent) {
                (Some(acc), Some(e)) => Some(acc * e),
                _ => None,
            };
        }
        if let Some(block) = block {
            resolved.push((axis.extent_param, block));
        }
    }

    spec.inputs
        .iter()
        .map(|input| resolve_one(input, &resolved, consts))
        .collect()
}

/// Builds one input tensor's tile. Every axis starts unresolved, and only
/// an axis this spec can actually place is written.
fn resolve_one(
    input: &TensorTileSpec,
    resolved: &[(&'static str, usize)],
    consts: &impl ConstLookup,
) -> Result<TileShape> {
    let mut tile: TileShape = vec![None; input.rank];

    for axis in input.axes {
        for &dim in axis.dims {
            if dim >= input.rank {
                return Err(Error::InvalidArgument(format!(
                    "`{}` axis `{}` binds dim {dim}, past its rank {}",
                    input.param, axis.block_const, input.rank
                ))
                .into());
            }
        }

        // Either the output propagated a block for this free variable, or
        // the caller can supply the axis's full extent — in which case
        // `divide_by` applies, per its own contract that it replaces
        // wherever this axis's raw full extent would be used.
        let block = match resolved.iter().find(|(name, _)| *name == axis.extent_param) {
            Some((_, block)) => Some(*block),
            None => consts
                .get(axis.extent_param)
                .map(|extent| extent / axis.divide_by.unwrap_or(1).max(1)),
        };

        let Some(block) = block else {
            continue; // Unresolved: leave the axis at its full extent.
        };

        // A windowed axis reads more than its block: the receptive field.
        // Padding shifts the origin, not the extent, so it is not read.
        let extent = match axis.window {
            None => block,
            Some(window) => {
                let (Some(stride), Some(kernel)) = (
                    consts.get(window.stride_const),
                    consts.get(window.kernel_size_const),
                ) else {
                    continue; // Window consts unavailable: full extent.
                };
                let steps = block.checked_sub(1).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "`{}` axis `{}` has a windowed block of 0",
                        input.param, axis.block_const
                    ))
                })?;
                steps * stride + kernel
            }
        };

        // The innermost bound dim carries the extent; the outer dims of a
        // flattened binding collapse to 1, preserving the element count.
        let last = axis.dims.len() - 1;
        for (i, &dim) in axis.dims.iter().enumerate() {
            tile[dim] = Some(if i == last { extent } else { 1 });
        }
    }

    Ok(tile)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use teeny_core::model::{TileAxisBinding, TileWindow};

    use crate::graph::{BATCHNORM2D_TILE_SPEC, CONV1D_TILE_SPEC, MATMUL_TILE_SPEC};

    /// A lookup backed by an explicit table, the way a real caller's
    /// per-node parameter values would be.
    struct Table(HashMap<&'static str, usize>);

    impl Table {
        fn new(pairs: &[(&'static str, usize)]) -> Self {
            Self(pairs.iter().copied().collect())
        }
    }

    impl ConstLookup for Table {
        fn get(&self, name: &str) -> Option<usize> {
            self.0.get(name).copied()
        }
    }

    fn tile(dims: &[Option<usize>]) -> TileShape {
        dims.to_vec()
    }

    // --- the shapes a real spec takes -----------------------------------

    /// GEMM: `M` and `N` propagate from the output by name; `K` never
    /// appears there, so it stays unresolved — full extent — which is what
    /// Welder's model expects of a reduction axis.
    #[test]
    fn test_matmul_propagates_m_and_n_and_leaves_k_unresolved() {
        let inputs =
            resolve_inputs(&MATMUL_TILE_SPEC, &tile(&[Some(64), Some(32)]), &NoConsts).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(
            inputs[0],
            tile(&[Some(64), None]),
            "a_ptr: [M_block, K_full]"
        );
        assert_eq!(
            inputs[1],
            tile(&[None, Some(32)]),
            "b_ptr: [K_full, N_block]"
        );
    }

    /// The flattened case: one `BLOCK_HW` spanning dims 2 and 3. The
    /// innermost dim carries the block and the outer collapses to 1, so the
    /// element count is preserved.
    #[test]
    fn test_batchnorm2d_resolves_its_flattened_binding() {
        let inputs = resolve_inputs(
            &BATCHNORM2D_TILE_SPEC,
            &tile(&[None, None, Some(1), Some(256)]),
            &NoConsts,
        )
        .unwrap();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0], tile(&[None, None, Some(1), Some(256)]));
    }

    /// `CONV1D_TILE_SPEC`'s input declares no axes at all, so every dim is
    /// at full extent — the documented fallback, not a failure.
    #[test]
    fn test_conv1d_input_is_entirely_unresolved() {
        let inputs = resolve_inputs(
            &CONV1D_TILE_SPEC,
            &tile(&[Some(2), Some(8), Some(64)]),
            &NoConsts,
        )
        .unwrap();

        assert_eq!(inputs[0], tile(&[None, None, None]));
    }

    // --- windows ---------------------------------------------------------

    const W: TileWindow = TileWindow {
        stride_const: "STRIDE_W",
        pad_const: "PAD_W",
        kernel_size_const: "KW",
    };

    /// No shipped spec carries a `TileWindow` yet — populating conv/pool is
    /// `teenygrad-1nr.29` — so the window cases use a spec of this shape.
    const WINDOWED: KernelTileSpec = {
        const AXIS: TileAxisBinding = TileAxisBinding {
            dims: &[2],
            block_const: "BLOCK_OW",
            extent_param: "OW",
            window: None,
            divide_by: None,
        };
        KernelTileSpec {
            inputs: &[TensorTileSpec {
                param: "x_ptr",
                rank: 3,
                axes: &[TileAxisBinding {
                    window: Some(W),
                    ..AXIS
                }],
                reduction_axis: None,
                untiled_dims: &[],
            }],
            outputs: &[TensorTileSpec {
                param: "y_ptr",
                rank: 3,
                axes: &[AXIS],
                reduction_axis: None,
                untiled_dims: &[],
            }],
            loop_spec: None,
        }
    };

    /// The receptive field: a block of 8 through a 3-wide kernel at stride
    /// 1 reads 10 elements, not 8.
    #[test]
    fn test_windowed_axis_resolves_to_its_receptive_field() {
        let consts = Table::new(&[("STRIDE_W", 1), ("KW", 3), ("PAD_W", 1)]);
        let inputs =
            resolve_inputs(&WINDOWED, &tile(&[Some(2), Some(16), Some(8)]), &consts).unwrap();

        assert_eq!(inputs[0][2], Some(10), "(8 - 1) * 1 + 3");
    }

    /// Padding shifts the origin, not the extent. This is the assertion
    /// that catches anyone reaching for whole-tensor reverse inference,
    /// whose `- 2p` term would make the answer depend on it.
    #[test]
    fn test_receptive_field_does_not_depend_on_padding() {
        let output = tile(&[Some(2), Some(16), Some(8)]);
        let unpadded = Table::new(&[("STRIDE_W", 2), ("KW", 3), ("PAD_W", 0)]);
        let padded = Table::new(&[("STRIDE_W", 2), ("KW", 3), ("PAD_W", 5)]);

        let a = resolve_inputs(&WINDOWED, &output, &unpadded).unwrap();
        let b = resolve_inputs(&WINDOWED, &output, &padded).unwrap();

        assert_eq!(a[0][2], Some(17), "(8 - 1) * 2 + 3");
        assert_eq!(a, b, "padding must not change the resolved extent");
    }

    /// A caller that cannot supply the window's consts gets the full
    /// extent, which over-estimates. Never a block, which would not.
    #[test]
    fn test_unresolvable_window_falls_back_to_full_extent() {
        let inputs =
            resolve_inputs(&WINDOWED, &tile(&[Some(2), Some(16), Some(8)]), &NoConsts).unwrap();

        assert_eq!(
            inputs[0][2], None,
            "unknown, i.e. full extent — not Some(8)"
        );
    }

    // --- fallbacks and errors --------------------------------------------

    /// An axis the output does not name can still resolve if the caller
    /// knows its extent, and `divide_by` applies there — its contract is to
    /// replace this axis's raw full extent.
    #[test]
    fn test_divide_by_applies_to_a_caller_supplied_full_extent() {
        const IN: &[TensorTileSpec] = &[TensorTileSpec {
            param: "x_ptr",
            rank: 2,
            axes: &[TileAxisBinding {
                dims: &[1],
                block_const: "BLOCK_C",
                extent_param: "C",
                window: None,
                divide_by: Some(4),
            }],
            reduction_axis: None,
            untiled_dims: &[],
        }];
        const OUT: &[TensorTileSpec] = &[TensorTileSpec {
            param: "y_ptr",
            rank: 2,
            axes: &[],
            reduction_axis: None,
            untiled_dims: &[],
        }];
        const SPEC: KernelTileSpec = KernelTileSpec {
            inputs: IN,
            outputs: OUT,
            loop_spec: None,
        };

        let consts = Table::new(&[("C", 64)]);
        let inputs = resolve_inputs(&SPEC, &tile(&[Some(2), Some(8)]), &consts).unwrap();
        assert_eq!(inputs[0][1], Some(16), "64 / 4");
    }

    #[test]
    fn test_output_tile_of_the_wrong_rank_is_rejected() {
        let msg = resolve_inputs(&MATMUL_TILE_SPEC, &tile(&[Some(64)]), &NoConsts)
            .expect_err("rank 1 against a rank-2 output")
            .to_string();
        assert!(msg.contains("rank 1"), "{msg}");
        assert!(
            msg.contains("c_ptr") || msg.contains("declares rank 2"),
            "{msg}"
        );
    }

    #[test]
    fn test_spec_with_no_outputs_is_rejected() {
        const SPEC: KernelTileSpec = KernelTileSpec {
            inputs: &[],
            outputs: &[],
            loop_spec: None,
        };
        let msg = resolve_inputs(&SPEC, &tile(&[]), &NoConsts)
            .expect_err("no outputs")
            .to_string();
        assert!(msg.contains("no outputs"), "{msg}");
    }

    /// An unresolved output axis leaves the matching input axis unresolved
    /// too, rather than inventing a block for it.
    #[test]
    fn test_unresolved_output_axis_does_not_resolve_its_input() {
        let inputs =
            resolve_inputs(&MATMUL_TILE_SPEC, &tile(&[None, Some(32)]), &NoConsts).unwrap();

        assert_eq!(
            inputs[0],
            tile(&[None, None]),
            "M unknown, so a_ptr's M is too"
        );
        assert_eq!(inputs[1], tile(&[None, Some(32)]));
    }
}
