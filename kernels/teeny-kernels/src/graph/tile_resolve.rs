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

/// The value a window's `stride`/`pad`/`kernel` field names: a decimal
/// literal stands for itself, anything else is asked of `consts`.
///
/// The pad family has no stride or kernel const to name — padding is a window
/// of `stride = 1, kernel = 1` — so those fields carry literals, and a
/// `ConstLookup` built from a kernel's real parameters would never contain a
/// key called `"1"` (teenygrad-1tl.5).
fn window_const(name: &str, consts: &impl ConstLookup) -> Option<usize> {
    match name.parse::<usize>() {
        Ok(literal) => Some(literal),
        Err(_) => consts.get(name),
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
        // A windowed axis resolves against the *output* variable its window
        // names, not its own `extent_param`: its own names the real input
        // extent, which never appears in the output (teenygrad-1nr.18.2). An
        // unwindowed axis resolves against its own name as before.
        let propagation_name = match axis.window {
            Some(window) => window.output_extent_param,
            None => axis.extent_param,
        };
        let block = match resolved.iter().find(|(name, _)| *name == propagation_name) {
            Some((_, block)) => Some(*block),
            // The const fallback supplies this axis's *full* extent, which does
            // not compose with a window: feeding `H` through
            // `(block - 1) * stride + kernel` yields more than `H` and claims
            // the kernel reads past the tensor. A windowed axis whose output
            // variable went unresolved therefore stays at full extent, which is
            // what an unresolved axis means everywhere else (teenygrad-1tl.7).
            None if axis.window.is_some() => None,
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
                    window_const(window.stride_const, consts),
                    window_const(window.kernel_size_const, consts),
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

    use teeny_core::model::{TensorTileSpec, TileAxisBinding, TileWindow};

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

    // --- fixtures --------------------------------------------------------
    //
    // These describe the *shapes* a spec can take, not any particular
    // kernel. They used to be the hand-authored `GEMM` and
    // friends from `graph::mod`, which are gone: every spec is derived
    // from its kernel's own `#[tile(...)]` now (`teenygrad-1tl`), and a
    // unit test of this module's arithmetic should not depend on which
    // kernels happen to have been converted yet.

    const fn axis(
        dims: &'static [usize],
        block: &'static str,
        extent: &'static str,
    ) -> TileAxisBinding {
        TileAxisBinding {
            dims,
            block_const: block,
            extent_param: extent,
            window: None,
            divide_by: None,
        }
    }

    const fn tensor(
        param: &'static str,
        rank: usize,
        axes: &'static [TileAxisBinding],
    ) -> TensorTileSpec {
        TensorTileSpec {
            param,
            rank,
            axes,
            reduction_axis: None,
            untiled_dims: &[],
        }
    }

    /// GEMM-shaped: `a_ptr: [M, K]`, `b_ptr: [K, N]`, `c_ptr: [M, N]`.
    /// `M`/`N` are shared with the output; `K` is on both inputs and
    /// neither output.
    const GEMM: KernelTileSpec = {
        const A: &[TileAxisBinding] = &[axis(&[0], "BLOCK_M", "M"), axis(&[1], "BLOCK_K", "K")];
        const B: &[TileAxisBinding] = &[axis(&[0], "BLOCK_K", "K"), axis(&[1], "BLOCK_N", "N")];
        const C: &[TileAxisBinding] = &[axis(&[0], "BLOCK_M", "M"), axis(&[1], "BLOCK_N", "N")];
        KernelTileSpec {
            inputs: &[tensor("a_ptr", 2, A), tensor("b_ptr", 2, B)],
            outputs: &[tensor("c_ptr", 2, C)],
            loop_spec: None,
        }
    };

    /// One block spanning two real dims, NCHW-style: `BLOCK_HW` over dims
    /// 2 and 3, batch and channels untiled.
    const FLATTENED: KernelTileSpec = {
        const AXES: &[TileAxisBinding] = &[axis(&[2, 3], "BLOCK_HW", "HW")];
        KernelTileSpec {
            inputs: &[tensor("x_ptr", 4, AXES)],
            outputs: &[tensor("y_ptr", 4, AXES)],
            loop_spec: None,
        }
    };

    /// An input declaring no axes at all, against a tiled output.
    const UNTILED_INPUT: KernelTileSpec = {
        const OUT: &[TileAxisBinding] = &[axis(&[2], "BLOCK_OL", "OL")];
        KernelTileSpec {
            inputs: &[tensor("x_ptr", 3, &[])],
            outputs: &[tensor("y_ptr", 3, OUT)],
            loop_spec: None,
        }
    };

    // --- the shapes a spec takes -----------------------------------------

    /// GEMM: `M` and `N` propagate from the output by name; `K` never
    /// appears there, so it stays unresolved — full extent — which is what
    /// Welder's model expects of a reduction axis.
    #[test]
    fn test_matmul_propagates_m_and_n_and_leaves_k_unresolved() {
        let inputs = resolve_inputs(&GEMM, &tile(&[Some(64), Some(32)]), &NoConsts).unwrap();

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
            &FLATTENED,
            &tile(&[None, None, Some(1), Some(256)]),
            &NoConsts,
        )
        .unwrap();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0], tile(&[None, None, Some(1), Some(256)]));
    }

    /// `UNTILED_INPUT`'s input declares no axes at all, so every dim is
    /// at full extent — the documented fallback, not a failure.
    #[test]
    fn test_conv1d_input_is_entirely_unresolved() {
        let inputs = resolve_inputs(
            &UNTILED_INPUT,
            &tile(&[Some(2), Some(8), Some(64)]),
            &NoConsts,
        )
        .unwrap();

        assert_eq!(inputs[0], tile(&[None, None, None]));
    }

    // --- windows ---------------------------------------------------------

    const W: TileWindow = TileWindow {
        output_extent_param: "OW",
        stride_const: "STRIDE_W",
        pad_const: Some("PAD_W"),
        kernel_size_const: "KW",
    };

    /// No spec carries a `TileWindow` yet — declaring them on conv/pool is
    /// `teenygrad-1tl.7` — so the window cases use a spec of this shape.
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
        let msg = resolve_inputs(&GEMM, &tile(&[Some(64)]), &NoConsts)
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
        let inputs = resolve_inputs(&GEMM, &tile(&[None, Some(32)]), &NoConsts).unwrap();

        assert_eq!(
            inputs[0],
            tile(&[None, None]),
            "M unknown, so a_ptr's M is too"
        );
        assert_eq!(inputs[1], tile(&[None, Some(32)]));
    }

    /// Propagation across the unary flat elementwise family (teenygrad-1tl.2).
    ///
    /// The rung's characteristic property is often stated as "the identity",
    /// and at rank 1 it literally is. Above rank 1 it is the *product
    /// preserving* identity instead, because the flat spec declares a single
    /// binding whose `dims` spans every dimension: `resolve_inputs` puts the
    /// whole element count on the innermost entry and a bare `1` on the rest,
    /// exactly as [`TileAxisBinding::dims`] documents. So `[8, 9]` resolves to
    /// `[1, 72]`, not back to `[8, 9]` -- the tile covers the same 72 elements
    /// without claiming to be an axis-aligned `8 x 9` subregion.
    ///
    /// These specs are derived from the kernels' own `#[tile(...)]`
    /// attributes, so this round-trips through what the macro actually emits
    /// rather than through a fixture written to agree with it. A representative
    /// handful across the converted modules: an activation with no extra
    /// scalar, one with a scalar, one with two, and a plain unary math op.
    #[test]
    fn test_unary_flat_kernels_resolve_to_a_product_preserving_tile() {
        use crate::nn::activation::{
            hard::HardtanhForward, misc::LeakyReluForward, relu::ReluForward,
        };
        use crate::nn::tensor::elemwise_unary::ElemwiseSqrtForward;

        macro_rules! round_trip {
            ($($kernel:ty),+ $(,)?) => {
                $({
                    let name = stringify!($kernel);
                    for rank in 1..=4usize {
                        let spec = <$kernel>::tile_spec(rank);
                        let out: TileShape = (0..rank).map(|i| Some(8 + i)).collect();
                        let inputs = resolve_inputs(&spec, &out, &NoConsts)
                            .unwrap_or_else(|e| panic!("{name} rank {rank}: {e}"));
                        assert_eq!(inputs.len(), 1, "{name}: one input");
                        let got = &inputs[0];

                        let want_elems: usize = out.iter().map(|d| d.unwrap()).product();
                        let got_elems: usize = got.iter().map(|d| d.unwrap()).product();
                        assert_eq!(
                            got_elems, want_elems,
                            "{name} rank {rank}: the input tile must cover the same element \
                             count as the output tile"
                        );

                        let mut want: TileShape = vec![Some(1); rank];
                        want[rank - 1] = Some(want_elems);
                        assert_eq!(
                            got, &want,
                            "{name} rank {rank}: a flattened axis resolves onto its innermost dim"
                        );
                        if rank == 1 {
                            assert_eq!(got, &out, "{name}: at rank 1 it is the literal identity");
                        }

                        // One unresolved output dim leaves the whole flattened
                        // axis unresolved: it spans every dim, so a partial
                        // element count cannot be split back across them.
                        let mut partial = out.clone();
                        partial[rank - 1] = None;
                        let inputs = resolve_inputs(&spec, &partial, &NoConsts)
                            .unwrap_or_else(|e| panic!("{name} rank {rank} partial: {e}"));
                        assert_eq!(
                            inputs[0],
                            vec![None; rank],
                            "{name} rank {rank}: an unresolved dim unresolves the flattened axis"
                        );
                    }
                })+
            };
        }

        round_trip!(
            ReluForward<f32>,
            LeakyReluForward<f32>,
            HardtanhForward<f32>,
            ElemwiseSqrtForward<f32>,
        );
    }

    /// The multi-operand counterpart of the test above (teenygrad-1tl.3).
    ///
    /// Every operand of a flat elementwise op declares the *same*
    /// `extent_param`, so one resolved block satisfies all of them and each
    /// input tile comes back identical -- for two operands and for three.
    /// `MATMUL_TILE_SPEC` above covers the opposite case, where the operands
    /// name different free variables.
    ///
    /// Operand order is asserted too, because `TileGraph::propagate` zips
    /// `spec.inputs` positionally against a node's parent edges: if the
    /// generated order ever stopped matching the signature, a binary op would
    /// resolve its operands the wrong way round. That is invisible for `add`
    /// and very visible for `sub`.
    #[test]
    fn test_multi_operand_flat_kernels_resolve_every_input_alike() {
        use crate::nn::tensor::elemwise_add::ElemwiseAddForward;
        use crate::nn::tensor::elemwise_binary::{ElemwiseSubForward, ElemwiseWhereForward};

        macro_rules! all_alike {
            ($($kernel:ty => $params:expr),+ $(,)?) => {
                $({
                    let name = stringify!($kernel);
                    let expected_params: &[&str] = &$params;
                    for rank in 1..=3usize {
                        let spec = <$kernel>::tile_spec(rank);
                        assert_eq!(
                            spec.inputs.len(),
                            expected_params.len(),
                            "{name}: one TensorTileSpec per operand"
                        );
                        assert_eq!(
                            spec.inputs.iter().map(|i| i.param).collect::<Vec<_>>(),
                            expected_params,
                            "{name}: operand order must follow the signature, since \
                             propagate zips inputs positionally against parent edges"
                        );
                        // All operands share one extent_param, so one block resolves all.
                        let names: Vec<&str> =
                            spec.inputs.iter().flat_map(|i| i.axes).map(|a| a.extent_param).collect();
                        assert!(
                            names.windows(2).all(|w| w[0] == w[1]),
                            "{name}: every operand names the same free variable, got {names:?}"
                        );

                        let out: TileShape = (0..rank).map(|i| Some(4 + i)).collect();
                        let inputs = resolve_inputs(&spec, &out, &NoConsts)
                            .unwrap_or_else(|e| panic!("{name} rank {rank}: {e}"));
                        let first = &inputs[0];
                        for (i, tile) in inputs.iter().enumerate() {
                            assert_eq!(
                                tile, first,
                                "{name} rank {rank}: operand {i} must resolve to the same tile \
                                 as operand 0"
                            );
                        }
                        // And that shared tile is the same product-preserving
                        // shape the unary family resolves to.
                        let elems: usize = out.iter().map(|d| d.unwrap()).product();
                        let mut want: TileShape = vec![Some(1); rank];
                        want[rank - 1] = Some(elems);
                        assert_eq!(first, &want, "{name} rank {rank}");
                    }
                })+
            };
        }

        all_alike!(
            ElemwiseAddForward<f32> => ["a", "b"],
            ElemwiseSubForward<f32> => ["a", "b"],
            ElemwiseWhereForward<f32> => ["cond", "x", "y"],
        );
    }

    /// A broadcast operand resolves to its own full extent, not to the output's
    /// block (teenygrad-1tl.4).
    ///
    /// `channel_bias_add_forward` is the worked example: `x` and `y` are
    /// `[N, C]` with `N` block-tiled, while `bias` is `(C,)` and declares only
    /// the `C` axis. So `bias` receives no block on any axis and must come back
    /// at full extent -- correct and conservative, and now *declared* that way
    /// rather than achieved by omission.
    ///
    /// Before this rung the generated spec gave every tensor the first
    /// parameter's axis list, so `bias` claimed rank 2 with `N` blocked -- an
    /// axis it does not have. A scheduler reading that would size its tile
    /// along a nonexistent dimension. The assertions on `rank` below are what
    /// catch a regression to that.
    #[test]
    fn test_a_broadcast_operand_resolves_to_its_own_full_extent() {
        use crate::nn::tensor::channel_bias_add::ChannelBiasAddForward;

        let spec = ChannelBiasAddForward::<f32>::tile_spec();
        spec.validate()
            .expect("a derived spec must be self-consistent");

        let (x, bias) = (spec.inputs[0], spec.inputs[1]);
        assert_eq!((x.param, bias.param), ("x", "bias"));
        assert_eq!(x.rank, 2, "the activation carries both axes");
        assert_eq!(
            bias.rank, 1,
            "the bias is (C,), so its spec must not claim the N axis it lacks"
        );
        assert!(
            bias.axes.is_empty(),
            "a broadcast operand receives no block: it declares only the axis \
             it shares, and that axis is untiled"
        );
        assert_eq!(bias.untiled_dims, &["C"]);

        // Resolving an output tile gives the activation a block on N and leaves
        // the bias entirely unresolved -- i.e. at its full extent.
        let inputs = resolve_inputs(&spec, &tile(&[Some(8), Some(4)]), &NoConsts)
            .expect("resolution should succeed");
        assert_eq!(inputs.len(), 2);
        assert_eq!(
            inputs[0],
            tile(&[Some(8), None]),
            "x takes the output's N block; its C dim has no binding so stays unresolved"
        );
        assert_eq!(
            inputs[1],
            tile(&[None]),
            "bias comes back rank 1 and unresolved -- its whole (C,) vector"
        );
    }

    /// `conv2d_forward` declares the accumulation loop it already writes by
    /// hand, so its spec reports a real `TileLoopSpec` (teenygrad-1nr.18.3).
    ///
    /// This is metadata only. The declaration does not generate the loop, and
    /// the design notes on that issue set out why: wrapping a body puts its
    /// trailing `T::store` inside the loop, delimiting the loop part needs
    /// markers in the body (which is what `84ca6eedf` was reverted for), and
    /// conv2d's `x` cannot be an `In<Tile<..>>` at all because it is read at
    /// offsets depending on the loop index while the prelude loads such a
    /// parameter once, up front.
    ///
    /// `trip_count_factors` is a list of names, not a formula: the real count
    /// is `(C_IN / G) * KH * KW`, which mixes a runtime param with three
    /// consts. Nothing evaluates it yet, and `TileLoopSpec`'s own doc says so.
    #[test]
    fn test_conv2d_declares_its_accumulation_loop() {
        use crate::nn::conv::conv2d::Conv2dForward;

        let spec = Conv2dForward::<f32>::tile_spec();
        let loop_spec = spec
            .loop_spec
            .expect("conv2d declares an accumulation loop, so its spec must carry one");

        assert_eq!(loop_spec.carries.len(), 1, "conv2d carries one accumulator");
        assert_eq!(loop_spec.carries[0].name, "acc");
        assert_eq!(
            loop_spec.carries[0].shape_consts,
            &["BLOCK_OW"],
            "acc is a [BLOCK_OW] strip of output columns"
        );
        assert_eq!(
            loop_spec.trip_count_factors,
            &["C_IN", "G", "KH", "KW"],
            "the factors of (C_IN / G) * KH * KW, in signature order"
        );

        spec.validate()
            .expect("a derived spec with a loop must still be self-consistent");
    }

    /// A kernel with no declared loop still reports `None`, so the declaration
    /// is genuinely opt-in and nothing infers a loop that was not stated.
    #[test]
    fn test_a_kernel_without_a_declared_loop_reports_none() {
        use crate::nn::activation::relu::ReluForward;
        use crate::nn::tensor::channel_bias_add::ChannelBiasAddForward;

        assert_eq!(ReluForward::<f32>::tile_spec(2).loop_spec, None);
        assert_eq!(ChannelBiasAddForward::<f32>::tile_spec().loop_spec, None);
    }

    /// Row reductions declare the axis that is *not* available for tiling, and
    /// the scalar carries they accumulate into (teenygrad-1tl.8).
    ///
    /// The carries are `[1]`, not `[BLOCK_N]`: each iteration sums its tile
    /// *into* a one-element accumulator (`T::zeros::<D>(&[1])`). That is why
    /// `TileCarryBinding::shape_consts` had to admit an integer literal --
    /// there is no const named `1`, so a names-only contract could describe
    /// conv2d's `acc: [BLOCK_OW]` and none of the nine reductions. Proving the
    /// declaration on conv2d alone did not surface that.
    #[test]
    fn test_row_reductions_declare_their_reduced_axis_and_scalar_carries() {
        use crate::nn::norm::layernorm::{LayerNormForward, LayerNormForwardInference};
        use crate::nn::norm::rmsnorm::RmsNormForward;

        // LayerNorm: x/y are [M, N] reduced over N; weight/bias are [N];
        // mean/rstd are [M]. Three outputs, which is new for this family.
        let spec = LayerNormForward::<f32>::tile_spec();
        spec.validate().expect("must validate");
        let x = spec.inputs[0];
        assert_eq!(x.param, "x_ptr");
        assert_eq!(x.rank, 2);
        assert_eq!(
            x.reduction_axis,
            Some(1),
            "N is dim 1 and cannot be tiled: a row's mean needs the whole row"
        );
        assert_eq!(x.untiled_dims, &["M", "N"]);
        assert_eq!(spec.outputs.len(), 3, "y, mean and rstd");
        assert_eq!(
            spec.outputs.iter().map(|o| o.param).collect::<Vec<_>>(),
            vec!["y_ptr", "mean_ptr", "rstd_ptr"]
        );

        let l = spec.loop_spec.expect("layernorm walks its row");
        assert_eq!(
            l.carries
                .iter()
                .map(|c| (c.name, c.shape_consts))
                .collect::<Vec<_>>(),
            vec![("sum", &["1"][..]), ("var_sum", &["1"][..])],
            "scalar accumulators, expressed as literals"
        );
        assert_eq!(l.trip_count_factors, &["N", "BLOCK_N"], "cdiv(N, BLOCK_N)");

        // RmsNorm carries one accumulator and has two outputs.
        let spec = RmsNormForward::<f32>::tile_spec();
        spec.validate().expect("must validate");
        assert_eq!(spec.inputs[0].reduction_axis, Some(1));
        let l = spec.loop_spec.expect("rmsnorm walks its row");
        assert_eq!(l.carries.len(), 1);
        assert_eq!(l.carries[0].name, "sq_sum");
        assert_eq!(l.carries[0].shape_consts, &["1"]);

        // The inference variant reduces the same axis with no saved statistics.
        let spec = LayerNormForwardInference::<f32>::tile_spec();
        spec.validate().expect("must validate");
        assert_eq!(spec.inputs[0].reduction_axis, Some(1));
        assert_eq!(spec.outputs.len(), 1, "inference writes only y");
    }

    /// A transpose's permuted axes resolve by name alone (teenygrad-1tl.5).
    ///
    /// `transpose_2d_forward` is the first kernel in the tree with *two*
    /// blocked axes, and the first whose input and output disagree about dim
    /// order: the input is `[M, N]`, the output `[N, M]`. Nothing in the spec
    /// relates them but the extent names. Seeding the output resolves the
    /// input's `M` axis wherever it sits, so a transpose needs no permutation
    /// field -- it falls out of the naming convention, which is what this rung
    /// set out to check.
    ///
    /// Its grid, however, does *not* fall out of dim order: the body decodes
    /// `pid_m` outer, so the grid runs M then N while the output's dims run
    /// N then M. That is what `#[tile_grid(order = ..)]` states.
    #[test]
    fn test_transpose_resolves_permuted_axes_by_name_and_states_its_grid_order() {
        use crate::nn::tensor::transpose::Transpose2dForward;

        let spec = Transpose2dForward::<f32>::tile_spec();
        spec.validate()
            .expect("a derived spec must be self-consistent");

        let axes = |t: &TensorTileSpec| -> Vec<(usize, &'static str, &'static str)> {
            t.axes
                .iter()
                .map(|a| (a.dims[0], a.extent_param, a.block_const))
                .collect()
        };
        assert_eq!(
            axes(&spec.inputs[0]),
            vec![(0, "M", "BLOCK_M"), (1, "N", "BLOCK_N")],
            "the input is [M, N]"
        );
        assert_eq!(
            axes(&spec.outputs[0]),
            vec![(0, "N", "BLOCK_N"), (1, "M", "BLOCK_M")],
            "and the output is [N, M] -- the permutation lives here"
        );
        assert!(
            spec.inputs[0].axes.iter().all(|a| a.window.is_none()),
            "a transpose reindexes, it does not slide a window"
        );

        // An output tile of 16 along N and 8 along M must give the input 8
        // along M and 16 along N: the blocks cross over, by name.
        let inputs = resolve_inputs(&spec, &tile(&[Some(16), Some(8)]), &NoConsts)
            .expect("resolution should succeed");
        assert_eq!(
            inputs[0],
            vec![Some(8), Some(16)],
            "blocks follow their names across the permutation"
        );

        // The grid is the body's decode order, not the output's dim order.
        let grid = Transpose2dForward::<f32>::grid_spec();
        let named: Vec<(&str, Option<&str>)> =
            grid.axes.iter().map(|a| (a.name, a.block_const)).collect();
        assert_eq!(
            named,
            vec![("M", Some("BLOCK_M")), ("N", Some("BLOCK_N"))],
            "pid_m is the outer index, so M comes first"
        );
    }

    /// Padding is a degenerate window, and resolves to the tile's own width
    /// (teenygrad-1tl.5).
    ///
    /// `stride = 1, kernel = 1` gives `(block - 1) * 1 + 1 = block`: each
    /// output element reads one input element, at an origin shifted by the
    /// leading pad. The pad family has no stride or kernel const to name, so
    /// those window fields carry decimal literals, which is why
    /// `window_const` resolves a literal to itself.
    ///
    /// This is the one family where the input region is *not* larger than the
    /// output tile — the opposite end of the same arithmetic the conv and pool
    /// rung exercises.
    #[test]
    fn test_padding_resolves_to_the_tile_width_as_a_degenerate_window() {
        use crate::nn::pad::{
            circular_pad1d::CircularPad1dForward, constant_pad1d::ConstantPad1dForward,
            constant_pad2d::ConstantPad2dForward, constant_pad3d::ConstantPad3dForward,
            reflection_pad1d::ReflectionPad1dForward, replication_pad1d::ReplicationPad1dForward,
        };

        // 1-D: all four families are the same shift, so the same spec.
        for (name, spec, pad) in [
            (
                "constant",
                ConstantPad1dForward::<f32>::tile_spec(),
                "PAD_LEFT",
            ),
            (
                "reflection",
                ReflectionPad1dForward::<f32>::tile_spec(),
                "PAD_LEFT",
            ),
            (
                "replication",
                ReplicationPad1dForward::<f32>::tile_spec(),
                "PAD_LEFT",
            ),
            (
                "circular",
                CircularPad1dForward::<f32>::tile_spec(),
                "PAD_LEFT",
            ),
        ] {
            spec.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            let x = spec.inputs[0];
            assert_eq!(x.axes.len(), 1, "{name}: one spatial axis");
            let axis = x.axes[0];
            assert_eq!(axis.extent_param, "L", "{name}: truthful about its extent");
            assert_eq!(axis.block_const, "BLOCK_OL", "{name}");
            let w = axis.window.expect("padding declares a window");
            assert_eq!(
                (w.stride_const, w.kernel_size_const, w.pad_const),
                ("1", "1", Some(pad)),
                "{name}: a window of stride 1, kernel 1"
            );
            assert_eq!(w.output_extent_param, "OL", "{name}: resolves against OL");

            // No stride or kernel const exists, so the lookup knows nothing
            // and the literals alone have to carry it.
            let inputs = resolve_inputs(&spec, &tile(&[Some(2), Some(3), Some(8)]), &NoConsts)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                inputs[0][2],
                Some(8),
                "{name}: (8 - 1) * 1 + 1 = 8, the tile's own width"
            );
        }

        // 2-D and 3-D: the innermost axis is blocked, the rest carry block 1,
        // and each window names its own output axis.
        let spec = ConstantPad2dForward::<f32>::tile_spec();
        spec.validate().expect("pad2d must validate");
        let axes: Vec<(&str, &str, &str)> = spec.inputs[0]
            .axes
            .iter()
            .map(|a| {
                (
                    a.extent_param,
                    a.block_const,
                    a.window
                        .expect("every spatial axis is windowed")
                        .pad_const
                        .expect("pads always name a pad const"),
                )
            })
            .collect();
        assert_eq!(axes, vec![("H", "1", "PT"), ("W", "BLOCK_OW", "PL")]);

        let spec = ConstantPad3dForward::<f32>::tile_spec();
        spec.validate().expect("pad3d must validate");
        let axes: Vec<(&str, &str)> = spec.inputs[0]
            .axes
            .iter()
            .map(|a| (a.extent_param, a.block_const))
            .collect();
        assert_eq!(
            axes,
            vec![("Dv", "1"), ("H", "1"), ("W", "BLOCK_OW")],
            "all three spatial axes windowed, innermost blocked"
        );
        let inputs = resolve_inputs(
            &spec,
            &tile(&[Some(2), Some(3), Some(1), Some(1), Some(8)]),
            &NoConsts,
        )
        .expect("pad3d resolution");
        assert_eq!(inputs[0][4], Some(8), "innermost resolves to the block");
    }

    /// A real kernel's windowed input resolves to its receptive field
    /// (teenygrad-1nr.18.2).
    ///
    /// Until this, `TileWindow` was constructed nowhere in the repo except the
    /// `WINDOWED` fixture above -- the arithmetic was implemented and tested,
    /// but no kernel could declare a window, so none ever reached it. This is
    /// the same computation against `conv2d_forward`'s own generated spec.
    ///
    /// Note what the declaration had to say for this to work: `x_ptr`'s spatial
    /// axis names the *output's* variable (`OW`, with `BLOCK_OW`), not its own
    /// extent `W`. That is forced, not chosen. Propagation resolves by name, so
    /// an input axis named `W` can never receive the block the output computed
    /// for `OW`. The window is precisely what explains the size difference
    /// between the two. This answers the naming question `teenygrad-1tl.7`
    /// raises for the conv and pool family.
    ///
    /// `x_ptr` now carries a window on *both* spatial axes: H's has a fixed
    /// block of 1, since one program instance computes one output row
    /// (teenygrad-1tl.7). So this picks the W axis out by the output variable
    /// its window names rather than taking the first windowed axis.
    #[test]
    fn test_conv2d_windowed_input_resolves_to_its_receptive_field() {
        use crate::nn::conv::conv2d::Conv2dForward;

        let spec = Conv2dForward::<f32>::tile_spec();
        spec.validate()
            .expect("a derived spec must be self-consistent");

        let x = spec.inputs[0];
        let windowed = x
            .axes
            .iter()
            .find(|a| a.window.is_some_and(|w| w.output_extent_param == "OW"))
            .expect("x_ptr's W axis declares a window");
        let w = windowed.window.expect("just checked");
        assert_eq!(
            (w.stride_const, w.pad_const, w.kernel_size_const),
            ("STRIDE_W", Some("PAD_W"), "KW")
        );
        assert_eq!(
            w.output_extent_param, "OW",
            "the window names the output axis it resolves against, so x keeps its real extent W"
        );
        assert_eq!(
            windowed.extent_param, "W",
            "and the spec stays truthful about the input's own spatial extent"
        );

        // 3x3 conv, stride 1: an 8-column output tile reads (8 - 1) * 1 + 3 = 10.
        let consts = Table::new(&[("STRIDE_W", 1), ("KW", 3), ("PAD_W", 1)]);
        let out = tile(&[Some(2), Some(16), Some(5), Some(8)]);
        let inputs = resolve_inputs(&spec, &out, &consts).expect("resolution should succeed");
        let ow_dim = windowed.dims[windowed.dims.len() - 1];
        assert_eq!(
            inputs[0][ow_dim],
            Some(10),
            "(block - 1) * stride + kernel, forward and exact"
        );

        // Padding shifts the origin, not the size: the same block with p = 0
        // reads the same 10 elements. An interior tile touches no padding.
        let unpadded = Table::new(&[("STRIDE_W", 1), ("KW", 3), ("PAD_W", 0)]);
        let inputs = resolve_inputs(&spec, &out, &unpadded).expect("resolution should succeed");
        assert_eq!(inputs[0][ow_dim], Some(10), "pad does not enter the extent");

        // Stride 2 spreads the same block further: (8 - 1) * 2 + 3 = 17.
        let strided = Table::new(&[("STRIDE_W", 2), ("KW", 3), ("PAD_W", 1)]);
        let inputs = resolve_inputs(&spec, &out, &strided).expect("resolution should succeed");
        assert_eq!(inputs[0][ow_dim], Some(17));
    }

    /// The conv and pool family resolves its windowed input to the receptive
    /// field (teenygrad-1tl.7).
    ///
    /// Three things about the declaration are worth stating, because none is
    /// obvious from reading it:
    ///
    /// - The windowed axis keeps its own extent (`L`, `W`) and the window names
    ///   the *output* axis whose block it resolves against. A windowed input's
    ///   extent never appears in the output, so both names are needed.
    /// - It also carries the output's `block_const`. That is structural, not a
    ///   claim about its tile size: only a blocked axis becomes a
    ///   `TileAxisBinding`, and a `TileWindow` hangs off one -- including an
    ///   axis the kernel does not block, whose binding carries the literal
    ///   block `1` (teenygrad-1tl.7). Resolution never reads the input axis's
    ///   own `block_const`.
    /// - `pad` is absent on the pools because most have no padding const at all,
    ///   and padding shifts the window's origin rather than its size.
    #[test]
    fn test_conv_and_pool_windows_resolve_to_the_receptive_field() {
        use crate::nn::conv::{
            conv1d::Conv1dForward, conv2d::Conv2dBiasForward, conv3d::Conv3dForward,
        };
        use crate::nn::pool::{
            avgpool1d::Avgpool1dForward, avgpool2d::Avgpool2dForward, avgpool3d::Avgpool3dForward,
            lppool1d::Lppool1dForward, lppool2d::Lppool2dForward, lppool3d::Lppool3dForward,
            maxpool1d::Maxpool1dForward, maxpool2d::Maxpool2dForward, maxpool3d::Maxpool3dForward,
        };

        // 1-D: one spatial axis, fully expressible. (8 - 1) * 2 + 3 = 17.
        for (name, spec) in [
            ("maxpool1d", Maxpool1dForward::<f32>::tile_spec()),
            ("avgpool1d", Avgpool1dForward::<f32>::tile_spec()),
            ("lppool1d", Lppool1dForward::<f32>::tile_spec()),
        ] {
            spec.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            let axis = spec.inputs[0]
                .axes
                .iter()
                .find(|a| a.window.is_some())
                .unwrap_or_else(|| panic!("{name}: the spatial axis declares a window"));
            let w = axis.window.expect("just checked");
            assert_eq!(axis.extent_param, "L", "{name}: keeps its own extent");
            assert_eq!(w.output_extent_param, "OL", "{name}: resolves against OL");
            assert_eq!(w.pad_const, None, "{name}: no padding const exists");

            let consts = Table::new(&[("STRIDE", 2), ("KL", 3)]);
            let inputs = resolve_inputs(&spec, &tile(&[Some(2), Some(4), Some(8)]), &consts)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(inputs[0][2], Some(17), "{name}: (8 - 1) * 2 + 3");
        }

        // conv1d does have padding, and it still does not enter the extent.
        let spec = Conv1dForward::<f32>::tile_spec();
        spec.validate().expect("conv1d spec must validate");
        let axis = spec.inputs[0]
            .axes
            .iter()
            .find(|a| a.window.is_some())
            .unwrap();
        assert_eq!(axis.window.unwrap().pad_const, Some("PAD"));
        let consts = Table::new(&[("STRIDE", 1), ("KL", 3), ("PAD", 1)]);
        let inputs = resolve_inputs(&spec, &tile(&[Some(2), Some(6), Some(8)]), &consts).unwrap();
        assert_eq!(inputs[0][2], Some(10), "(8 - 1) * 1 + 3, padding excluded");

        // 2-D: *both* spatial axes carry a window now (teenygrad-1tl.7).
        // They differ only in block -- `BLOCK_OW` on W, the literal 1 on H,
        // because the body computes a tile of columns for one scalar `oh`.
        for (name, spec, pad_h) in [
            (
                "maxpool2d",
                Maxpool2dForward::<f32>::tile_spec(),
                Some("PAD_H"),
            ),
            ("lppool2d", Lppool2dForward::<f32>::tile_spec(), None),
            ("avgpool2d", Avgpool2dForward::<f32>::tile_spec(), None),
            (
                "conv2d_bias",
                Conv2dBiasForward::<f32>::tile_spec(),
                Some("PAD_H"),
            ),
        ] {
            spec.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            let x = spec.inputs[0];
            let windowed: Vec<(&str, &str)> = x
                .axes
                .iter()
                .filter(|a| a.window.is_some())
                .map(|a| (a.extent_param, a.block_const))
                .collect();
            assert_eq!(
                windowed,
                vec![("H", "1"), ("W", "BLOCK_OW")],
                "{name}: every spatial axis is windowed, H with a fixed block of 1"
            );
            assert!(
                !x.untiled_dims.contains(&"H"),
                "{name}: H is a binding now, so it has left untiled_dims"
            );
            let h = x.axes.iter().find(|a| a.extent_param == "H").unwrap();
            let hw = h.window.expect("H declares a window");
            assert_eq!(
                hw.output_extent_param, "OH",
                "{name}: H resolves against OH"
            );
            assert_eq!(hw.pad_const, pad_h, "{name}: only maxpool2d has padding");

            let consts = Table::new(&[("STRIDE_W", 2), ("KW", 3), ("STRIDE_H", 2), ("KH", 3)]);
            let out = tile(&[Some(2), Some(4), Some(5), Some(8)]);
            let inputs =
                resolve_inputs(&spec, &out, &consts).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                inputs[0][3],
                Some(17),
                "{name}: W resolves, (8 - 1) * 2 + 3"
            );
            // H's window is declared but still unresolvable: it resolves
            // against `OH`, and the *output* leaves OH untiled, so no block
            // ever propagates for it. It stays at full extent rather than
            // feeding `H` through the receptive field and claiming the kernel
            // reads past the tensor. Giving it a block means rewriting the
            // `pid` decode (teenygrad-1nr.18.4).
            assert_eq!(
                inputs[0][2], None,
                "{name}: H unresolved, so full extent -- not an over-read"
            );
        }

        // 3-D: all three spatial axes, same shape of answer.
        for (name, spec, pads) in [
            ("maxpool3d", Maxpool3dForward::<f32>::tile_spec(), false),
            ("avgpool3d", Avgpool3dForward::<f32>::tile_spec(), false),
            ("lppool3d", Lppool3dForward::<f32>::tile_spec(), false),
            ("conv3d", Conv3dForward::<f32>::tile_spec(), true),
        ] {
            spec.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            let x = spec.inputs[0];
            let windowed: Vec<(&str, &str)> = x
                .axes
                .iter()
                .filter(|a| a.window.is_some())
                .map(|a| (a.extent_param, a.block_const))
                .collect();
            assert_eq!(
                windowed,
                vec![("Dv", "1"), ("H", "1"), ("W", "BLOCK_OW")],
                "{name}: all three spatial axes are windowed"
            );
            let outs: Vec<&str> = x
                .axes
                .iter()
                .filter_map(|a| a.window.map(|w| w.output_extent_param))
                .collect();
            assert_eq!(
                outs,
                vec!["OD", "OH", "OW"],
                "{name}: each window names its own output axis"
            );
            let has_pad = x
                .axes
                .iter()
                .filter_map(|a| a.window)
                .all(|w| w.pad_const.is_some());
            assert_eq!(has_pad, pads, "{name}: only the conv declares padding");

            let consts = Table::new(&[("STRIDE_W", 2), ("KW", 3)]);
            let out = tile(&[Some(2), Some(4), Some(3), Some(5), Some(8)]);
            let inputs =
                resolve_inputs(&spec, &out, &consts).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                inputs[0][4],
                Some(17),
                "{name}: W resolves, (8 - 1) * 2 + 3"
            );
        }
    }
}
