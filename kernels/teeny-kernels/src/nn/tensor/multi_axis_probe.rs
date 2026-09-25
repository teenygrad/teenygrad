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

//! Test-only probe for the N-axis auto-prelude (teenygrad-1nr.18.1).
//!
//! Every real multi-axis kernel in the tree also loops -- conv and pool over
//! a receptive field, the norms over a row, `channel_bias_add` over its own
//! tile range -- so none of them can exercise the multi-axis prelude on its
//! own until `teenygrad-1nr.18.3` lands the wrapper loop. This kernel exists
//! so the generated indexing is type-checked and its spec asserted in the
//! meantime. It is not lowered from any `Op` and is not part of the public
//! kernel set.

#![allow(non_snake_case)]

use teeny_core::dtype::Num;
use teeny_macros::tiled_kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Copy, over a rank-2 `[H, W]` tensor: `H` is one index per CTA, `W` is
/// block-tiled. Exercises the flat-pid decode and the row-major stride
/// derivation the prelude generates.
#[tiled_kernel]
pub fn multi_axis_probe_forward<T: Triton, D: Num, const BLOCK_W: i32>(
    #[tile(extent = H)]
    #[tile(block = BLOCK_W, extent = W)]
    x: In<Tile<T, D>>,
    #[tile(extent = H)]
    #[tile(block = BLOCK_W, extent = W)]
    y: Out<Tile<T, D>>,
    H: i32,
    W: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    T::store(y.tensor, x.tensor, x.mask, &[], None, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two declared axes give a fixed-rank spec with one binding per
    /// *blocked* axis -- the untiled one is named in `untiled_dims`, the
    /// same shape the raw-pointer path produces. Contrast the single-axis
    /// kernels (`relu`/`silu`), whose spec takes a runtime `rank` because
    /// the same flat kernel really does apply at any rank.
    #[test]
    fn test_two_declared_axes_give_a_fixed_rank_spec() {
        let spec = MultiAxisProbeForward::<f32>::tile_spec();

        assert_eq!(spec.inputs.len(), 1);
        assert_eq!(spec.outputs.len(), 1);
        assert_eq!(spec.loop_spec, None);

        let x = spec.inputs[0];
        let y = spec.outputs[0];
        assert_eq!((x.param, y.param), ("x", "y"));

        for tensor in [x, y] {
            assert_eq!(tensor.rank, 2, "declared by the signature, not per node");
            assert_eq!(tensor.untiled_dims, &["H"]);
            assert_eq!(tensor.axes.len(), 1, "only W carries `block = ..`");
            assert_eq!(tensor.axes[0].dims, &[1], "W is the innermost dim");
            assert_eq!(tensor.axes[0].block_const, "BLOCK_W");
            assert_eq!(tensor.axes[0].extent_param, "W");
            assert_eq!(tensor.axes[0].window, None);
        }

        spec.validate()
            .expect("a derived spec must be self-consistent");
    }
}
