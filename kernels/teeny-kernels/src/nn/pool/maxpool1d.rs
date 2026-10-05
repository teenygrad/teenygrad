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

#![allow(non_snake_case)]

use teeny_core::dtype::Num;
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 1-D max-pooling forward pass.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// Initialises the accumulator to `-inf` and reduces over KL positions.
///
/// **Constraints**: no padding; `OL = (L - KL) / STRIDE + 1`.
// teenygrad-1tl.7: the spatial axis is read through a strided sliding window.
// An output tile of `block` positions reads `(block - 1) * STRIDE + KL` input
// elements -- forward and exact. The window names the OUTPUT axis (`OL`) whose
// block it resolves against, while this axis keeps its own extent `L`: a
// windowed input's extent never appears in the output, so propagation needs
// both names (teenygrad-1nr.18.2).
#[tiled_kernel]
// teenygrad-3dp5. `init` is the point: a max-pool's accumulator starts at
// negative infinity, and while the generated carry was hardcoded to `zeros`
// this kernel could not use `generate` at all. No `finish` -- a max-pool has no
// epilogue, the carry IS the result.
#[tile_loop(trip_count = [KL], axes = [kl = KL], generate)]
#[tile_carry(
    acc = [BLOCK_OL],
    init = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OL], -3.4028235e38_f32), None, false)
)]
pub fn maxpool1d_forward<T: Triton, D: Num, const KL: i32, const STRIDE: i32, const BLOCK_OL: i32>(
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    // `block = BLOCK_OL` is not a claim that this axis is tiled in BLOCK_OL-sized
    // pieces: it names the block the window resolves against. The real per-tile
    // extent here is the receptive field, which `resolve_inputs` computes from
    // the window; it never reads this axis's own `block_const`.
    // `fill = neg_inf` states the reduction's identity: the generated read's
    // default is zeros, which is the identity for a SUM and would beat a
    // genuinely negative maximum.
    //
    // It is NOT observable here, and that is structural rather than a gap in
    // the fixture. This pool has no padding and declares no `bounds`, so the
    // read's mask is exactly the output tile's `in_bounds` -- every masked lane
    // is an out-of-range OUTPUT lane, which the masked store discards. Removing
    // this line leaves every test passing; I checked.
    //
    // It is kept because it is true of the operand, and because the padded
    // 2-D/3-D pools DO bounds-check input coordinates that belong to stored
    // output lanes. That is where a zeros fill corrupts the answer, and where
    // the test for it belongs (teenygrad-3dp5).
    #[tile(
        block = BLOCK_OL,
        extent = L,
        window(stride = STRIDE, kernel = KL, output = OL),
        fill = neg_inf
    )]
    #[tile_loop_tile(index = [tile_b = _B, tile_c = C, (__tile_range * STRIDE + kl) = L])]
    input: In<Tile<T, D>>,
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    #[tile(block = BLOCK_OL, extent = OL)]
    output: Out<Tile<T, D>>,
    _B: i32,
    C: i32,
    L: i32,
    OL: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    acc = T::maximum(acc, input);
}

/// 1-D max-pooling backward pass.
///
/// Re-scans the input window to find the max position, then scatters `dy` to
/// all input elements that equal the stored output maximum. `dx` must be
/// zero-initialised before launch.
#[kernel]
pub fn maxpool1d_backward<
    T: Triton,
    D: Num,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    dy_ptr: In<T::Pointer<D>>,
    x_ptr: In<T::Pointer<D>>,
    y_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    _B: i32,
    C: i32,
    L: i32,
    OL: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ol_tiles = T::cdiv(OL, BLOCK_OL);

    let ol_tile = pid % num_ol_tiles;
    let bc = pid / num_ol_tiles;
    let c = bc % C;
    let b = bc / C;

    let ol_start = ol_tile * BLOCK_OL;
    let ol_range = T::arange(0, BLOCK_OL) + ol_start;
    let ol_mask = ol_range.lt(OL);

    let in_bc_base = (b * C + c) * L;
    let out_bc_base = (b * C + c) * OL;

    let dy_offsets = ol_range + out_bc_base;
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ol_mask),
        Some(T::zeros::<D>(&[BLOCK_OL])),
        &[],
        None,
        None,
        None,
        false,
    );

    let y_tile = T::load(
        y_ptr.add_offsets(dy_offsets),
        Some(ol_mask),
        Some(T::zeros::<D>(&[BLOCK_OL])),
        &[],
        None,
        None,
        None,
        false,
    );

    let loop_bound = KL;
    for kl in 0..loop_bound {
        let il_range = ol_range * STRIDE + kl;
        let in_offsets = il_range + in_bc_base;
        let x_tile = T::load(
            x_ptr.add_offsets(in_offsets),
            Some(ol_mask),
            Some(T::cast::<f32, D>(
                T::full::<f32>(&[BLOCK_OL], -3.4028235e38_f32),
                None,
                false,
            )),
            &[],
            None,
            None,
            None,
            false,
        );
        let is_max = T::eq(x_tile, y_tile);
        let grad = T::where_(is_max, dy_tile, T::zeros::<D>(&[BLOCK_OL]));
        T::atomic_add(
            dx_ptr.add_offsets(in_offsets),
            grad,
            Some(ol_mask),
            None,
            None,
        );
    }
}

pub struct Maxpool1dOp<'a, T: Num> {
    pub forward: Maxpool1dForward<T>,
    pub backward: Maxpool1dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
