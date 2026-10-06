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

use teeny_core::dtype::Float;
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 1-D Lp-norm pooling forward pass.
///
/// `y = (Σ |x_i|^p)^(1/p)` over the kernel window.
///
/// `pow(|x|, p)` is computed as `exp(p * log(max(|x|, ε)))` to avoid
/// `log(0)`. `p` is a runtime float parameter.
///
/// Grid: `pid = (b * C + c) * num_ol_tiles + ol_tile`
///
/// **Constraints**: no padding; `OL = (L - KL) / STRIDE + 1`.
// teenygrad-1tl.7: the spatial axis is read through a strided sliding window.
// An output tile of `block` positions reads `(block - 1) * STRIDE + KL` input
// elements -- forward and exact. The window names the OUTPUT axis (`OL`) whose
// block it resolves against, while this axis keeps its own extent `L`: a
// windowed input's extent never appears in the output, so propagation needs
// both names (teenygrad-1nr.18.2).
#[tiled_kernel]
// teenygrad-3dp5. The loop-invariant `p_vec`/`inv_p_vec`/`eps_vec` are inlined
// rather than bound in a prologue: the author's body IS the loop body, so there
// is nowhere before it for a binding to live. They are `T::full` of a constant,
// so inlining costs nothing the compiler will not hoist -- and it means these
// three kernels needed no new macro feature after all.
//
// `init` carries the f32 accumulator: this kernel reduces in f32 while its
// output is D, so the generated `zeros::<D>` default would be the wrong dtype.
// `finish` carries the p-norm root, and casts back.
#[tile_loop(trip_count = [KL], axes = [kl = KL], generate)]
#[tile_carry(
    acc = [BLOCK_OL],
    init = T::zeros::<f32>(&[BLOCK_OL]),
    finish = T::cast::<f32, D>(
        T::exp(
            T::log(T::maximum(acc, T::full::<f32>(&[BLOCK_OL], 1e-12_f32)))
                * T::full::<f32>(&[BLOCK_OL], 1.0_f32 / p)
        ),
        None,
        false
    )
)]
pub fn lppool1d_forward<
    T: Triton,
    D: Float,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    // `block = BLOCK_OL` is not a claim that this axis is tiled in BLOCK_OL-sized
    // pieces: it names the block the window resolves against. The real per-tile
    // extent here is the receptive field, which `resolve_inputs` computes from
    // the window; it never reads this axis's own `block_const`.
    #[tile(
        block = BLOCK_OL,
        extent = L,
        window(stride = STRIDE, kernel = KL, output = OL)
    )]
    // No `bounds`: an lp-pool does not pad, so every windowed coordinate of an
    // in-range output tile is in range and the only masked lanes are
    // out-of-range OUTPUT lanes, which the masked store discards. The generated
    // read's default zeros fill matches the hand-written one exactly: a masked
    // lane becomes `max(|0|, eps)` and contributes `eps^p`, as before
    // (teenygrad-3dp5).
    #[tile_loop_tile(
        index = [
            tile_b = _B,
            tile_c = C,
            (__tile_range * STRIDE + kl) = L
        ]
    )]
    input: In<Tile<T, D>>,
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C)]
    #[tile(block = BLOCK_OL, extent = OL)]
    output: Out<Tile<T, D>>,
    _B: i32,
    C: i32,
    L: i32,
    OL: i32,
    p: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One iteration: |x|^p, accumulated. The epsilon floor keeps `log` finite
    // for a zero lane, exactly as the hand-written body did.
    acc = acc
        + T::exp(
            T::full::<f32>(&[BLOCK_OL], p)
                * T::log(T::maximum(
                    T::abs(T::cast::<D, f32>(input, None, false)),
                    T::full::<f32>(&[BLOCK_OL], 1e-12_f32),
                )),
        );
}

/// 1-D Lp-norm pooling backward pass.
///
/// `dx_i = dy * sign(x_i) * (|x_i| / max(y, ε))^(p-1) / max(y, ε)`.
///
/// Requires both the original input `x` and the forward output `y`.
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn lppool1d_backward<
    T: Triton,
    D: Float,
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
    p: f32,
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

    let pm1_vec = T::full::<f32>(&[BLOCK_OL], p - 1.0_f32);
    let eps_vec = T::full::<f32>(&[BLOCK_OL], 1e-12_f32);
    let zeros_f32 = T::zeros::<f32>(&[BLOCK_OL]);

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
    let dy_f32 = T::cast::<D, f32>(dy_tile, None, false);

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
    let y_f32 = T::cast::<D, f32>(y_tile, None, false);
    let safe_y = T::maximum(y_f32, eps_vec);

    let loop_bound = KL;
    for kl in 0..loop_bound {
        let il_range = ol_range * STRIDE + kl;
        let in_offsets = il_range + in_bc_base;
        let x_tile = T::load(
            x_ptr.add_offsets(in_offsets),
            Some(ol_mask),
            Some(T::zeros::<D>(&[BLOCK_OL])),
            &[],
            None,
            None,
            None,
            false,
        );
        let x_f32 = T::cast::<D, f32>(x_tile, None, false);
        let abs_x = T::abs(x_f32);
        let safe_abs = T::maximum(abs_x, eps_vec);

        // sign(x): 1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0.
        let pos = T::where_(
            T::gt(x_f32, zeros_f32),
            T::full(&[BLOCK_OL], 1.0_f32),
            zeros_f32,
        );
        let neg = T::where_(
            T::gt(zeros_f32, x_f32),
            T::full(&[BLOCK_OL], 1.0_f32),
            zeros_f32,
        );
        let sign_x = pos - neg;

        // (|x| / y)^(p-1) = exp((p-1) * log(|x| / y))
        let ratio = safe_abs / safe_y;
        let safe_ratio = T::maximum(ratio, eps_vec);
        let pow_ratio = T::exp(pm1_vec * T::log(safe_ratio));

        let dx_f32 = dy_f32 * sign_x * pow_ratio;
        let dx_tile = T::cast::<f32, D>(dx_f32, None, false);

        T::atomic_add(
            dx_ptr.add_offsets(in_offsets),
            dx_tile,
            Some(ol_mask),
            None,
            None,
        );
    }
}

pub struct Lppool1dOp<'a, T: Float> {
    pub forward: Lppool1dForward<T>,
    pub backward: Lppool1dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
