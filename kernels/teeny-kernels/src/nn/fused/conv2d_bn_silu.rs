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

use core::ops::BitAnd;

use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Fused Conv2d + BatchNorm2d (inference) + SiLU forward pass.
///
/// Epilog fusion: after the conv accumulation loop, applies BN affine and
/// SiLU in registers before the final global store, eliminating 2 intermediate
/// global memory round-trips vs 3 separate kernels.
///
/// BN parameters must be precomputed by the caller as:
///   `bn_scale[c] = gamma[c] / sqrt(var[c] + eps)`
///   `bn_shift[c] = beta[c] - bn_scale[c] * mean[c]`
///
/// Grid: `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Inference-only; no backward pass.
#[tiled_kernel]
// teenygrad-3dp5: the loop is generated now, and `finish` carries the epilogue
// the comment here used to say this kernel was waiting for -- the batch-norm
// affine and then SiLU, both pure expressions over the carry.
#[tile_loop(
    trip_count = [C_IN, G, KH, KW],
    axes = [c_in_local = (C_IN / G), kh = KH, kw = KW],
    generate
)]
#[tile_carry(
    acc = [BLOCK_OW],
    finish = {
        let bn_off = T::arange(0, 1) + tile_c_out;
        let scale = T::broadcast_to(
            T::load(bn_scale_ptr.add_offsets(bn_off), None, None, &[], None, None, None, false),
            &[BLOCK_OW],
        );
        let shift = T::broadcast_to(
            T::load(bn_shift_ptr.add_offsets(bn_off), None, None, &[], None, None, None, false),
            &[BLOCK_OW],
        );
        let bn_out = scale * acc + shift;
        // SiLU: y = x * sigmoid(x) = x / (1 + exp(-x)).
        let one = T::full(&[BLOCK_OW], 1.0_f32);
        let neg1 = T::full(&[BLOCK_OW], -1.0_f32);
        bn_out * (one / (one + T::exp(neg1 * bn_out)))
    }
)]
pub fn conv2d_bn_silu_forward<
    T: Triton,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const G: i32,
    const BLOCK_OW: i32,
>(
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C_IN)]
    #[tile(extent = H, window(stride = STRIDE_H, pad = PAD_H, kernel = KH, output = OH))]
    #[tile(
        block = BLOCK_OW,
        extent = W,
        window(stride = STRIDE_W, pad = PAD_W, kernel = KW, output = OW)
    )]
    // `#[tile(..)]` above says what `x` IS; this says how to read it per
    // iteration, as conv2d_forward does. `bounds` names the padded spatial axes.
    #[tile_loop_tile(
        index = [
            tile_b = _B,
            ((tile_c_out / (C_OUT / G)) * (C_IN / G) + c_in_local) = C_IN,
            (tile_oh * STRIDE_H + kh - PAD_H) = H,
            (__tile_range * STRIDE_W + kw - PAD_W) = W
        ],
        bounds = [2, 3]
    )]
    x: In<Tile<T, f32>>,
    // Weights and the two per-channel batchnorm operands stay untagged, as
    // `conv2d_bias_forward` leaves its own: none is sliced by the output tile.
    #[tile_loop_scalar(
        index = [tile_c_out = C_OUT, c_in_local = (C_IN / G), kh = KH, kw = KW]
    )]
    w: In<Tile<T, f32>>,
    bn_scale_ptr: In<T::Pointer<f32>>,
    bn_shift_ptr: In<T::Pointer<f32>>,
    #[tile(name = "B", extent = _B)]
    #[tile(extent = C_OUT)]
    #[tile(extent = OH)]
    #[tile(block = BLOCK_OW, extent = OW)]
    y: Out<Tile<T, f32>>,
    _B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    acc = acc + x * w;
}

// ── RuntimeOp ────────────────────────────────────────────────────────────────
//
// Params layout: [weight [C_OUT, C_IN/G, KH, KW], bn_scale [C_OUT], bn_shift [C_OUT]]
// pack_args order: x_ptr, w_ptr, bn_scale_ptr, bn_shift_ptr, y_ptr,
//                  B, C_IN, C_OUT, H, W, OH, OW

impl teeny_core::model::RuntimeOp for Conv2dBnSiluForward {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c_in = input_shapes[0][1];
        let c_out = output_shape[1];
        vec![
            vec![
                c_out,
                c_in / self.g as usize,
                self.kh as usize,
                self.kw as usize,
            ],
            vec![c_out],
            vec![c_out],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bn_scale", "bn_shift"]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(params[0]); // w_ptr
        visitor.visit_ptr(params[1]); // bn_scale_ptr
        visitor.visit_ptr(params[2]); // bn_shift_ptr
        visitor.visit_ptr(output); // y_ptr
        visitor.visit_i32(input_shape[0] as i32); // B
        visitor.visit_i32(input_shape[1] as i32); // C_IN
        visitor.visit_i32(output_shape[1] as i32); // C_OUT
        visitor.visit_i32(input_shape[2] as i32); // H
        visitor.visit_i32(input_shape[3] as i32); // W
        visitor.visit_i32(output_shape[2] as i32); // OH
        visitor.visit_i32(output_shape[3] as i32); // OW
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32,
            1,
            1,
        ]
    }
}
