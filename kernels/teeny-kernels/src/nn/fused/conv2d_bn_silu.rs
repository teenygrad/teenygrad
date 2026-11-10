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

use teeny_macros::kernel;
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
#[kernel]
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
    x_ptr: InPtr<T::Pointer<f32>>,
    w_ptr: InPtr<T::Pointer<f32>>,
    bn_scale_ptr: InPtr<T::Pointer<f32>>,
    bn_shift_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<f32>>,
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
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    // Decode flat pid → (b, c_out, oh, ow_tile).
    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let out_bc_base = (b * C_OUT + c_out) * OH * OW;

    let c_in_per_group = C_IN / G;
    let g_idx = c_out / (C_OUT / G);
    let c_in_start = g_idx * c_in_per_group;

    // ── Conv accumulation (same as conv2d_forward) ────────────────────────────
    let mut acc = T::zeros::<f32>(&[BLOCK_OW]);

    let loop_bound = c_in_per_group * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH;
        let c_in = c_in_start + c_in_local;

        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let load_mask = ow_mask & h_in_bounds & w_in_bounds;

        let x_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(load_mask),
            Some(T::zeros::<f32>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        // Weight layout [C_OUT, C_IN/G, KH, KW]: load scalar and broadcast.
        let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        acc = acc + x_tile * w_tile;
    }

    // ── BatchNorm epilog: acc = bn_scale[c_out] * acc + bn_shift[c_out] ───────
    let bn_off = T::arange(0, 1) + c_out;
    let scale_1 = T::load(
        bn_scale_ptr.add_offsets(bn_off),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let scale = T::broadcast_to(scale_1, &[BLOCK_OW]);
    let shift_1 = T::load(
        bn_shift_ptr.add_offsets(bn_off),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let shift = T::broadcast_to(shift_1, &[BLOCK_OW]);
    let bn_out = scale * acc + shift;

    // ── SiLU epilog: y = x * sigmoid(x) = x / (1 + exp(-x)) ─────────────────
    let one = T::full(&[BLOCK_OW], 1.0_f32);
    let neg1 = T::full(&[BLOCK_OW], -1.0_f32);
    let y = bn_out * (one / (one + T::exp(neg1 * bn_out)));

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        y_ptr.add_offsets(out_offsets),
        y,
        Some(ow_mask),
        &[],
        None,
        None,
    );
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
