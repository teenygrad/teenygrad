/*
 * Copyright (c) 2026 Teenygrad.
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

/// Fused Conv2d + BatchNorm2d (inference) + SiLU — channel-tiled variant.
///
/// Processes BLOCK_N output channels and BLOCK_OW output-width positions
/// per thread block using a 2-D [BLOCK_N, BLOCK_OW] accumulator built from
/// outer-product updates:  acc += w_row[:, None] * x_col[None, :]
///
/// This gives BLOCK_N× better x-value reuse vs the scalar direct-conv kernel
/// and substantially higher arithmetic intensity for large-channel layers.
///
/// Restrictions (enforced by dispatch in graph/mod.rs):
///   - groups == 1  (non-depthwise, non-grouped)
///   - C_OUT is a multiple of BLOCK_N (no channel padding needed)
///     OR the extra channels are masked by the descriptor's shape bound.
///
/// BN parameters must be precomputed by the caller as:
///   bn_scale[c] = gamma[c] / sqrt(var[c] + eps)
///   bn_shift[c] = beta[c] - bn_scale[c] * mean[c]
///
/// Grid: pid = ((b * OH + oh) * num_n_tiles + n_tile) * num_ow_tiles + ow_tile
///
/// Inference-only; no backward pass.
#[kernel]
pub fn conv2d_bn_silu_tiled_forward<
    T: Triton,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const BLOCK_OW: i32,
    const BLOCK_N: i32,
>(
    x_ptr: T::Pointer<f32>,
    w_ptr: T::Pointer<f32>,
    bn_scale_ptr: T::Pointer<f32>,
    bn_shift_ptr: T::Pointer<f32>,
    y_ptr: T::Pointer<f32>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
    // y_col_stride: allocated column width per oh row.
    //   Must satisfy: y_col_stride >= max(OW, BLOCK_OW) AND divisible by 4.
    //   Ensures TMA store positions oh*y_col_stride are always 16-byte aligned,
    //   and adjacent oh tiles never overlap (y_col_stride >= BLOCK_OW).
    //   Caller allocates B * C_OUT * OH * y_col_stride floats for y.
    y_col_stride: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);
    let num_n_tiles = T::cdiv(C_OUT, BLOCK_N);

    // Decode flat pid → (b, oh, n_tile, ow_tile).
    let ow_tile = pid % num_ow_tiles;
    let tmp = pid / num_ow_tiles;
    let n_tile = tmp % num_n_tiles;
    let tmp2 = tmp / num_n_tiles;
    let oh = tmp2 % OH;
    let b = tmp2 / OH;

    let ow_start = ow_tile * BLOCK_OW;
    let c_out_start = n_tile * BLOCK_N;

    // 1-D lane ranges
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;      // [BLOCK_OW]
    let c_out_range = T::arange(0, BLOCK_N) + c_out_start; // [BLOCK_N]

    let ow_mask = ow_range.lt(OW);          // [BLOCK_OW] bool
    let n_mask = c_out_range.lt(C_OUT);     // [BLOCK_N] bool

    // 2-D accumulator [BLOCK_N, BLOCK_OW]  (channel dim first — matches NCHW output).
    let mut acc = T::zeros::<f32>(&[BLOCK_N, BLOCK_OW]);

    // G=1 is enforced by dispatch — all c_out in this tile use the same c_in range.
    let loop_bound = C_IN * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH;

        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        // Broadcast ih to [BLOCK_OW] for comparison.
        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let x_load_mask = ow_mask & h_in_bounds & w_in_bounds;

        // ── x_col [BLOCK_OW]: one spatial slice (same for all output channels) ─
        let x_offsets = iw_range + ((b * C_IN + c_in_local) * H * W + ih * W);
        let x_col = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(x_load_mask),
            Some(T::zeros::<f32>(&[BLOCK_OW])),
            &[],
            None, None, None, false,
        );

        // ── w_row [BLOCK_N]: weights for BLOCK_N output channels at this k ─────
        // Weight layout [C_OUT, C_IN, KH, KW] (groups=1):
        //   w[c_out, c_in_local, kh, kw] = w_flat[c_out*(C_IN*KH*KW) + (c_in_local*KH+kh)*KW + kw]
        let k_scalar = (c_in_local * KH + kh) * KW + kw;
        let w_offsets = c_out_range * (C_IN * KH * KW) + k_scalar;
        let w_row = T::load(
            w_ptr.add_offsets(w_offsets),
            Some(n_mask),
            Some(T::zeros::<f32>(&[BLOCK_N])),
            &[],
            None, None, None, false,
        );

        // ── Outer product: w_row[:,None] * x_col[None,:] → [BLOCK_N, BLOCK_OW] ─
        let w_2d = T::broadcast_to(T::expand_dims(w_row, 1), &[BLOCK_N, BLOCK_OW]);
        let x_2d = T::broadcast_to(T::expand_dims(x_col, 0), &[BLOCK_N, BLOCK_OW]);
        acc = acc + w_2d * x_2d;
    }

    // ── BatchNorm epilog ──────────────────────────────────────────────────────
    let bn_scale = T::load(
        bn_scale_ptr.add_offsets(c_out_range),
        Some(n_mask),
        Some(T::zeros::<f32>(&[BLOCK_N])),
        &[], None, None, None, false,
    );
    let bn_shift = T::load(
        bn_shift_ptr.add_offsets(c_out_range),
        Some(n_mask),
        Some(T::zeros::<f32>(&[BLOCK_N])),
        &[], None, None, None, false,
    );
    let scale_2d = T::broadcast_to(T::expand_dims(bn_scale, 1), &[BLOCK_N, BLOCK_OW]);
    let shift_2d = T::broadcast_to(T::expand_dims(bn_shift, 1), &[BLOCK_N, BLOCK_OW]);
    let bn_out = scale_2d * acc + shift_2d;

    // ── SiLU epilog: y = x * sigmoid(x) ─────────────────────────────────────
    let y = bn_out * T::sigmoid(bn_out);

    // ── Store via TMA descriptor ──────────────────────────────────────────────
    // Output layout [B*C_OUT, OH * y_col_stride]:
    //   y[b, c_out, oh, ow] = y_flat[(b*C_OUT + c_out) * OH * y_col_stride
    //                                 + oh * y_col_stride + ow]
    //
    // y_col_stride >= BLOCK_OW ensures adjacent oh tiles never overlap.
    // y_col_stride divisible by 4 ensures store positions oh*y_col_stride are
    // 16-byte (4-float) aligned for the v2.b32 store the compiler generates.
    let oh_ycs = OH * y_col_stride;
    let y_desc = T::make_tensor_descriptor(
        y_ptr,
        &[B * C_OUT, oh_ycs],
        &[oh_ycs, 1],
        &[BLOCK_N, BLOCK_OW],
        Some(PaddingOption::Zero),
    );
    T::store_tensor_descriptor(
        y_desc,
        &[b * C_OUT + c_out_start, oh * y_col_stride + ow_start],
        y,
    );
}

// ── RuntimeOp ────────────────────────────────────────────────────────────────
//
// Params layout: [weight [C_OUT, C_IN, KH, KW], bn_scale [C_OUT], bn_shift [C_OUT]]
// pack_args order: x_ptr, w_ptr, bn_scale_ptr, bn_shift_ptr, y_ptr,
//                  B, C_IN, C_OUT, H, W, OH, OW, y_col_stride

impl teeny_core::model::RuntimeOp for Conv2dBnSiluTiledForward {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c_in  = input_shapes[0][1];
        let c_out = output_shape[1];
        vec![
            vec![c_out, c_in, self.kh as usize, self.kw as usize],
            vec![c_out],
            vec![c_out],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bn_scale", "bn_shift"]
    }

    // y_col_stride is the padded column width of each OH row in the output buffer.
    // Must be a multiple of BLOCK_OW to guarantee that the BLOCK_OW-wide TMA store
    // tile for the last OW tile never crosses into the next oh row's data.
    //
    // Example: OW=40, BLOCK_OW=16 → 3 tiles (ow_start = 0, 16, 32).
    //   Tile 2 stores [BLOCK_OW=16] columns at oh*stride+32..oh*stride+47.
    //   With stride=40: oh*40+40 = (oh+1)*40, so the last 8 columns overwrite
    //   (oh+1)'s first 8 positions. With stride=48 (next multiple of 16):
    //   oh*48+47 < oh*48+48 = (oh+1)*48 — fully within oh's padding. ✓
    //
    // The runtime row-stride contract expects the stride between adjacent
    // last-dimension rows (each of natural_stride = OW elements), which is
    // exactly y_col_stride — NOT OH * y_col_stride.
    fn forward_output_row_stride(&self, output_shape: &[usize]) -> usize {
        let ow = output_shape[3];
        ow.next_multiple_of(self.block_ow as usize)
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        let oh = output_shape[2] as i32;
        // output_row_stride == y_col_stride (the per-oh-row column stride).
        let y_col_stride = output_row_stride;
        visitor.visit_ptr(inputs[0].0);              // x_ptr
        visitor.visit_ptr(params[0]);                // w_ptr
        visitor.visit_ptr(params[1]);                // bn_scale_ptr
        visitor.visit_ptr(params[2]);                // bn_shift_ptr
        visitor.visit_ptr(output);                   // y_ptr
        visitor.visit_i32(input_shape[0] as i32);    // B
        visitor.visit_i32(input_shape[1] as i32);    // C_IN
        visitor.visit_i32(output_shape[1] as i32);   // C_OUT
        visitor.visit_i32(input_shape[2] as i32);    // H
        visitor.visit_i32(input_shape[3] as i32);    // W
        visitor.visit_i32(oh);                       // OH
        visitor.visit_i32(output_shape[3] as i32);   // OW
        visitor.visit_i32(y_col_stride);             // y_col_stride
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        let num_n_tiles  = output_shape[1].div_ceil(self.block_n as usize);
        [
            (output_shape[0] * output_shape[2] * num_n_tiles * num_ow_tiles) as u32,
            1, 1,
        ]
    }
}
