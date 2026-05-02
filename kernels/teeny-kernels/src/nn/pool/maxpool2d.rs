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

use core::marker::PhantomData;

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 2-D max-pooling forward pass with optional symmetric padding.
///
/// Grid: `pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile`
///
/// `OH = (H + 2*PAD_H - KH) / STRIDE_H + 1`, `OW = (W + 2*PAD_W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn maxpool2d_forward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const BLOCK_OW: i32,
>(
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c = bc % C;
    let b = bc / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let in_bc_base = (b * C + c) * H * W;
    let out_bc_base = (b * C + c) * OH * OW;

    let mut acc = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OW], -3.4028235e38_f32), None, false);

    // The Triton MLIR frontend only generates correct scf.while (with body) for
    // single-variable, single-condition loops with no nested control flow.
    // A nested `for kw` inside `while kh` creates extra basic blocks that break
    // the outer loop's structural conversion — the body gets dropped silently.
    //
    // Solution: flatten (kh, kw) into a single linear loop `iter < n_valid_kh * KW`,
    // recover kh/kw per-iteration via div/rem (KW is a compile-time constant).
    //
    // Phase 1: skip-loop — advance kh until ih = oh*STRIDE_H + kh - PAD_H >= 0.
    let mut kh: i32 = 0;
    let mut ih: i32 = oh * STRIDE_H - PAD_H;
    while ih < 0 {
        kh += 1;
        ih += 1;  // each kh step advances ih by exactly 1
    }
    // Phase 2: clamp kh_hi = min(kh + (H - ih), KH) via countdown.
    let mut kh_hi: i32 = kh + (H - ih);
    while kh_hi > KH {
        kh_hi -= 1;
    }
    // Phase 3: flat loop over all (kh, kw) pairs — matches BN kernel structure
    // (single variable `iter`, single condition, vector body, no nested loops).
    // If kh_hi <= kh (no valid rows), total_iters <= 0 → loop runs 0 times.
    let total_iters = (kh_hi - kh) * KW;
    let ih_lo = ih;
    let mut iter: i32 = 0;
    while iter < total_iters {
        let kw_idx = iter % KW;
        let ih_local = ih_lo + iter / KW;
        let iw_range = ow_range * STRIDE_W + kw_idx - PAD_W;
        let iw_valid = iw_range.ge(0) & iw_range.lt(W);
        let valid_mask = ow_mask & iw_valid;
        let in_offsets = iw_range + (in_bc_base + ih_local * W);
        let tile = T::load(
            input_ptr.add_offsets(in_offsets),
            Some(valid_mask),
            Some(T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OW], -3.4028235e38_f32), None, false)),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = T::maximum(acc, tile);
        iter += 1;
    }

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        output_ptr.add_offsets(out_offsets),
        acc,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 2-D max-pooling backward pass with optional symmetric padding.
///
/// Re-scans the input window and scatters `dy` to positions where
/// `input == output_max`. `dx` must be zero-initialised before launch.
#[kernel]
pub fn maxpool2d_backward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c = bc % C;
    let b = bc / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let in_bc_base = (b * C + c) * H * W;
    let out_bc_base = (b * C + c) * OH * OW;

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(out_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );
    let y_tile = T::load(
        y_ptr.add_offsets(out_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );

    let mut kh: i32 = 0;
    let mut ih: i32 = oh * STRIDE_H - PAD_H;
    while ih < 0 {
        kh += 1;
        ih += 1;
    }
    let mut kh_hi: i32 = kh + (H - ih);
    while kh_hi > KH {
        kh_hi -= 1;
    }
    let total_iters = (kh_hi - kh) * KW;
    let ih_lo = ih;
    let mut iter: i32 = 0;
    while iter < total_iters {
        let kw_idx = iter % KW;
        let ih_local = ih_lo + iter / KW;
        let iw_range = ow_range * STRIDE_W + kw_idx - PAD_W;
        let iw_valid = iw_range.ge(0) & iw_range.lt(W);
        let valid_mask = ow_mask & iw_valid;
        let in_offsets = iw_range + (in_bc_base + ih_local * W);
        let x_tile = T::load(
            x_ptr.add_offsets(in_offsets),
            Some(valid_mask),
            Some(T::cast::<f32, D>(T::full::<f32>(&[BLOCK_OW], -3.4028235e38_f32), None, false)),
            &[],
            None,
            None,
            None,
            false,
        );
        let is_max = T::eq(x_tile, y_tile);
        let grad = T::where_(is_max, dy_tile, T::zeros::<D>(&[BLOCK_OW]));
        T::atomic_add(
            dx_ptr.add_offsets(in_offsets),
            grad,
            Some(valid_mask),
            None,
            None,
        );
        iter += 1;
    }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for Maxpool2dForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(input_shape[0] as i32);  // B
        visitor.visit_i32(input_shape[1] as i32);  // C
        visitor.visit_i32(input_shape[2] as i32);  // H
        visitor.visit_i32(input_shape[3] as i32);  // W
        visitor.visit_i32(output_shape[2] as i32); // OH
        visitor.visit_i32(output_shape[3] as i32); // OW
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [(output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32, 1, 1]
    }
}

pub struct Maxpool2dOp<'a, T: Num> {
    pub forward: Maxpool2dForward<T>,
    pub backward: Maxpool2dBackward<T>,
    _marker: PhantomData<&'a ()>,
}
