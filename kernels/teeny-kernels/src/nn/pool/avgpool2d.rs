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

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 2-D average-pooling forward pass.
///
/// Grid: one flat 1-D pid per (b, c, oh, ow-tile) combination:
///   pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile
///
/// Each CTA accumulates a BLOCK_OW-wide strip of output columns via a flat
/// loop over all KH×KW kernel positions (avoids nested scf.for).
///
/// **Constraints**: no padding; `OH = (H - KH) / STRIDE_H + 1`, `OW = (W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn avgpool2d_forward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
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

    // Decode flat pid → (b, c, oh, ow_tile).
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

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    // Flat loop over KH * KW kernel positions to avoid nested scf.for.
    let loop_bound = KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh = idx / KW;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let in_offsets = iw_range + (in_bc_base + ih * W);
        let tile = T::load(
            input_ptr.add_offsets(in_offsets),
            Some(ow_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + tile;
    }

    // Scale by 1/(KH*KW): build [1] i32 tensor from arange then broadcast.
    let ksize_1 = T::full::<i32>(&[1], KH * KW);
    let ksize_f_1 = T::cast::<i32, D>(ksize_1, None, false);
    let ksize = T::broadcast_to(ksize_f_1, &[BLOCK_OW]);
    let result = acc / ksize;

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        output_ptr.add_offsets(out_offsets),
        result,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 2-D average-pooling backward pass.
///
/// Uses the same flat 1-D grid as the forward pass, iterating over output
/// positions. For each output element dy[b, c, oh, ow], the gradient is
/// spread uniformly across the KH×KW input window via `atomic_add` so that
/// overlapping pooling windows (stride < kernel size) are handled correctly.
///
/// **Constraints**: `dx` must be zero-initialised before launch; same spatial
/// constraints as `avgpool2d_forward`.
#[kernel]
pub fn avgpool2d_backward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
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

    let dy_bc_base = (b * C + c) * OH * OW;
    let dx_bc_base = (b * C + c) * H * W;

    // Load upstream gradient tile and scale by 1/(KH*KW).
    let dy_offsets = ow_range + (dy_bc_base + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );
    let ksize_1 = T::full::<i32>(&[1], KH * KW);
    let ksize_f_1 = T::cast::<i32, D>(ksize_1, None, false);
    let ksize = T::broadcast_to(ksize_f_1, &[BLOCK_OW]);
    let grad = dy_tile / ksize;

    // Scatter scaled gradient to the KH×KW input window via flat loop.
    let loop_bound = KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh = idx / KW;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let dx_offsets = iw_range + (dx_bc_base + ih * W);
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad,
            Some(ow_mask),
            None,
            None,
        );
    }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for Avgpool2dForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // kernel args: input_ptr, output_ptr, B, C, H, W, OH, OW
        // input_shape = [B, C, H, W], output_shape = [B, C, OH, OW]
        let input_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(input_shape[0] as i32);   // B
        visitor.visit_i32(input_shape[1] as i32);   // C
        visitor.visit_i32(input_shape[2] as i32);   // H
        visitor.visit_i32(input_shape[3] as i32);   // W
        visitor.visit_i32(output_shape[2] as i32);  // OH
        visitor.visit_i32(output_shape[3] as i32);  // OW
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [(output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32, 1, 1]
    }
}

pub struct Avgpool2dOp<'a, T: Num> {
    pub forward: Avgpool2dForward<T>,
    pub backward: Avgpool2dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
