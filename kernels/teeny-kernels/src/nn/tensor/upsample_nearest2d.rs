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

/// Nearest-neighbour 2-D upsample forward pass — NCHW layout.
///
/// Grid: one flat pid per (b, c, oh, ow-tile):
///   `pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Each CTA writes a BLOCK_OW-wide strip of output columns by reading from
/// the nearest input position via floor division:
///   `ih = oh / SCALE_H`,  `iw = ow / SCALE_W`
///
/// Output shape: `[B, C, OH=H*SCALE_H, OW=W*SCALE_W]`.
#[kernel]
pub fn upsample_nearest2d_forward<
    T: Triton,
    D: Num,
    const SCALE_H: i32,
    const SCALE_W: i32,
    const BLOCK_OW: i32,
>(
    x_ptr:   T::Pointer<D>,  // input  [B, C, H, W]
    y_ptr:   T::Pointer<D>,  // output [B, C, OH, OW]
    _B: i32,
    C:  i32,
    H:  i32,
    W:  i32,
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
    let bco     = pid / num_ow_tiles;
    let oh      = bco % OH;
    let bc      = bco / OH;
    let c       = bc % C;
    let b       = bc / C;

    // Nearest-neighbour source row (scalar).
    let ih = oh / SCALE_H;

    let ow_start  = ow_tile * BLOCK_OW;
    let ow_range  = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask   = ow_range.lt(OW);

    // Nearest-neighbour source column per output lane: iw = ow / SCALE_W.
    let iw_range = ow_range / SCALE_W;

    let in_bc_base  = (b * C + c) * H * W;
    let out_bc_base = (b * C + c) * OH * OW;

    let in_offsets  = iw_range + (in_bc_base + ih * W);
    let out_offsets = ow_range + (out_bc_base + oh * OW);

    let x = T::load(
        x_ptr.add_offsets(in_offsets),
        Some(ow_mask),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(out_offsets), x, Some(ow_mask), &[], None, None);
}

/// Nearest-neighbour 2-D upsample backward pass — NCHW layout.
///
/// Grid: one flat pid per (b, c, ih, iw-tile) — same spatial extent as the
/// *input* tensor:
///   `pid = ((b * C + c) * H + ih) * num_iw_tiles + iw_tile`
///
/// Each CTA computes the gradient for a BLOCK_IW-wide strip of input columns
/// by summing the SCALE_H × SCALE_W upstream gradients that each input pixel
/// received during the forward pass.  No atomic operations are needed because
/// each input element is the sole accumulator for exactly SCALE_H×SCALE_W
/// output lanes (their ranges are disjoint).
///
/// `dx` does NOT need to be zero-initialised (every element is written once).
#[kernel]
pub fn upsample_nearest2d_backward<
    T: Triton,
    D: Num,
    const SCALE_H: i32,
    const SCALE_W: i32,
    const BLOCK_IW: i32,
>(
    dy_ptr: T::Pointer<D>,  // upstream grad [B, C, OH, OW]
    dx_ptr: T::Pointer<D>,  // input    grad [B, C, H,  W]
    _B: i32,
    C:  i32,
    H:  i32,
    W:  i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_iw_tiles = T::cdiv(W, BLOCK_IW);

    // Decode flat pid → (b, c, ih, iw_tile).
    let iw_tile = pid % num_iw_tiles;
    let bch     = pid / num_iw_tiles;
    let ih      = bch % H;
    let bc      = bch / H;
    let c       = bc % C;
    let b       = bc / C;

    let iw_start  = iw_tile * BLOCK_IW;
    let iw_range  = T::arange(0, BLOCK_IW) + iw_start;
    let iw_mask   = iw_range.lt(W);

    let dy_bc_base = (b * C + c) * OH * OW;
    let dx_bc_base = (b * C + c) * H * W;

    let mut acc = T::zeros::<D>(&[BLOCK_IW]);

    // Flat loop over SCALE_H × SCALE_W upstream gradient positions.
    let loop_bound = SCALE_H * SCALE_W;
    for idx in 0..loop_bound {
        let sw = idx % SCALE_W;
        let sh = idx / SCALE_W;
        let oh       = ih * SCALE_H + sh;                   // scalar row
        let ow_range = iw_range * SCALE_W + sw;             // tensor cols
        let dy_offsets = ow_range + (dy_bc_base + oh * OW);
        let dy_mask  = ow_range.lt(OW);
        let tile = T::load(
            dy_ptr.add_offsets(dy_offsets),
            Some(dy_mask),
            Some(T::zeros::<D>(&[BLOCK_IW])),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + tile;
    }

    let dx_offsets = iw_range + (dx_bc_base + ih * W);
    T::store(dx_ptr.add_offsets(dx_offsets), acc, Some(iw_mask), &[], None, None);
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp
    for UpsampleNearest2dForward<D>
{
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(
        &self,
        _input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> Vec<Vec<usize>> {
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
        // kernel args: x_ptr, y_ptr, B, C, H, W, OH, OW
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
        // pid = ((b * C + c) * OH + oh) * num_ow_tiles + ow_tile
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [(output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32, 1, 1]
    }
}

pub struct UpsampleNearest2dOp<'a, D: Num> {
    pub forward:  UpsampleNearest2dForward<D>,
    pub backward: UpsampleNearest2dBackward<D>,
    _marker: PhantomData<&'a ()>,
}
