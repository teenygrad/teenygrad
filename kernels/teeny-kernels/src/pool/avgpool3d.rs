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

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 3-D average-pooling forward pass.
///
/// Grid: `pid = (((b * C + c) * OD + od) * OH + oh) * num_ow_tiles + ow_tile`
///
/// **Constraints**: no padding;
/// `OD = (D - KD) / STRIDE_D + 1`, `OH = (H - KH) / STRIDE_H + 1`,
/// `OW = (W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn avgpool3d_forward<
    T: Triton,
    D: Float,
    const KD: i32,
    const KH: i32,
    const KW: i32,
    const STRIDE_D: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    input_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
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
    let rest = pid / num_ow_tiles;
    let oh = rest % OH;
    let rest2 = rest / OH;
    let od = rest2 % OD;
    let bco = rest2 / OD;
    let c = bco % C;
    let b = bco / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let in_bc_base = (b * C + c) * Dv * H * W;
    let out_base = ((b * C + c) * OD * OH * OW) + od * OH * OW + oh * OW;

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    let loop_bound = KD * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let tmp = idx / KW;
        let kh = tmp % KH;
        let kd = tmp / KH;

        let id = od * STRIDE_D + kd;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let in_offsets = iw_range + (in_bc_base + id * H * W + ih * W);
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

    let ksize_1 = T::full::<i32>(&[1], KD * KH * KW);
    let ksize_f_1 = T::cast::<i32, D>(ksize_1, None, false);
    let ksize = T::broadcast_to(ksize_f_1, &[BLOCK_OW]);
    let result = acc / ksize;

    let out_offsets = ow_range + out_base;
    T::store(
        output_ptr.add_offsets(out_offsets),
        result,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 3-D average-pooling backward pass.
///
/// Grid: `pid = (((b * C + c) * OD + od) * OH + oh) * num_ow_tiles + ow_tile`
///
/// `dx` must be zero-initialised before launch.
#[kernel]
pub fn avgpool3d_backward<
    T: Triton,
    D: Float,
    const KD: i32,
    const KH: i32,
    const KW: i32,
    const STRIDE_D: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C: i32,
    Dv: i32,
    H: i32,
    W: i32,
    OD: i32,
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
    let rest = pid / num_ow_tiles;
    let oh = rest % OH;
    let rest2 = rest / OH;
    let od = rest2 % OD;
    let bco = rest2 / OD;
    let c = bco % C;
    let b = bco / C;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let dy_base = ((b * C + c) * OD * OH * OW) + od * OH * OW + oh * OW;
    let dx_bc_base = (b * C + c) * Dv * H * W;

    let dy_offsets = ow_range + dy_base;
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
    let ksize_1 = T::full::<i32>(&[1], KD * KH * KW);
    let ksize_f_1 = T::cast::<i32, D>(ksize_1, None, false);
    let ksize = T::broadcast_to(ksize_f_1, &[BLOCK_OW]);
    let grad = dy_tile / ksize;

    let loop_bound = KD * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let tmp = idx / KW;
        let kh = tmp % KH;
        let kd = tmp / KH;

        let id = od * STRIDE_D + kd;
        let ih = oh * STRIDE_H + kh;
        let iw_range = ow_range * STRIDE_W + kw;
        let dx_offsets = iw_range + (dx_bc_base + id * H * W + ih * W);
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad,
            Some(ow_mask),
            None,
            None,
        );
    }
}

pub struct Avgpool3dOp<'a, T: Float> {
    pub forward: Avgpool3dForward<T>,
    pub backward: Avgpool3dBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
