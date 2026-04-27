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

/// 1-D convolution forward pass.
///
/// Grid: one CTA per (b, c_out, ol-tile):
///   `pid = (b * C_OUT + c_out) * num_ol_tiles + ol_tile`
///
/// Each CTA computes a BLOCK_OL-wide strip of output positions by iterating
/// over all `C_IN * KL` combinations.
///
/// **Constraints**: no padding; `OL = (L - KL) / STRIDE + 1`.
#[kernel]
pub fn conv1d_forward<
    T: Triton,
    D: Num,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
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
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ol_start = ol_tile * BLOCK_OL;
    let ol_range = T::arange(0, BLOCK_OL) + ol_start;
    let ol_mask = ol_range.lt(OL);

    let out_bc_base = (b * C_OUT + c_out) * OL;

    let mut acc = T::zeros::<D>(&[BLOCK_OL]);

    let loop_bound = C_IN * KL;
    for idx in 0..loop_bound {
        let kl = idx % KL;
        let c_in = idx / KL;

        let il_range = ol_range * STRIDE + kl;
        let x_offsets = il_range + (b * C_IN + c_in) * L;
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(ol_mask),
            Some(T::zeros::<D>(&[BLOCK_OL])),
            &[],
            None,
            None,
            None,
            false,
        );

        let w_idx = (c_out * C_IN + c_in) * KL + kl;
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
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OL]);

        acc = acc + x_tile * w_tile;
    }

    let out_offsets = ol_range + out_bc_base;
    T::store(
        y_ptr.add_offsets(out_offsets),
        acc,
        Some(ol_mask),
        &[],
        None,
        None,
    );
}

/// 1-D convolution backward pass — gradient with respect to input (`dx`).
///
/// Grid: `pid = (b * C_OUT + c_out) * num_ol_tiles + ol_tile`
///
/// Scatters gradient back via `atomic_add` to handle overlapping receptive fields.
#[kernel]
pub fn conv1d_backward_dx<
    T: Triton,
    D: Num,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    dy_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
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
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ol_start = ol_tile * BLOCK_OL;
    let ol_range = T::arange(0, BLOCK_OL) + ol_start;
    let ol_mask = ol_range.lt(OL);

    let dy_offsets = ol_range + (b * C_OUT + c_out) * OL;
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

    let loop_bound = C_IN * KL;
    for idx in 0..loop_bound {
        let kl = idx % KL;
        let c_in = idx / KL;

        let w_idx = (c_out * C_IN + c_in) * KL + kl;
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
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OL]);

        let grad_tile = dy_tile * w_tile;

        let il_range = ol_range * STRIDE + kl;
        let dx_offsets = il_range + (b * C_IN + c_in) * L;
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad_tile,
            Some(ol_mask),
            None,
            None,
        );
    }
}

/// 1-D convolution backward pass — gradient with respect to weights (`dw`).
///
/// Grid: `pid = (b * C_OUT + c_out) * num_ol_tiles + ol_tile`
///
/// `dw` must be zero-initialised before launch.
#[kernel]
pub fn conv1d_backward_dw<
    T: Triton,
    D: Num,
    const KL: i32,
    const STRIDE: i32,
    const BLOCK_OL: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dw_ptr: T::Pointer<D>,
    B: i32,
    C_IN: i32,
    C_OUT: i32,
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
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ol_start = ol_tile * BLOCK_OL;
    let ol_range = T::arange(0, BLOCK_OL) + ol_start;
    let ol_mask = ol_range.lt(OL);

    let dy_offsets = ol_range + (b * C_OUT + c_out) * OL;
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

    let loop_bound = C_IN * KL;
    for idx in 0..loop_bound {
        let kl = idx % KL;
        let c_in = idx / KL;

        let il_range = ol_range * STRIDE + kl;
        let x_offsets = il_range + (b * C_IN + c_in) * L;
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(ol_mask),
            Some(T::zeros::<D>(&[BLOCK_OL])),
            &[],
            None,
            None,
            None,
            false,
        );

        let partial = T::sum(dy_tile * x_tile, Some(0), false);
        let partial_1 = T::expand_dims(partial, 0);

        let w_idx = (c_out * C_IN + c_in) * KL + kl;
        let dw_off = T::arange(0, 1) + w_idx;
        T::atomic_add(dw_ptr.add_offsets(dw_off), partial_1, None, None, None);
    }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for Conv1dForward<D> {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        // input_shapes[0] = [B, C_IN, L], output_shape = [B, C_OUT, OL]
        let c_in = input_shapes[0][1];
        let c_out = output_shape[1];
        vec![vec![c_out, c_in, self.kl as usize]]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // kernel args: x_ptr, w_ptr, y_ptr, B, C_IN, C_OUT, L, OL
        let input_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(input_shape[0] as i32); // B
        visitor.visit_i32(input_shape[1] as i32); // C_IN
        visitor.visit_i32(output_shape[1] as i32); // C_OUT
        visitor.visit_i32(input_shape[2] as i32); // L
        visitor.visit_i32(output_shape[2] as i32); // OL
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let num_ol_tiles = output_shape[2].div_ceil(self.block_ol as usize);
        [(output_shape[0] * output_shape[1] * num_ol_tiles) as u32, 1, 1]
    }
}

pub struct Conv1dOp<'a, T: Num> {
    pub forward: Conv1dForward<T>,
    pub backward_dx: Conv1dBackwardDx<T>,
    pub backward_dw: Conv1dBackwardDw<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
