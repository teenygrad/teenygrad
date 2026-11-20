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

//! `GridSample`, bilinear, over an NCHW tensor.
//!
//! Each output element reads a position the grid gives it, in normalised coordinates
//! that run from -1 at one edge of the input to +1 at the other, and takes the four
//! elements around that position weighted by distance. Unlike [`super::resize`] the
//! positions are not known when the graph is lowered: the grid is computed, so the
//! coordinate arithmetic has to happen per lane on the device.
//!
//! What the host does precompute is the two numbers that turn a normalised coordinate
//! into an index, `scale` and `offset`, which is where `align_corners` lives: with it
//! the edges of the input are the centres of its first and last elements, and without
//! it they are the outer faces of those elements. One multiply-add either way, and no
//! flag in the kernel.
//!
//! Only the zero padding mode is implemented: a position outside the input contributes
//! nothing, which is what the masked load does already.

#![allow(non_snake_case)]

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::Float;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Forward: bilinear `GridSample` with zero padding.
///
/// `x_scale`/`x_offset` and `y_scale`/`y_offset` turn a normalised coordinate into an
/// input index: `i = g * scale + offset`.
#[kernel]
pub fn grid_sample_forward<T, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    g_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    C: i32,
    H_in: i32,
    W_in: i32,
    H_out: i32,
    W_out: i32,
    x_scale: f32,
    x_offset: f32,
    y_scale: f32,
    y_offset: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // The output is [N, C, H_out, W_out]. There is no integer remainder in the kernel
    // DSL, so each coordinate is `a - (a / b) * b`.
    let q_w = p / W_out;
    let w_o = p - q_w * W_out;
    let q_h = q_w / H_out;
    let h_o = q_w - q_h * H_out;
    let q_c = q_h / C;
    let c = q_h - q_c * C;
    let n = q_c;

    // The grid is [N, H_out, W_out, 2], one position per output pixel, shared by
    // every channel.
    let g_base = ((n * H_out + h_o) * W_out + w_o) * 2;
    let gx = T::load(
        g_ptr.add_offsets(g_base),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let gy = T::load(
        g_ptr.add_offsets(g_base + 1),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // Normalised coordinates to input indices.
    let x = gx * T::full(&[BLOCK_SIZE], D::from_f64(x_scale as f64))
        + T::full(&[BLOCK_SIZE], D::from_f64(x_offset as f64));
    let y = gy * T::full(&[BLOCK_SIZE], D::from_f64(y_scale as f64))
        + T::full(&[BLOCK_SIZE], D::from_f64(y_offset as f64));

    let x0f = T::floor(x);
    let y0f = T::floor(y);
    let wx1 = x - x0f;
    let wy1 = y - y0f;
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let wx0 = one - wx1;
    let wy0 = one - wy1;
    let x0 = T::cast::<D, i32>(x0f, None, false);
    let y0 = T::cast::<D, i32>(y0f, None, false);

    // The plane this lane reads from, and the four positions around (y, x). Each tap
    // is either inside the input or nothing, and zero padding is a masked load.
    let plane = (n * C + c) * H_in * W_in;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let x0_ok = x0.ge(0).bitand(x0.lt(W_in));
    let x1_ok = x1.ge(0).bitand(x1.lt(W_in));
    let y0_ok = y0.ge(0).bitand(y0.lt(H_in));
    let y1_ok = y1.ge(0).bitand(y1.lt(H_in));

    let v00 = T::load(
        x_ptr.add_offsets(plane + y0 * W_in + x0),
        Some(in_bounds.bitand(y0_ok).bitand(x0_ok)),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let v01 = T::load(
        x_ptr.add_offsets(plane + y0 * W_in + x1),
        Some(in_bounds.bitand(y0_ok).bitand(x1_ok)),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let v10 = T::load(
        x_ptr.add_offsets(plane + y1 * W_in + x0),
        Some(in_bounds.bitand(y1_ok).bitand(x0_ok)),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let v11 = T::load(
        x_ptr.add_offsets(plane + y1 * W_in + x1),
        Some(in_bounds.bitand(y1_ok).bitand(x1_ok)),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let acc = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1;
    T::store(
        out_ptr.add_offsets(p),
        acc,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Dispatch for [`grid_sample_forward`].
pub struct GridSampleRuntimeOp<D: Float + Send + Sync + 'static> {
    kernel: GridSampleForward<D>,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    x_scale: f32,
    x_offset: f32,
    y_scale: f32,
    y_offset: f32,
    n_out: usize,
}

impl<D: Float + Send + Sync + 'static> GridSampleRuntimeOp<D> {
    /// `None` if the input has no extent to sample along.
    ///
    /// `align_corners` chooses what -1 and +1 mean: the centres of the first and last
    /// elements, or the outer faces of them. Either way it is a multiply and an add,
    /// worked out here so the kernel needs no flag.
    pub fn new(
        block_size: i32,
        n: usize,
        c: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        align_corners: bool,
    ) -> Option<Self> {
        if h_in == 0 || w_in == 0 {
            return None;
        }
        let map = |extent: usize| -> (f32, f32) {
            if align_corners {
                // -1 and 1 are the centres of the end elements.
                let half = (extent as f32 - 1.0) / 2.0;
                (half, half)
            } else {
                // -1 and 1 are the outer faces, so a half-element inset either side.
                (extent as f32 / 2.0, (extent as f32 - 1.0) / 2.0)
            }
        };
        let (x_scale, x_offset) = map(w_in);
        let (y_scale, y_offset) = map(h_in);
        Some(Self {
            kernel: GridSampleForward::<D>::new(block_size),
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            x_scale,
            x_offset,
            y_scale,
            y_offset,
            n_out: n * c * h_out * w_out,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &GridSampleForward<D> {
        &self.kernel
    }
}

impl<D: Float + Send + Sync + 'static> RuntimeOp for GridSampleRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n_out as i32);
        visitor.visit_i32(self.c as i32);
        visitor.visit_i32(self.h_in as i32);
        visitor.visit_i32(self.w_in as i32);
        visitor.visit_i32(self.h_out as i32);
        visitor.visit_i32(self.w_out as i32);
        visitor.visit_f32(self.x_scale);
        visitor.visit_f32(self.x_offset);
        visitor.visit_f32(self.y_scale);
        visitor.visit_f32(self.y_offset);
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n_out.div_ceil(self.kernel.block_size as usize) as u32).max(1),
            1,
            1,
        ]
    }
}
