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

//! `Pad` with a constant, over any number of axes.
//!
//! The existing pad kernels are shaped for images: [`super::constant_pad2d`] pads the
//! last two axes of an NCHW tensor. An export pads whatever it likes, and the ones here
//! are rank 4 and rank 5 with the padding on a middle axis, so this walks the output
//! and works out where each element came from, one axis at a time.
//!
//! Each output lane peels its coordinates from the output's extents, innermost first,
//! shifts each by that axis' leading pad, and is inside the input only if every shifted
//! coordinate landed within it. Outside, the lane takes the pad value, which arrives in
//! a parameter slot because its type is the tensor's and a kernel argument would need
//! one signature per dtype.
//!
//! The axis descriptions live in a parameter slot too, so the rank is not capped by the
//! kernel's arity: four numbers per axis, innermost first, which is the order the
//! coordinates peel in.

#![allow(non_snake_case)]

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// How many numbers describe one axis in the `meta` slot.
pub const PER_AXIS: usize = 4;

/// Forward: `out[c] = src[c - lo]` where that is inside the input, else the pad value.
///
/// `meta_ptr` holds, innermost axis first and four numbers per axis: the output's
/// extent, the input's extent, the input's stride, and the leading pad.
#[kernel]
pub fn pad_nd_forward<T, D: Num, const BLOCK_SIZE: i32>(
    src_ptr: InPtr<T::Pointer<D>>,
    value_ptr: InPtr<T::Pointer<D>>,
    meta_ptr: InPtr<T::Pointer<i32>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    rank: i32,
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
    T::I32Tensor: core::ops::Div<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let lane_ok = p.lt(n_elements);

    let zero = T::full::<i32>(&[BLOCK_SIZE], 0);
    let one = T::full::<i32>(&[BLOCK_SIZE], 1);

    let mut rest = p;
    let mut src = T::full::<i32>(&[BLOCK_SIZE], 0);
    // 1 while every axis so far landed inside the input, 0 once one did not.
    let mut inside = T::full::<i32>(&[BLOCK_SIZE], 1);

    for axis in 0..rank {
        let base = axis * 4;
        let at_out = T::full::<i32>(&[BLOCK_SIZE], base);
        let at_in = T::full::<i32>(&[BLOCK_SIZE], base + 1);
        let at_stride = T::full::<i32>(&[BLOCK_SIZE], base + 2);
        let at_lo = T::full::<i32>(&[BLOCK_SIZE], base + 3);
        let out_ext = T::load(
            meta_ptr.add_offsets(at_out),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let in_ext = T::load(
            meta_ptr.add_offsets(at_in),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let stride = T::load(
            meta_ptr.add_offsets(at_stride),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let lo = T::load(
            meta_ptr.add_offsets(at_lo),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // This axis' output coordinate, innermost first. There is no integer
        // remainder in the kernel DSL, so it is `a - (a / b) * b`.
        let q = rest / out_ext;
        let c = rest - q * out_ext;
        rest = q;

        let ci = c - lo;
        // 1 when 0 <= ci < in_ext, without a branch: each factor saturates to 0 or 1.
        let above = T::maximum(T::minimum(ci + 1, one), zero);
        let below = T::maximum(T::minimum(in_ext - ci, one), zero);
        inside = inside * above * below;
        src = src + ci * stride;
    }

    // Reading outside the input is what `inside` prevents; the mask also keeps the
    // lanes past the end of the output from loading at all.
    let take = T::ne(inside * T::where_(p.lt(n_elements), one, zero), zero);
    let v = T::load(
        src_ptr.add_offsets(src),
        Some(take),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    // The pad value, read into every lane from element 0 of its slot.
    let fill = T::load(
        value_ptr.add_offsets(zero),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(p),
        T::where_(take, v, fill),
        Some(lane_ok),
        &[],
        None,
        None,
    );
}

/// Dispatch for [`pad_nd_forward`].
///
/// `axes` describes the output innermost-axis-first: each entry is the output extent,
/// the input extent, the input's stride, and the leading pad.
pub struct PadNdRuntimeOp<D: Num + Send + Sync + 'static> {
    kernel: PadNdForward<D>,
    meta: Vec<i32>,
    value: Vec<u8>,
    rank: usize,
    n_out: usize,
}

impl<D: Num + Send + Sync + 'static> PadNdRuntimeOp<D> {
    /// `out_shape`, `in_shape` and `lo` describe the axes outermost-first, as a shape
    /// is written; `value` is the pad value's little-endian bytes, or empty for zero.
    ///
    /// `None` if the three do not describe the same rank, if an axis has no extent, or
    /// if the pad value is neither empty nor one element wide.
    pub fn new(
        block_size: i32,
        out_shape: &[usize],
        in_shape: &[usize],
        lo: &[i64],
        value: &[u8],
    ) -> Option<Self> {
        let rank = out_shape.len();
        if rank == 0 || in_shape.len() != rank || lo.len() != rank {
            return None;
        }
        if out_shape.contains(&0) {
            return None;
        }
        let width = core::mem::size_of::<D>();
        let value = if value.is_empty() {
            vec![0u8; width]
        } else if value.len() == width {
            value.to_vec()
        } else {
            return None;
        };

        // Row-major strides of the input.
        let mut strides = vec![1usize; rank];
        for ax in (0..rank.saturating_sub(1)).rev() {
            strides[ax] = strides[ax + 1] * in_shape[ax + 1];
        }

        // Innermost axis first, which is the order the kernel peels coordinates in.
        let mut meta = Vec::with_capacity(rank * PER_AXIS);
        for ax in (0..rank).rev() {
            meta.push(out_shape[ax] as i32);
            meta.push(in_shape[ax] as i32);
            meta.push(strides[ax] as i32);
            meta.push(lo[ax] as i32);
        }

        Some(Self {
            kernel: PadNdForward::<D>::new(block_size),
            meta,
            value,
            rank,
            n_out: out_shape.iter().product(),
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &PadNdForward<D> {
        &self.kernel
    }

    /// The axis slot's contents, zero-padded to a whole number of the node's elements:
    /// a slot is allocated in the node's dtype, whatever width that is.
    fn meta_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = self.meta.iter().flat_map(|v| v.to_le_bytes()).collect();
        let width = core::mem::size_of::<D>();
        let padded = bytes.len().div_ceil(width) * width;
        bytes.resize(padded, 0);
        bytes
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for PadNdRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        let width = core::mem::size_of::<D>();
        vec![vec![1], vec![self.meta_bytes().len().div_ceil(width)]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["value", "axes"]
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        match param_idx {
            0 => Some(self.value.clone()),
            1 => Some(self.meta_bytes()),
            _ => None,
        }
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n_out as i32);
        visitor.visit_i32(self.rank as i32);
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n_out.div_ceil(self.kernel.block_size as usize) as u32).max(1),
            1,
            1,
        ]
    }
}
