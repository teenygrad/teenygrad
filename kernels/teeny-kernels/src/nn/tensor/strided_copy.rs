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

//! Copy under an arbitrary affine index mapping.
//!
//! One kernel for the family of ops that move elements without computing on them.
//! Each output element is read from
//!
//! ```text
//! in_index = in_offset + sum_d ( coord_d * in_stride_d )
//! ```
//!
//! where `coord_d` is the output coordinate along dimension `d`. Choosing the strides
//! and offset expresses several ONNX operators with no further kernels:
//!
//! | op | expressed as |
//! |---|---|
//! | `Transpose` | input strides permuted to match the output axis order |
//! | `Slice` | `in_offset = sum(start_d * input_stride_d)`, `in_stride_d = step_d * input_stride_d` |
//! | `Expand` / broadcast | `in_stride_d = 0` on every broadcast dimension |
//! | contiguous copy | the input's own row-major strides |
//!
//! Padding is deliberately **not** covered: a pad reads outside the input and must
//! substitute a constant, which this mapping cannot express. That needs per-dimension
//! bounds and belongs in a separate kernel.
//!
//! Shapes are given as [`MAX_RANK`] dimensions, outermost first and row-major, so the
//! last is fastest-varying. A tensor of lower rank is passed left-padded with extent 1
//! and stride 0, which contributes nothing to the index.
//!
//! Grid: `cdiv(n_elements, BLOCK_SIZE)`, one CTA per block of output elements.

use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Dimensions the kernel indexes. Rank-5 is the most FoundationStereo needs before
/// its rank-4 rewrite, so six leaves a margin without widening the signature further.
pub const MAX_RANK: usize = 6;

/// Copy `y[coords] = x[in_offset + dot(coords, in_strides)]`.
///
/// `s0..s5` are the output extents and `t0..t5` the corresponding input strides, both
/// outermost first. Unused leading dimensions take extent 1 and stride 0.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn strided_copy_forward<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    in_offset: i32,
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
    s4: i32,
    s5: i32,
    t0: i32,
    t1: i32,
    t2: i32,
    t3: i32,
    t4: i32,
    t5: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    let izero = T::zeros::<i32>(&[BLOCK_SIZE]);
    let mut idx = izero + in_offset;

    // Peel the flat output index into coordinates, fastest-varying dimension first,
    // accumulating the input index as we go. There is no integer remainder in the
    // kernel DSL, so each coordinate is `rem - (rem / extent) * extent`. A dimension
    // of extent 1 yields coordinate 0 and contributes nothing, which is what makes the
    // left-padding convention work.
    let mut rem = p;

    let q5 = rem / s5;
    idx = idx + (rem - q5 * s5) * t5;
    rem = q5;

    let q4 = rem / s4;
    idx = idx + (rem - q4 * s4) * t4;
    rem = q4;

    let q3 = rem / s3;
    idx = idx + (rem - q3 * s3) * t3;
    rem = q3;

    let q2 = rem / s2;
    idx = idx + (rem - q2 * s2) * t2;
    rem = q2;

    let q1 = rem / s1;
    idx = idx + (rem - q1 * s1) * t1;
    rem = q1;

    let q0 = rem / s0;
    idx = idx + (rem - q0 * s0) * t0;

    // Lanes past n_elements are masked, so their index is never dereferenced.
    let v = T::load(
        x_ptr.add_offsets(idx),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(p), v, Some(in_bounds), &[], None, None);
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use teeny_core::device::program::ArgVisitor;
// The runtime op holds only i32 scalars and a marker, so `Num` is bound enough: nothing
// here converts a value of `D` to or from bytes. Keeping it at `FloatBytes` would have
// limited the copy to float tensors, and a model's shape arithmetic is integer.
use teeny_core::model::{RawPtr, RuntimeOp};

/// Row-major strides for `shape`, in elements.
pub fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as i64;
    }
    strides
}

/// Left-pads `values` to [`MAX_RANK`] with `fill`, keeping the trailing (fastest)
/// dimensions in place.
fn pad_left(values: &[i64], fill: i64) -> [i32; MAX_RANK] {
    let mut out = [fill as i32; MAX_RANK];
    let start = MAX_RANK.saturating_sub(values.len());
    for (i, v) in values.iter().take(MAX_RANK).enumerate() {
        out[start + i] = *v as i32;
    }
    out
}

/// Runtime dispatch for [`strided_copy_forward`].
pub struct StridedCopyRuntimeOp<D: Num + Send + Sync + 'static> {
    block_size: i32,
    in_offset: i32,
    out_shape: [i32; MAX_RANK],
    in_strides: [i32; MAX_RANK],
    _phantom: PhantomData<D>,
}

impl<D: Num + Send + Sync + 'static> StridedCopyRuntimeOp<D> {
    /// Builds the op from an explicit mapping: the output's extents, the input stride
    /// to advance per output coordinate, and a starting offset into the input.
    pub fn new(block_size: i32, out_shape: &[usize], in_strides: &[i64], in_offset: i64) -> Self {
        let shape: Vec<i64> = out_shape.iter().map(|&d| d as i64).collect();
        Self {
            block_size,
            in_offset: in_offset as i32,
            out_shape: pad_left(&shape, 1),
            in_strides: pad_left(in_strides, 0),
            _phantom: PhantomData,
        }
    }

    /// `Transpose`: output axis `d` is input axis `perm[d]`, so it advances by that
    /// axis's input stride.
    pub fn transpose(block_size: i32, in_shape: &[usize], perm: &[usize]) -> Self {
        let src = contiguous_strides(in_shape);
        let out_shape: Vec<usize> = perm.iter().map(|&a| in_shape[a]).collect();
        let strides: Vec<i64> = perm.iter().map(|&a| src[a]).collect();
        Self::new(block_size, &out_shape, &strides, 0)
    }

    /// `Slice`: start at `starts`, advance `steps` input elements per output element.
    pub fn slice(
        block_size: i32,
        in_shape: &[usize],
        out_shape: &[usize],
        starts: &[i64],
        steps: &[i64],
    ) -> Self {
        let src = contiguous_strides(in_shape);
        let strides: Vec<i64> = src.iter().zip(steps).map(|(s, k)| s * k).collect();
        let offset: i64 = src.iter().zip(starts).map(|(s, b)| s * b).sum();
        Self::new(block_size, out_shape, &strides, offset)
    }

    /// `Expand` / broadcast: a dimension of extent 1 that grows takes stride 0, so every
    /// output coordinate along it reads the same input element.
    pub fn expand(block_size: i32, in_shape: &[usize], out_shape: &[usize]) -> Self {
        let src = contiguous_strides(in_shape);
        // Align trailing dimensions, as broadcasting does.
        let lead = out_shape.len() - in_shape.len();
        let strides: Vec<i64> = (0..out_shape.len())
            .map(|d| {
                if d < lead {
                    0
                } else {
                    let i = d - lead;
                    if in_shape[i] == 1 && out_shape[d] != 1 {
                        0
                    } else {
                        src[i]
                    }
                }
            })
            .collect();
        Self::new(block_size, out_shape, &strides, 0)
    }
}

impl<D: Num + Send + Sync + 'static> StridedCopyRuntimeOp<D> {
    /// The kernel's scalar arguments after the two pointers, in order:
    /// `n_elements`, `in_offset`, then [`MAX_RANK`] output extents and [`MAX_RANK`]
    /// input strides. Exposed so callers and tests can see the mapping without
    /// going through an `ArgVisitor`.
    pub fn kernel_scalars(&self, n_elements: i32) -> Vec<i32> {
        let mut v = Vec::with_capacity(2 + 2 * MAX_RANK);
        v.push(n_elements);
        v.push(self.in_offset);
        v.extend_from_slice(&self.out_shape);
        v.extend_from_slice(&self.in_strides);
        v
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for StridedCopyRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        for v in self.kernel_scalars(n as i32) {
            visitor.visit_i32(v);
        }
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}
