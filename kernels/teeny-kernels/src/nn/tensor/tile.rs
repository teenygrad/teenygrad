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

//! Repeat a tensor along each axis (ONNX `Tile`).
//!
//! `Tile` is the one member of the move-without-computing family that
//! [`crate::nn::tensor::strided_copy`] cannot express. That kernel reads each output
//! element at `in_offset + dot(coords, in_strides)`, an affine function of the output
//! coordinate. Tiling is not affine: the input coordinate wraps,
//!
//! ```text
//! in_coord_d = out_coord_d mod in_extent_d
//! ```
//!
//! and no choice of stride and offset produces a wrap. Broadcasting looks similar and
//! *is* affine, because it only ever repeats an axis of extent 1, where stride 0 reads
//! the same element every time. Tiling repeats an axis of any extent, so it has to
//! divide.
//!
//! Conventions follow `strided_copy`, so the two read the same way: shapes are
//! [`MAX_RANK`] dimensions, outermost first and row-major, with a lower-rank tensor
//! left-padded by extent 1 and stride 0. A padded dimension yields coordinate 0 and
//! contributes nothing.
//!
//! Grid: `cdiv(n_elements, BLOCK_SIZE)`, one CTA per block of output elements.

use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

pub use crate::nn::tensor::strided_copy::MAX_RANK;

/// Copy `y[coords] = x[dot(coords mod in_extents, in_strides)]`.
///
/// `s0..s5` are the output extents and `d0..d5` the input extents, both outermost
/// first; unused leading dimensions take extent 1. The input strides are not passed:
/// the input is the tensor itself rather than a view, so it is contiguous and its
/// row-major strides follow from its extents. Deriving them here also keeps the
/// signature inside the 20-argument kernel ABI limit.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn tile_forward<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
    s4: i32,
    s5: i32,
    d0: i32,
    d1: i32,
    d2: i32,
    d3: i32,
    d4: i32,
    d5: i32,
    t0: i32,
    t1: i32,
    t2: i32,
    t3: i32,
    t4: i32,
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

    let mut idx = T::zeros::<i32>(&[BLOCK_SIZE]);

    // Peel the flat output index into coordinates, fastest-varying dimension first.
    // Each step needs two remainders and the DSL has no integer remainder operator, so
    // both are `a - (a / b) * b`: one to split the output coordinate out of the flat
    // index, and one to wrap it into the input extent. The input extents are clamped to
    // at least 1 by the runtime op, so neither division can be by zero even on the
    // lanes past `n_elements`, which compute an index that is then never dereferenced.
    let mut rem = p;

    let q5 = rem / s5;
    let c5 = rem - q5 * s5;
    // The innermost stride of a contiguous input is 1, so it is a literal rather than
    // a parameter, which keeps the signature inside the 20-argument kernel ABI limit.
    idx = idx + (c5 - (c5 / d5) * d5);
    rem = q5;

    let q4 = rem / s4;
    let c4 = rem - q4 * s4;
    idx = idx + (c4 - (c4 / d4) * d4) * t4;
    rem = q4;

    let q3 = rem / s3;
    let c3 = rem - q3 * s3;
    idx = idx + (c3 - (c3 / d3) * d3) * t3;
    rem = q3;

    let q2 = rem / s2;
    let c2 = rem - q2 * s2;
    idx = idx + (c2 - (c2 / d2) * d2) * t2;
    rem = q2;

    let q1 = rem / s1;
    let c1 = rem - q1 * s1;
    idx = idx + (c1 - (c1 / d1) * d1) * t1;
    rem = q1;

    let q0 = rem / s0;
    let c0 = rem - q0 * s0;
    idx = idx + (c0 - (c0 / d0) * d0) * t0;

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
use teeny_core::dtype::FloatBytes;
use teeny_core::model::{RawPtr, RuntimeOp};

use crate::nn::tensor::strided_copy::contiguous_strides;

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

/// Runtime dispatch for [`tile_forward`].
pub struct TileRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    block_size: i32,
    out_shape: [i32; MAX_RANK],
    in_shape: [i32; MAX_RANK],
    /// Input strides for the outer [`MAX_RANK`] - 1 dimensions; the innermost is 1.
    in_strides: [i32; MAX_RANK],
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> TileRuntimeOp<D> {
    /// Builds the op for `Tile(input, repeats)`.
    ///
    /// Returns `None` unless `repeats` has one non-negative entry per input dimension
    /// and the rank fits [`MAX_RANK`], which is what ONNX requires of `Tile`. A caller
    /// that has the repeats as a materialised constant can check this before lowering
    /// rather than discovering it on the device.
    pub fn new(block_size: i32, in_shape: &[usize], repeats: &[i64]) -> Option<Self> {
        if repeats.len() != in_shape.len() || in_shape.len() > MAX_RANK {
            return None;
        }
        if repeats.iter().any(|&r| r < 0) {
            return None;
        }
        let out_shape: Vec<i64> = in_shape
            .iter()
            .zip(repeats)
            .map(|(&d, &r)| d as i64 * r)
            .collect();
        // An extent of 0 makes the output empty, so nothing is ever stored, but the
        // kernel still divides by it on every lane. Clamp to 1 to keep that division
        // defined; with no elements to write the value cannot be observed.
        let extents: Vec<i64> = in_shape.iter().map(|&d| (d as i64).max(1)).collect();
        Some(Self {
            block_size,
            out_shape: pad_left(&out_shape, 1),
            in_shape: pad_left(&extents, 1),
            in_strides: pad_left(&contiguous_strides(in_shape), 0),
            _phantom: PhantomData,
        })
    }

    /// The output shape this op produces, outermost first, with the left padding
    /// stripped back to `rank` dimensions.
    pub fn output_shape(&self, rank: usize) -> Vec<usize> {
        self.out_shape[MAX_RANK - rank..]
            .iter()
            .map(|&d| d as usize)
            .collect()
    }

    /// The kernel's scalar arguments after the two pointers, in order: `n_elements`,
    /// then [`MAX_RANK`] output extents, [`MAX_RANK`] input extents and the outer
    /// [`MAX_RANK`] - 1 input strides. Exposed so
    /// callers and tests can see the mapping without going through an [`ArgVisitor`].
    pub fn kernel_scalars(&self, n_elements: i32) -> Vec<i32> {
        let mut v = Vec::with_capacity(3 * MAX_RANK);
        v.push(n_elements);
        v.extend_from_slice(&self.out_shape);
        v.extend_from_slice(&self.in_shape);
        // The innermost stride is the literal 1 in the kernel, so it is not passed.
        v.extend_from_slice(&self.in_strides[..MAX_RANK - 1]);
        v
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for TileRuntimeOp<D> {
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
