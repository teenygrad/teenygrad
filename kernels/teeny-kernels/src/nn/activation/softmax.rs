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

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Row-wise softmax forward pass.
///
/// Grid: one CTA per row — `pid = row index`.
///
/// Each CTA loads the entire row of `BLOCK_SIZE` elements, applies Triton's
/// numerically-stable `softmax` builtin (`max`-subtraction + exp + normalise),
/// and stores the result.
///
/// **Constraint**: `BLOCK_SIZE` must equal `n_cols` for this kernel; the caller
/// is responsible for rounding `n_cols` up to the next power of two and passing
/// that as `BLOCK_SIZE`.  No masking is needed when `BLOCK_SIZE == n_cols`.
// ANCHOR: softmax_forward
#[kernel]
pub fn softmax_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    _n_rows: i32,
    n_cols: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let row_offset = pid * n_cols;
    let col_offsets = T::arange(0, BLOCK_SIZE);
    let offsets = col_offsets + row_offset;

    // `BLOCK_SIZE` is `n_cols` rounded up to a power of two, so the lanes past the row
    // belong to the next one, or to nothing at all on the last row. Unmasked, they were
    // read into the softmax, which put the next row's values in this row's denominator,
    // and written back, which overwrote that row's first elements. On the last row both
    // ran past the end of the buffer.
    //
    // The sentinel is a large negative number rather than negative infinity: the
    // generated kernel core has no `f64::NEG_INFINITY` to name, and a finite one cannot
    // give `-inf - -inf` either. `exp` of it underflows to zero in both f32 and f64,
    // which is all the sum needs.
    let in_row = col_offsets.lt(n_cols);
    let outside = T::full(&[BLOCK_SIZE], D::from_f64(-1e30));

    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_row),
        Some(outside),
        &[],
        None,
        None,
        None,
        false,
    );

    // Triton's builtin: numerically-stable softmax (max subtraction, exp, sum, div).
    let y = T::softmax(x, None, false, false);

    T::store(y_ptr.add_offsets(offsets), y, Some(in_row), &[], None, None);
}
// ANCHOR_END: softmax_forward

/// Row-wise softmax backward pass.
///
/// Given the saved softmax output `y = softmax(x)` and the upstream gradient
/// `dy`, computes the input gradient:
///
/// ```text
/// dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
/// ```
///
/// Grid: one CTA per row — `pid = row index`.
///
/// The dot product `sum(y * dy)` is a row-scalar that is broadcast back to the
/// full row when computing `dy - dot`.
///
/// **Constraint**: `BLOCK_SIZE` must equal `n_cols` (same as the forward pass).
#[kernel]
pub fn softmax_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    y_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    _n_rows: i32,
    n_cols: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let row_offset = pid * n_cols;
    let col_offsets = T::arange(0, BLOCK_SIZE);
    let offsets = col_offsets + row_offset;

    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let y = T::load(
        y_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // dot = sum_j(y_j * dy_j)  — a per-row scalar (0-D tensor after reduction).
    let dot = T::sum(y * dy, Some(0), false);

    // dx_i = y_i * (dy_i - dot)  — broadcast dot across the row.
    let dx = y * (dy - dot);

    T::store(dx_ptr.add_offsets(offsets), dx, None, &[], None, None);
}

/// A tensor's `[rows, cols]` view for a softmax over its innermost axis: the last
/// extent is the row width and the product of the rest is the row count.
///
/// A rank-1 tensor is one row, and a rank-0 one is a single element, which is its own
/// softmax.
pub(crate) fn rows_and_cols(shape: &[usize]) -> (usize, usize) {
    match shape.split_last() {
        Some((&cols, outer)) => (outer.iter().product::<usize>().max(1), cols),
        None => (1, 1),
    }
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp for SoftmaxForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

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
        // kernel args: x_ptr, y_ptr, n_rows, n_cols
        //
        // One CTA per row over the innermost axis, so the row count is the product of
        // every other axis. Reading them as `output_shape[0]` and `[1]` only agreed
        // with that at rank 2: at rank 4 it passed the second axis as the row width and
        // launched one row of however many there were, leaving the rest of the output
        // at whatever the buffer held. It also panicked on a rank-1 shape.
        let (n_rows, n_cols) = rows_and_cols(output_shape);
        let n_rows = n_rows as i32;
        let n_cols = n_cols as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_rows);
        visitor.visit_i32(n_cols);
    }

    // One CTA per row.
    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [rows_and_cols(output_shape).0 as u32, 1, 1]
    }
}

pub struct SoftmaxOp<'a, T: Float> {
    pub forward: SoftmaxForward<T>,
    pub backward: SoftmaxBackward<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
