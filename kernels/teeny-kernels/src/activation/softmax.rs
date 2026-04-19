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
#[kernel]
pub fn softmax_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
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

    let x = T::load(
        x_ptr.add_offsets(offsets),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // Triton's builtin: numerically-stable softmax (max subtraction, exp, sum, div).
    let y = T::softmax(x, None, false, false);

    T::store(y_ptr.add_offsets(offsets), y, None, &[], None, None);
}

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
    dy_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
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
