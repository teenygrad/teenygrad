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
use teeny_macros::{kernel, tiled_kernel};
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── LogSigmoid ────────────────────────────────────────────────────────────────

/// Forward: y = log(sigmoid(x)) = -log(1 + exp(-x))
#[tiled_kernel(backward = LogSigmoidBackward)]
pub fn log_sigmoid_forward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x: In<Tile<T, D>>,
    y: Out<Tile<T, D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    let neg1 = T::full(&[BLOCK_SIZE], D::from_f64(-1.0));
    // -log(1 + exp(-x)) = log(1/(1+exp(-x))) = log(sigmoid(x))
    // But we want to avoid negating the result: use (neg1 * log(1 + exp(neg1*x)))
    // Actually: y = neg1 * log(one + T::exp(neg1 * x))
    // But neg1 * log(...) would require negating a tensor result.
    // Use subtraction: y = T::zeros_like(x) - T::log(one + T::exp(neg1 * x))
    let zeros = T::zeros_like(x.tensor);
    let y1 = zeros - T::log(one + T::exp(neg1 * x.tensor));
    T::store(y.tensor, y1, x.mask, &[], None, None);
}

/// Backward: dx = dy * sigmoid(-x) = dy / (1 + exp(x))
#[kernel]
pub fn log_sigmoid_backward<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    dy_ptr: In<T::Pointer<D>>,
    x_ptr: In<T::Pointer<D>>,
    dx_ptr: Out<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let dy = T::load(
        dy_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full(&[BLOCK_SIZE], D::from_f64(1.0));
    // sigmoid(-x) = 1 / (1 + exp(x))
    let dx = dy / (one + T::exp(x));
    T::store(
        dx_ptr.add_offsets(offsets),
        dx,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

pub struct LogSigmoidOp<D: Float> {
    pub forward: LogSigmoidForward<D>,
    pub backward: LogSigmoidBackward<D>,
}
