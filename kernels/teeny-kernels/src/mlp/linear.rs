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

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::{Axis, Triton};

#[kernel]
pub fn linear<T: Triton, D: Float, const USE_BIAS: bool, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<D>,
    weight_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    let x = T::load(
        input_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let w = T::load(
        weight_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let b = T::load(
        bias_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let output = x * w;
    let output = if USE_BIAS { output + b } else { output };

    T::store(
        output_ptr.add_offsets(offsets),
        output,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}
