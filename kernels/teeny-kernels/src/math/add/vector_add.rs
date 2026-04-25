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

#[kernel]
pub fn vector_add<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);

    // Calculate the starting offset for this block
    let block_start = pid * BLOCK_SIZE;

    // Create offsets for the elements this block will process
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;

    // Create a mask to handle cases where n_elements is not divisible by BLOCK_SIZE
    let mask = offsets.lt(n_elements);

    // Load data from global memory with masking
    let x = T::load(
        x_ptr.add_offsets(offsets),
        Some(mask),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let y = T::load(
        y_ptr.add_offsets(offsets),
        Some(mask),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // Perform element-wise addition
    let output = x + y;

    // Store result back to global memory
    T::store(
        output_ptr.add_offsets(offsets),
        output,
        Some(mask),
        &[],
        None,
        None,
    );
}

pub struct VectorAddOp<'a, T: Float> {
    pub forward: VectorAdd<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}
