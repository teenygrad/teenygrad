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

use core::ops::Add;

use self::types::{self as ty};

pub mod llvm;
pub mod types;

pub use types::*;

#[repr(i32)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

pub trait Triton
where
    Self::I32Tensor: Add<i32, Output = Self::I32Tensor>,
    Self::I32Tensor: Comparison<i32, BoolTensor = Self::BoolTensor>,
{
    type BF16: ty::BF16;
    type BoolTensor: ty::BoolTensor;
    type I32Tensor: ty::I32Tensor;
    type Tensor<D: ty::Dtype>: ty::Tensor<D> + Add<Self::Tensor<D>, Output = Self::Tensor<D>>;
    type Pointer<D: ty::Dtype>: ty::Pointer<D, I32Tensor = Self::I32Tensor>
        + AddOffsets<i32, Self::I32Tensor, Output = Self::Tensor<Self::Pointer<D>>>;

    fn program_id(axis: Axis) -> i32;

    fn num_programs(axis: Axis) -> i32;

    fn arange(start: impl Into<i32>, end: impl Into<i32>) -> Self::I32Tensor;

    fn zeroes_like<D: ty::Dtype>(offsets: Self::Tensor<D>) -> Self::Tensor<D>;

    fn maximum<D: ty::Dtype>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;

    fn load<D: ty::Dtype>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        mask: Self::BoolTensor,
    ) -> Self::Tensor<D>;

    fn store<D: ty::Dtype>(
        dest: Self::Tensor<Self::Pointer<D>>,
        src: Self::Tensor<D>,
        mask: Self::BoolTensor,
    );
}
