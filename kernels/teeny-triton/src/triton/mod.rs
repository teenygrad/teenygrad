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

use std::ops::{Add, Mul};

use self::types::{self as ty};

pub mod llvm;
pub mod types;

pub use types::*;

#[repr(i32)]
pub enum ProgramAxis {
    Axis0 = 0,
    Axis1 = 1,
    Axis2 = 2,
}

pub trait Triton
where
    Self::I32: Mul<u32, Output = Self::I32>,
    Self::I32Tensor: Add<Self::I32, Output = Self::I32Tensor>,
    Self::I32Tensor: Comparison<Self::I32, BoolTensor = Self::BoolTensor>,
{
    type Bool: ty::Bool;
    type I32: ty::I32;
    type I64: ty::I64;
    type BF16: ty::BF16;

    type BoolTensor: ty::BoolTensor<Bool = Self::Bool>;
    type I32Tensor: ty::I32Tensor<I32 = Self::I32>;
    type Tensor<D: ty::Dtype>: ty::Tensor<D> + Add<Self::Tensor<D>, Output = Self::Tensor<D>>;
    type Pointer<D: ty::Dtype>: ty::Pointer<D, I32 = Self::I32, I32Tensor = Self::I32Tensor>
        + AddOffsets<Self::I32, Self::I32Tensor, Output = Self::Tensor<Self::Pointer<D>>>;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;

    fn arange(start: impl Into<Self::I32>, end: impl Into<Self::I32>) -> Self::I32Tensor;

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
