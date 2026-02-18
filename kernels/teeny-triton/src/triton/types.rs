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

// Dtype Type
pub trait Dtype: Copy + Clone {}

pub trait Num: Dtype {}

pub trait Float: Num {}
pub trait Int: Num {}
pub trait Bool: Dtype + Copy {}

// Tensor
pub trait RankedTensor<D: Dtype>: Copy + Clone {}

// Floating-point types
pub trait F8E4M3FN: Float {}
pub trait F8E4M3FNUZ: Float {}
pub trait F8E5M2: Float {}
pub trait F8E5M2FNUZ: Float {}

pub trait F16: Float {}
pub trait BF16: Float {}
pub trait F32: Float {}
pub trait F64: Float {}

// Supported integer types
pub trait I1: Int {}

pub trait I4: Int {}
pub trait I8: Int {}
pub trait I16: Int {}
pub trait I32: Int + From<u32> + From<i32> + Mul<u32> {}

pub trait I64: Int {}

// Int Tensor
pub trait Tensor<D: Dtype>: RankedTensor<D> {}

pub trait BoolTensor: Tensor<Self::Bool> {
    type Bool: Bool;
}

pub trait Comparison<I: Num> {
    type BoolTensor: BoolTensor;

    fn lt(&self, other: I) -> Self::BoolTensor;
}
pub trait I32Tensor: Tensor<Self::I32> + Add<Self::I32> + Comparison<Self::I32> {
    type I32: I32;
}

// Offsets trait for adding tensor offsets to pointers
pub trait AddOffsets<I: Int, T: Tensor<I>> {
    type Output;

    fn add_offsets(self, offsets: T) -> Self::Output;
}

// Pointer Type
pub trait Pointer<D: Dtype>:
    Sized + Copy + Clone + Dtype + AddOffsets<Self::I32, Self::I32Tensor> + Add<Self>
{
    type I32: I32;
    type I32Tensor: I32Tensor<I32 = Self::I32>;
}
