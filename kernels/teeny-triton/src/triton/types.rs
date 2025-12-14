/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::ops::{Add, Mul};

// Comparison
pub trait Comparison<T> {
    type Output;

    fn eq(&self, other: T) -> Self::Output;
    fn ne(&self, other: T) -> Self::Output;
    fn slt(&self, other: T) -> Self::Output;
    fn sle(&self, other: T) -> Self::Output;
    fn sgt(&self, other: T) -> Self::Output;
    fn sge(&self, other: T) -> Self::Output;
}

// Dtype Type
pub trait Dtype {}

pub trait Num: Dtype + Copy {}

pub trait Float: Num {}
pub trait Int: Num {}
pub trait Bool: Dtype + Copy {}

// Tensor
pub trait RankedTensor<D: Dtype> {}

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
pub trait I32: Int + From<u32> + From<i32> + Mul<u32>
where
    <Self as Mul<u32>>::Output: I64,
{
}

pub trait I64: Int {}

// Int Tensor
pub trait Tensor<D: Dtype>: RankedTensor<D> {}

pub trait BoolTensor<D: Bool>: Tensor<D> {}

pub trait TensorComparison<I: Num> {
    type B: Bool;
    type T: BoolTensor<Self::B>;

    fn less_than(&self, other: I) -> Self::T;
}
pub trait I32Tensor: Tensor<Self::I32> + Add<Self::I64, Output = Self::I64Tensor>
where
    <Self::I32 as Mul<u32>>::Output: I64,
{
    type I32: I32;
    type I64: I64;

    type I64Tensor: I64Tensor;
}

pub trait I64Tensor: Tensor<Self::I64> {
    type I64: I64;
}

// Pointer Type
pub trait Pointer<D: Dtype> {}
