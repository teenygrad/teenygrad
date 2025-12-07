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

// Helper trait to express that Add output depends on both Self and the other operand
pub trait AddWith<O> {
    type Output;
}

// Helper trait to express that Mul output depends on both Self and the other operand
pub trait MulWith<O> {
    type Output;
}

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

// Any type
pub trait AnyType {}

// Tensor
pub trait RankedTensor<D: Dtype>: Into<Self::AnyType> {
    type AnyType: AnyType;
}

// Floating-point Type
pub trait FloatLike {}
pub trait Float: Dtype + Copy + Into<Self::FloatLike> + Into<Self::AnyType> {
    type AnyType: AnyType;
    type FloatLike: FloatLike;
}
pub trait FloatTensor<T: Float<FloatLike = Self::FloatLike>>:
    Into<Self::FloatLike> + RankedTensor<T>
where
    <Self as RankedTensor<T>>::AnyType: AnyType,
{
    type AnyType: AnyType;
    type FloatLike: FloatLike;
}

pub trait F8E4M3FN: Float {}
pub trait F8E4M3FNUZ: Float {}
pub trait F8E5M2: Float {}
pub trait F8E5M2FNUZ: Float {}

pub trait F16: Float {}
pub trait BF16: Float {}
pub trait F32: Float {}
pub trait F64: Float {}

// Boolean Type
pub trait BoolLike {}

pub trait Bool: Dtype + Copy + Into<Self::BoolLike> + Into<Self::AnyType> {
    type AnyType: AnyType;
    type BoolLike: BoolLike;
}

pub trait BoolTensor: RankedTensor<Self::Bool> + Into<Self::BoolLike>
where
    <Self as RankedTensor<Self::Bool>>::AnyType: AnyType,
{
    type Bool: Bool;
    type AnyType: AnyType;
    type BoolLike: BoolLike;
}

// Integer Type
pub trait IntLike {}

pub trait Int: Dtype + Copy + Into<Self::IntLike> + Into<Self::AnyType> {
    type AnyType: AnyType;
    type IntLike: IntLike;
}

pub trait I1: Int + Into<Self::BoolLike> {
    type BoolLike: BoolLike;
}

pub trait I4: Int {}
pub trait I8: Int {}
pub trait I16: Int {}
pub trait I32:
    Int
    + AddWith<Self, Output = Self::I64>
    + MulWith<Self, Output = Self::I64>
    + Add<Self, Output = <Self as AddWith<Self>>::Output>
    + Mul<Self, Output = <Self as MulWith<Self>>::Output>
    + From<isize>
{
    type I64: I64;
}

pub trait I64: Int {}

// Int Tensor
pub trait IntTensor:
    RankedTensor<Self::Int>
    + Into<Self::IntLike>
    + AddWith<Self::I64>
    + MulWith<Self::I64>
    + Add<Self::I64, Output = <Self as AddWith<Self::I64>>::Output>
    + Mul<Self::I64, Output = <Self as MulWith<Self::I64>>::Output>
    + Comparison<Self::I32, Output = Self::BoolTensor>
    + Comparison<Self::I64, Output = Self::BoolTensor>
{
    type Int: Int;
    type IntLike: IntLike;

    type I32: I32;
    type I64: I64;
    type BoolTensor: BoolTensor;
}

// Pointer Type
pub trait Pointer<D: Dtype> {
    fn add(&self, other: &Self) -> Self;

    fn add_offsets<T: IntTensor>(&self, other: &T) -> Self;
}
