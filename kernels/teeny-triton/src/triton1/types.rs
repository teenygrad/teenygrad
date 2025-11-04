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

// Any type
pub trait AnyType {}

// Tensor
pub trait RankedTensor<S: AnyType, T>: Dtype + Into<S> {}

// Floating-point Type
pub trait FloatLike {}
pub trait Float<S: FloatLike, T: AnyType>: Dtype + Copy + Into<S> + Into<T> {}
pub trait FloatTensor<S: FloatLike, T: AnyType, U: Float<S, T>>:
    RankedTensor<T, U> + Into<S> + Into<T>
{
}

pub trait F8E4M3FN<S: FloatLike, T: AnyType>: Float<S, T> {}
pub trait F8E4M3FNUZ<S: FloatLike, T: AnyType>: Float<S, T> {}
pub trait F8E5M2<S: FloatLike, T: AnyType>: Float<S, T> {}
pub trait F8E5M2FNUZ<S: FloatLike, T: AnyType>: Float<S, T> {}

pub trait F16<S: FloatLike, T: AnyType>: Float<S, T> {}
pub trait BF16<S: FloatLike, T: AnyType>: Float<S, T> {}
pub trait F32<S: FloatLike, T: AnyType>: Float<S, T> {}
pub trait F64<S: FloatLike, T: AnyType>: Float<S, T> {}

// Boolean Type
pub trait BoolLike {}
pub trait BoolTensor<S: BoolLike, T: AnyType, U: I1<S, T>>:
    RankedTensor<T, U> + Into<S> + Into<T>
{
}

// Integer Type
pub trait Int: Dtype + Copy {}

pub trait I1<S: BoolLike, T: AnyType>: Int + Into<S> + Into<T> {}
pub trait I4<S: IntLike, T: AnyType>: Int + Into<S> + Into<T> {}
pub trait I8<S: IntLike, T: AnyType>: Int + Into<S> + Into<T> {}
pub trait I16<S: IntLike, T: AnyType>: Int + Into<S> + Into<T> {}
pub trait I32<S: IntLike, T: AnyType, O: I64<S, T>>:
    Int + Into<S> + Into<T> + Add<Self, Output = O> + Mul<Self, Output = O> + From<isize>
{
}
pub trait I64<S: IntLike, T: AnyType>: Dtype + Int + Into<S> + Into<T> {}

// Int Tensor
pub trait IntTensor<
    S: IntLike,
    B: BoolLike,
    T: AnyType,
    O: I64<S, T>,
    V: I32<S, T, O>,
    U: I1<B, T>,
    BT: BoolTensor<B, T, U>,
>:
    RankedTensor<T, V>
    + Into<S>
    + Into<T>
    + Add<O, Output = Self>
    + Mul<O, Output = Self>
    + Comparison<O, Output = BT>
    + Comparison<V, Output = BT>
{
}

pub trait IntLike {}

// I32 Type
pub trait I32Like {}
pub trait I32Tensor<S: IntLike, T: AnyType, U: I64Like, O: I64<S, T>, V: I32<S, T, O>>:
    RankedTensor<T, V> + Into<S> + Into<T> + Comparison<V>
{
}

// I64 Type
pub trait I64Like {}

// Pointer Type
pub trait Pointer<
    'a,
    D: Dtype,
    PL: PointerLike,
    S: IntLike,
    B: BoolLike,
    T: AnyType,
    O: I64<S, T>,
    V: I32<S, T, O>,
    U: I1<B, T>,
    BT: BoolTensor<B, T, U>,
>: Into<PL>
{
    fn add(&self, other: &Self) -> Self;

    fn add_offsets<IT: IntTensor<S, B, T, O, V, U, BT>>(&self, other: &IT) -> Self;
}

pub trait PointerLike {}
