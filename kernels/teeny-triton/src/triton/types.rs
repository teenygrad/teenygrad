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

// // Boolean Type
// pub trait BoolLike {}
// pub trait BoolTensor<S: BoolLike, T: AnyType, U: I1<S, T>>:
//     RankedTensor<T, U> + Into<S> + Into<T>
// {
// }

// // Integer Type
// pub trait Int: Dtype + Copy {}

// pub trait I1<S: BoolLike, T: AnyType>: Int + Into<S> + Into<T> {}
// pub trait I4<S: IntLike, T: AnyType>: Int + Into<S> + Into<T> {}
// pub trait I8<S: IntLike, T: AnyType>: Int + Into<S> + Into<T> {}
// pub trait I16<S: IntLike, T: AnyType>: Int + Into<S> + Into<T> {}
// pub trait I32:
//     Int
//     + Into<Self::IntLike>
//     + Into<Self::AnyType>
//     + AddWith<Self::I64>
//     + MulWith<Self::I64>
//     + Add<Self::I64, Output = <Self as AddWith<Self::I64>>::Output>
//     + Mul<Self::I64, Output = <Self as MulWith<Self::I64>>::Output>
//     + From<isize>
// {
//     type IntLike: IntLike;
//     type AnyType: AnyType;
//     type I64: I64<IntLike = Self::IntLike, AnyType = Self::AnyType>;
// }
// pub trait I64: Dtype + Int + Into<Self::IntLike> + Into<Self::AnyType> {
//     type IntLike: IntLike;
//     type AnyType: AnyType;
// }

// // Int Tensor
// pub trait IntTensor<D: Dtype>:
//     RankedTensor<D, Self::AnyType>
//     + Into<Self::AnyType>
//     + Into<Self::IntLike>
//     + AddWith<Self::I64>
//     + MulWith<Self::I64>
//     + Add<Self::I64, Output = <Self as AddWith<Self::I64>>::Output>
//     + Mul<Self::I64, Output = <Self as MulWith<Self::I64>>::Output>
//     + Comparison<Self::I64, Output = Self::BoolTensor>
//     + Comparison<Self::I32, Output = Self::BoolTensor>
// {
//     type AnyType: AnyType;
//     type IntLike: IntLike;
//     type BoolLike: BoolLike;

//     type I64: I64<Self::IntLike, Self::AnyType>;
//     type I32: I32<Self::IntLike, Self::AnyType, Self::I64>;
//     type I1: I1<Self::BoolLike, Self::AnyType>;
//     type BoolTensor: BoolTensor<Self::BoolLike, Self::AnyType, Self::I1>;
// }

// pub trait IntLike {}

// // I32 Type
// pub trait I32Like {}
// pub trait I32Tensor<S: IntLike, T: AnyType, U: I64Like, O: I64<S, T>, V: I32<S, T, O>>:
//     RankedTensor<T, V> + Into<S> + Into<T> + Comparison<V>
// {
// }

// // I64 Type
// pub trait I64Like {}

// // Pointer Type
// pub trait Pointer<
//     D: Dtype,
//     PL: PointerLike,
//     S: IntLike,
//     B: BoolLike,
//     T: AnyType,
//     O: I64<S, T>,
//     V: I32<S, T, O>,
//     U: I1<B, T>,
//     BT: BoolTensor<B, T, U>,
// >: Into<PL>
// {
//     fn add(&self, other: &Self) -> Self;

//     fn add_offsets<IT: IntTensor<S, B, T, O, V, U, BT>>(&self, other: &IT) -> Self;
// }

// pub trait PointerLike {}
