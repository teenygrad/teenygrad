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

// Any type
pub trait AnyType {}

// Tensor
pub trait RankedTensor<S: AnyType, T>: Into<S> {}

// Floating-point Type
pub trait FloatLike {}
pub trait Float<S: FloatLike, T: AnyType>: Into<S> + Into<T> {}
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
pub trait I1<S: BoolLike, T: AnyType>: Into<S> + Into<T> {}
pub trait I4<S: I32Like, T: AnyType>: Into<S> + Into<T> {}
pub trait I8<S: I32Like, T: AnyType>: Into<S> + Into<T> {}
pub trait I16<S: I32Like, T: AnyType>: Into<S> + Into<T> {}
pub trait I32<S: I32Like, T: AnyType>: Into<S> + Into<T> {}
pub trait I64<S: I64Like, T: AnyType>: Into<S> + Into<T> {}

// I32 Type
pub trait I32Like {}
pub trait I32Tensor<S: I32Like, T: AnyType, U: I32<S, T>>:
    RankedTensor<T, U> + Into<S> + Into<T>
{
}

// I64 Type
pub trait I64Like {}
pub trait I64Tensor<S: I64Like, T: AnyType, U: I64<S, T>>:
    RankedTensor<T, U> + Into<S> + Into<T>
{
}

// Pointer Type
pub trait PointerLike {}
pub trait Pointer<S: PointerLike, T: AnyType>: Into<S> + Into<T> {}
