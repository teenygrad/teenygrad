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

// Tensor
pub trait RankedTensor<T> {}

// Floating-point Type
pub trait IntoFloat: IntoFloatLike {}
pub trait IntoFloatLike {}
pub trait FloatTensor<T: IntoFloat>: RankedTensor<T> + IntoFloatLike {}

pub trait F8E4M3FN: IntoFloat {}
pub trait F8E4M3FNUZ: IntoFloat {}
pub trait F8E5M2: IntoFloat {}
pub trait F8E5M2FNUZ: IntoFloat {}

pub trait F16: IntoFloat {}
pub trait BF16: IntoFloat {}
pub trait F32: IntoFloat {}
pub trait F64: IntoFloat {}

// Boolean Type
pub trait BoolTensor<T: I1>: RankedTensor<T> + IntoBoolLike {}
pub trait IntoBoolLike {}

// Integer Type
pub trait I1: Sized + IntoBoolLike {}
pub trait I4: Sized {}
pub trait I8: Sized {}
pub trait I16: Sized {}
pub trait I32: Sized + IntoI32Like {}
pub trait I64: Sized {}

// I32 Type
pub trait I32Tensor<T: I32>: RankedTensor<T> + IntoI32Like {}
pub trait IntoI32Like {}

// I64 Type
pub trait I64Tensor<T: I64>: RankedTensor<T> + IntoI64Like {}
pub trait IntoI64Like {}

// Pointer Type
pub trait Pointer<T>: IntoPointerLike {}
pub trait PointerTensor<S, T: Pointer<S>>: RankedTensor<T> + IntoPointerLike {}
pub trait IntoPointerLike {}
