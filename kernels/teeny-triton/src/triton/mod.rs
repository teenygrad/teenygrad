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
    Self::BoolTensor: ty::BoolTensor<B = Self::Bool>,
    Self::I32: ty::I32<I64 = Self::I64>,
    Self::I32Tensor: ty::I32Tensor<I32 = Self::I32, I64 = Self::I64, I64Tensor = Self::I64Tensor>,
    Self::I64Tensor: ty::I64Tensor<I32 = Self::I32, I64 = Self::I64>
        + ty::Comparison<Self::I32, BoolTensor = Self::BoolTensor>,
{
    type Bool: ty::Bool;
    type I32: ty::I32;
    type I64: ty::I64;
    type BF16: ty::BF16;

    type BoolTensor: ty::BoolTensor;
    type I32Tensor: ty::I32Tensor;
    type I64Tensor: ty::I64Tensor;
    type Tensor<D: ty::Dtype>: ty::Tensor<D>;
    type Pointer<D: ty::Dtype>: ty::Pointer<D, I32 = Self::I32, I64 = Self::I64, I64Tensor = Self::I64Tensor>;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;

    fn arange(start: impl Into<Self::I32>, end: impl Into<Self::I32>) -> Self::I32Tensor;

    fn load_with_mask<D: ty::Dtype>(
        ptr: Self::Pointer<D>,
        mask: Self::BoolTensor,
    ) -> Self::Pointer<D>;

    fn store_with_mask<D: ty::Dtype>(
        dest: Self::Pointer<D>,
        src: Self::Pointer<D>,
        mask: Self::BoolTensor,
    );
}
