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
        + AddOffsets<
            D,
            Self::I32,
            Self::I32Tensor,
            Pointer = Self::Pointer<D>,
            Output = Self::Tensor<Self::Pointer<D>>,
        >;

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
