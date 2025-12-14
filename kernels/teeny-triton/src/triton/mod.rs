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

use std::ops::Mul;

use crate::triton::types::{self as ty};

pub mod dummy;
pub mod llvm;
pub mod types;

#[derive(Debug)]
pub struct TritonKernel {
    pub name: &'static str,
    pub sig: &'static str,
    pub block_str: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramAxis {
    Axis0,
    Axis1,
    Axis2,
}

pub trait Triton
where
    Self::BoolTensor: ty::BoolTensor<B = Self::Bool>,
    Self::I32: ty::I32<I64 = Self::I64>,
    Self::I32Tensor: ty::I32Tensor<I32 = Self::I32, I64 = Self::I64, I64Tensor = Self::I64Tensor>,
    Self::I64Tensor: ty::I64Tensor<I32 = Self::I32, I64 = Self::I64>
        + ty::TensorComparison<Self::I32, BoolTensor = Self::BoolTensor>,
{
    type Bool: ty::Bool;
    type I32: ty::I32;
    type I64: ty::I64;
    type BF16: ty::BF16;

    type BoolTensor: ty::BoolTensor;
    type I32Tensor: ty::I32Tensor;
    type I64Tensor: ty::I64Tensor;
    type Tensor<D: ty::Dtype>: ty::Tensor<D>;
    type Pointer<D: ty::Dtype>: ty::Pointer<D>;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;

    fn arange<T: Into<Self::I32>>(start: T, end: T) -> Self::I32Tensor;

    fn load<D: ty::Dtype>(
        ptr: &Self::Pointer<D>,
        mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D>;

    fn store<D: ty::Dtype>(
        dest: &Self::Pointer<D>,
        src: &Self::Tensor<D>,
        mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D>;
}
