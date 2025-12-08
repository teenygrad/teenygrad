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

pub trait ProgramOps {
    type I32: ty::I32;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;
}

pub trait ArangeOp {
    type I32: ty::I32;
    type I32Tensor: ty::I32Tensor<Self::I32>;

    fn arange(start: Self::I32, end: Self::I32) -> Self::I32Tensor;
}

pub trait LoadOp<D: ty::Dtype> {
    type Bool: ty::Bool;
    type BoolTensor: ty::BoolTensor<Self::Bool>;
    type Pointer: ty::Pointer<D>;

    fn load(ptr: &Self::Pointer, mask: &Option<Self::BoolTensor>) -> Self::Pointer;
}

pub trait StoreOp<D: ty::Dtype> {
    type Bool: ty::Bool;
    type BoolTensor: ty::BoolTensor<Self::Bool>;
    type Tensor: ty::Tensor<D>;
    type Pointer: ty::Pointer<D>;

    fn store(dest: &Self::Pointer, src: &Self::Tensor, mask: &Option<Self::BoolTensor>);
}

pub trait Triton: ProgramOps + ArangeOp + LoadOp + StoreOp<Self::BF16>
where
    <Self as ProgramOps>::I32: ty::I32,
    <Self as ArangeOp>::I32: ty::I32,
    <Self as LoadOp<Self::BF16>>::Pointer: ty::Pointer<Self::BF16>,
    <Self as StoreOp<Self::BF16>>::Pointer: ty::Pointer<Self::BF16>,
{
    type I32;
    type BF16;
}
