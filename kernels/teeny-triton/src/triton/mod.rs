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

use crate::triton::types as ty;

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

pub trait Triton {
    type AnyType: ty::AnyType;
    type IntLike: ty::IntLike;
    type I64Like: ty::I64Like;
    type BoolLike: ty::BoolLike;
    type PointerLike: ty::PointerLike;

    type Pointer<D: ty::Dtype>: ty::Pointer<
            D,
            Self::PointerLike,
            Self::IntLike,
            Self::BoolLike,
            Self::AnyType,
            Self::I64,
            Self::I32,
            Self::I1,
            Self::BoolTensor,
        >;

    type I1: ty::I1<Self::BoolLike, Self::AnyType>;
    type I32: ty::I32<Self::IntLike, Self::AnyType, Self::I64>;
    type I64: ty::I64<Self::IntLike, Self::AnyType>;

    type Tensor<D: ty::Dtype>;
    type BoolTensor: ty::BoolTensor<Self::BoolLike, Self::AnyType, Self::I1>;
    type IntTensor: ty::IntTensor<
            Self::IntLike,
            Self::BoolLike,
            Self::AnyType,
            Self::I64,
            Self::I32,
            Self::I1,
            Self::BoolTensor,
        >;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;

    fn arange<'a, S1, S2>(start: S1, end: S2) -> Self::IntTensor<Self::I32>
    where
        S1: Into<Self::I32>,
        S2: Into<Self::I32>;

    fn load<'a, D: ty::Dtype>(
        ptr: &Self::Pointer<D>,
        mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D>;

    fn store<'a, D: ty::Dtype>(
        src: &Self::Pointer<D>,
        dest: &Self::Pointer<D>,
        mask: &Option<Self::BoolTensor>,
    );
}
