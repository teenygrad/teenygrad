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

use crate::triton::Triton;
use crate::triton::llvm::triton::pointer::Pointer;
use crate::triton::llvm::triton::tensor::BoolTensor;
use crate::triton::types as ty;

pub mod num;
pub mod pointer;
pub mod tensor;
pub mod types;

use num::*;
use types::*;

pub enum IntLike {}
impl ty::IntLike for IntLike {}

pub struct LlvmTriton {}

impl Triton for LlvmTriton {
    type AnyType = AnyType;
    type IntLike = IntLike;
    type BoolLike = BoolLike;

    type Pointer<D: ty::Dtype> = Pointer<D>;

    type Bool = Bool;
    type I1 = I1;
    type I32 = I32;
    type I64 = I64;

    // type Tensor<D: ty::Dtype> = Tensor<D>;

    type BoolTensor = BoolTensor<Self::Bool, Self::AnyType, Self::BoolLike>;
    type IntTensor = IntTensor;

    fn program_id(_axis: crate::triton::ProgramAxis) -> Self::I32 {
        todo!()
    }

    fn num_programs(_axis: crate::triton::ProgramAxis) -> Self::I32 {
        todo!()
    }

    fn arange<S1, S2>(start: S1, end: S2) -> Self::IntTensor
    where
        S1: Into<Self::I32>,
        S2: Into<Self::I32>,
    {
        todo!()
    }

    fn load<'a, D: ty::Dtype>(
        _ptr: &Self::Pointer<D>,
        _mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D> {
        todo!()
    }

    fn store<'a, D: ty::Dtype>(
        _src: &Self::Pointer<D>,
        _dest: &Self::Pointer<D>,
        _mask: &Option<Self::BoolTensor>,
    ) {
        todo!()
    }
}
