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
use crate::triton::llvm::triton::num::I32;
use crate::triton::llvm::triton::pointer::Pointer;
use crate::triton::llvm::triton::tensor::{BoolTensor, I32Tensor, Tensor};
use crate::triton::{ProgramAxis, types as ty};

pub mod num;
pub mod pointer;
pub mod tensor;
pub mod types;

use types::*;

pub struct LlvmTriton {}

impl Triton for LlvmTriton {
    type I32 = I32;
    type BF16 = BF16;

    type Bool = Bool;
    type BoolTensor = BoolTensor;
    type I32Tensor = I32Tensor;
    type Tensor<D: ty::Dtype> = Tensor<D>;
    type Pointer<D: ty::Dtype> = Pointer<D>;

    fn program_id(_axis: ProgramAxis) -> Self::I32 {
        todo!()
    }

    fn num_programs(_axis: ProgramAxis) -> Self::I32 {
        todo!()
    }

    fn arange<T: Into<Self::I32>>(_start: T, _end: T) -> Self::I32Tensor {
        todo!()
    }

    fn load<D: ty::Dtype>(
        _ptr: &Self::Pointer<D>,
        _mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D> {
        todo!()
    }

    fn store<D: ty::Dtype>(
        _dest: &Self::Pointer<D>,
        _src: &Self::Tensor<D>,
        _mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D> {
        todo!()
    }
}
