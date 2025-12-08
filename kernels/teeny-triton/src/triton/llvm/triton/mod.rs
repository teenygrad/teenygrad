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

use crate::triton::llvm::triton::num::I32;
use crate::triton::llvm::triton::pointer::Pointer;
use crate::triton::llvm::triton::tensor::{BoolTensor, I32Tensor, Tensor};
use crate::triton::{ArangeOp, LoadOp, ProgramAxis, ProgramOps, types as ty};
use crate::triton::{StoreOp, Triton};

pub mod num;
pub mod pointer;
pub mod tensor;
pub mod types;

use types::*;

pub struct LlvmTriton {}

impl Triton for LlvmTriton {
    type I32 = I32;
    type BF16 = BF16;
}

impl ProgramOps for LlvmTriton {
    type I32 = I32;

    fn program_id(_axis: ProgramAxis) -> Self::I32 {
        todo!()
    }

    fn num_programs(_axis: ProgramAxis) -> Self::I32 {
        todo!()
    }
}

impl ArangeOp for LlvmTriton {
    type I32 = I32;
    type I32Tensor = I32Tensor;

    fn arange(_start: Self::I32, _end: Self::I32) -> Self::I32Tensor {
        todo!()
    }
}

impl<D: ty::Dtype> LoadOp<D> for LlvmTriton {
    type Bool = Bool;
    type BoolTensor = BoolTensor;
    type Pointer = Pointer<D>;

    fn load(_ptr: &Self::Pointer, _mask: &Option<Self::BoolTensor>) -> Self::Pointer {
        todo!()
    }
}

impl<D: ty::Dtype> StoreOp<D> for LlvmTriton {
    type Bool = Bool;
    type BoolTensor = BoolTensor;
    type Tensor = Tensor<D>;
    type Pointer = Pointer<D>;

    fn store(_dest: &Self::Pointer, _src: &Self::Tensor, _mask: &Option<Self::BoolTensor>) {
        todo!()
    }
}
