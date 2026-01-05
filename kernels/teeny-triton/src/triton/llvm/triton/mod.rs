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

use super::super::Triton;
use super::super::{ProgramAxis, types as ty};

pub mod num;
pub mod pointer;
pub mod tensor;
pub mod types;

pub struct LlvmTriton {}

impl Triton for LlvmTriton {
    type I32 = num::I32;
    type I64 = num::I64;
    type BF16 = num::BF16;

    type Bool = types::Bool;
    type BoolTensor = tensor::BoolTensor;
    type I32Tensor = tensor::I32Tensor;
    type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
    type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

    #[inline(never)]
    fn program_id(_axis: ProgramAxis) -> Self::I32 {
        // dummy implementation not used in final output
        0.into()
    }

    #[inline(never)]
    fn num_programs(_axis: ProgramAxis) -> Self::I32 {
        // dummy implementation not used in final output
        0.into()
    }

    #[inline(never)]
    fn arange(_start: impl Into<Self::I32>, _end: impl Into<Self::I32>) -> Self::I32Tensor {
        loop {}
    }

    #[inline(never)]
    fn load<D: ty::Dtype>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _mask: Self::BoolTensor,
    ) -> Self::Tensor<D> {
        // dummy implementation not used in final output
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    fn store<D: ty::Dtype>(
        _dest: Self::Tensor<Self::Pointer<D>>,
        _src: Self::Tensor<D>,
        _mask: Self::BoolTensor,
    ) {
        // nop
    }
}
