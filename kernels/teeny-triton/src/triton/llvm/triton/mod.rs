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
    type I64Tensor = tensor::I64Tensor;
    type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
    type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

    #[inline(never)]
    fn program_id(_axis: ProgramAxis) -> Self::I32 {
        loop {}
    }

    #[inline(never)]
    fn num_programs(_axis: ProgramAxis) -> Self::I32 {
        loop {}
    }

    #[inline(never)]
    fn arange<T: Into<Self::I32>>(_start: T, _end: T) -> Self::I32Tensor {
        loop {}
    }

    #[inline(never)]
    fn load<D: ty::Dtype>(
        _ptr: &Self::Pointer<D>,
        _mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D> {
        loop {}
    }

    #[inline(never)]
    fn store<D: ty::Dtype>(
        _dest: &Self::Pointer<D>,
        _src: &Self::Pointer<D>,
        _mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D> {
        loop {}
    }
}
