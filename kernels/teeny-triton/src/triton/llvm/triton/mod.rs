/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
