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
use super::super::{Axis, types as ty};

pub mod num;
pub mod pointer;
pub mod tensor;
pub mod types;

pub struct LlvmTriton {}

impl Triton for LlvmTriton {
    type BF16 = num::BF16;

    type BoolTensor = tensor::BoolTensor;
    type I32Tensor = tensor::I32Tensor;
    type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
    type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

    #[inline(never)]
    fn program_id(_axis: Axis) -> i32 {
        // dummy implementation not used in final output
        0
    }

    #[inline(never)]
    fn num_programs(_axis: Axis) -> i32 {
        // dummy implementation not used in final output
        0
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn arange(_start: impl Into<i32>, _end: impl Into<i32>) -> Self::I32Tensor {
        // dummy implementation not used in final output
        tensor::Tensor(0 as *mut i32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn zeroes_like<D: ty::Dtype>(_offsets: Self::Tensor<D>) -> Self::Tensor<D> {
        // dummy implementation not used in final output
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn maximum<D: ty::Dtype>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::Tensor<D> {
        // dummy implementation not used in final output
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
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
