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

use core::ops::Add;

use crate::triton::llvm::triton::tensor::{LlvmI32Tensor, LlvmTensor};

use super::super::super::types::{self as ty};

pub struct LlvmPointer<D: ty::Dtype>(pub *mut D);
impl<D: ty::Dtype> Clone for LlvmPointer<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: ty::Dtype> Copy for LlvmPointer<D> {}

impl<D: ty::Dtype> ty::Dtype for LlvmPointer<D> {}

impl<D: ty::Dtype, const RANK: usize> ty::Pointer<D, RANK> for LlvmPointer<D> {
    type I32Tensor = LlvmI32Tensor;
}

// Implement AddOffsets for Pointer
impl<D: ty::Dtype, const RANK: usize> ty::AddOffsets<i32, RANK, LlvmI32Tensor> for LlvmPointer<D> {
    type Output = LlvmTensor<Self>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add_offsets(self, _offsets: LlvmI32Tensor) -> Self::Output {
        // dummy implementation not used in final output
        LlvmTensor(0 as *mut Self)
    }
}

impl<D: ty::Dtype> Add<LlvmPointer<D>> for LlvmPointer<D> {
    type Output = Self;

    #[inline(never)]
    fn add(self, _other: LlvmPointer<D>) -> Self::Output {
        // dummy implementation not used in final output
        self
    }
}
