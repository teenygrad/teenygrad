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

use std::ops::Add;

use crate::triton::llvm::triton::tensor::{I32Tensor, Tensor};

use super::super::super::types::{self as ty};
use super::num::I32;

pub struct Pointer<D: ty::Dtype>(pub *mut D);
impl<D: ty::Dtype> Clone for Pointer<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: ty::Dtype> Copy for Pointer<D> {}

impl<D: ty::Dtype> ty::Dtype for Pointer<D> {}

impl<D: ty::Dtype> ty::Pointer<D> for Pointer<D> {
    type I32 = I32;
    type I32Tensor = I32Tensor;
}

// Implement AddOffsets for I64Tensor
impl<D: ty::Dtype> ty::AddOffsets<I32, I32Tensor> for Pointer<D> {
    type Output = Tensor<Self>;

    #[inline(never)]
    fn add_offsets(self, _offsets: I32Tensor) -> Self::Output {
        // dummy implementation not used in final output
        Tensor(self.0 as *mut Self)
    }
}

impl<D: ty::Dtype> Add<Pointer<D>> for Pointer<D> {
    type Output = Self;

    #[inline(never)]
    fn add(self, _other: Pointer<D>) -> Self::Output {
        // dummy implementation not used in final output
        self
    }
}
