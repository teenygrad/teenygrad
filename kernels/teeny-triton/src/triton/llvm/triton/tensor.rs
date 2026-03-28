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

use super::super::super::types::{self as ty};

/*--------------------------------- Tensor ---------------------------------*/

pub struct Tensor<D: ty::Dtype>(pub *mut D);
impl<D: ty::Dtype> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: ty::Dtype> Copy for Tensor<D> {}

impl<D: ty::Dtype> ty::Tensor<D> for Tensor<D> {}
impl<D: ty::Dtype> ty::RankedTensor<D> for Tensor<D> {}

// Element-wise addition for tensors
impl<D: ty::Dtype> Add<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: Tensor<D>) -> Self::Output {
        // dummy implementation not used in final output
        Tensor(0 as *mut D)
    }
}

pub type BoolTensor = Tensor<bool>;
impl ty::BoolTensor for BoolTensor {}

pub type I32Tensor = Tensor<i32>;

impl ty::I32Tensor for I32Tensor {}

impl ty::Comparison<i32> for I32Tensor {
    type BoolTensor = BoolTensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn lt(self, _other: i32) -> Self::BoolTensor {
        // dummy implementation not used in final output
        Tensor(0 as *mut bool)
    }
}

// Blanket implementation for any type implementing I64, including <I32 as Mul<u32>>::Output
impl Add<i32> for I32Tensor {
    type Output = I32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: i32) -> Self::Output {
        // dummy implementation not used in final output
        Tensor(0 as *mut i32)
    }
}
