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

use core::ops::{Add, Div, Mul, Neg, Sub};

use super::super::super::types::{self as ty};

/*--------------------------------- Tensor ---------------------------------*/

pub struct LlvmTensor<D: ty::Dtype>(pub *mut D);
impl<D: ty::Dtype> Clone for LlvmTensor<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: ty::Dtype> Copy for LlvmTensor<D> {}

impl<D: ty::Dtype, const RANK: usize> ty::RankedTensor<D, RANK> for LlvmTensor<D> {
    const SHAPE: [usize; RANK] = [0; RANK];
}
impl<D: ty::Dtype, const RANK: usize> ty::Tensor<D, RANK> for LlvmTensor<D> {}

impl<D: ty::Dtype> Add<LlvmTensor<D>> for LlvmTensor<D> {
    type Output = LlvmTensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: LlvmTensor<D>) -> Self::Output {
        LlvmTensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Sub<LlvmTensor<D>> for LlvmTensor<D> {
    type Output = LlvmTensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sub(self, _rhs: LlvmTensor<D>) -> Self::Output {
        LlvmTensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Mul<LlvmTensor<D>> for LlvmTensor<D> {
    type Output = LlvmTensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn mul(self, _rhs: LlvmTensor<D>) -> Self::Output {
        LlvmTensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Div<LlvmTensor<D>> for LlvmTensor<D> {
    type Output = LlvmTensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn div(self, _rhs: LlvmTensor<D>) -> Self::Output {
        LlvmTensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Neg for LlvmTensor<D> {
    type Output = LlvmTensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn neg(self) -> Self::Output {
        LlvmTensor(0 as *mut D)
    }
}

pub type LlvmBoolTensor = LlvmTensor<bool>;
impl<const RANK: usize> ty::BoolTensor<RANK> for LlvmBoolTensor {}

pub type LlvmI32Tensor = LlvmTensor<i32>;

impl<const RANK: usize> ty::I32Tensor<RANK> for LlvmI32Tensor {}

impl<D: ty::Num, const RANK: usize> ty::Comparison<D, RANK> for LlvmTensor<D> {
    type BoolTensor = LlvmBoolTensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn lt(self, _other: D) -> Self::BoolTensor {
        LlvmTensor(0 as *mut bool)
    }
}

impl Add<i32> for LlvmI32Tensor {
    type Output = LlvmI32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: i32) -> Self::Output {
        LlvmTensor(0 as *mut i32)
    }
}

impl Sub<i32> for LlvmI32Tensor {
    type Output = LlvmI32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sub(self, _rhs: i32) -> Self::Output {
        LlvmTensor(0 as *mut i32)
    }
}

impl Mul<i32> for LlvmI32Tensor {
    type Output = LlvmI32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn mul(self, _rhs: i32) -> Self::Output {
        LlvmTensor(0 as *mut i32)
    }
}
