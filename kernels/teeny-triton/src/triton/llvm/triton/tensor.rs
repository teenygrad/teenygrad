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

pub struct Tensor<D: ty::Dtype>(pub *mut D);
impl<D: ty::Dtype> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: ty::Dtype> Copy for Tensor<D> {}

impl<D: ty::Dtype, const RANK: usize> ty::RankedTensor<D, RANK> for Tensor<D> {
    const SHAPE: [usize; RANK] = [0; RANK];
}
impl<D: ty::Dtype, const RANK: usize> ty::Tensor<D, RANK> for Tensor<D> {}

impl<D: ty::Dtype> Add<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: Tensor<D>) -> Self::Output {
        Tensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Sub<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sub(self, _rhs: Tensor<D>) -> Self::Output {
        Tensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Mul<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn mul(self, _rhs: Tensor<D>) -> Self::Output {
        Tensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Div<Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn div(self, _rhs: Tensor<D>) -> Self::Output {
        Tensor(0 as *mut D)
    }
}

impl<D: ty::Dtype> Neg for Tensor<D> {
    type Output = Tensor<D>;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn neg(self) -> Self::Output {
        Tensor(0 as *mut D)
    }
}

pub type BoolTensor = Tensor<bool>;
impl<const RANK: usize> ty::BoolTensor<RANK> for BoolTensor {}

pub type I32Tensor = Tensor<i32>;

impl<const RANK: usize> ty::I32Tensor<RANK> for I32Tensor {}

impl<D: ty::Num, const RANK: usize> ty::Comparison<D, RANK> for Tensor<D> {
    type BoolTensor = BoolTensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn lt(self, _other: D) -> Self::BoolTensor {
        Tensor(0 as *mut bool)
    }
}

impl Add<i32> for I32Tensor {
    type Output = I32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: i32) -> Self::Output {
        Tensor(0 as *mut i32)
    }
}

impl Sub<i32> for I32Tensor {
    type Output = I32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sub(self, _rhs: i32) -> Self::Output {
        Tensor(0 as *mut i32)
    }
}

impl Mul<i32> for I32Tensor {
    type Output = I32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn mul(self, _rhs: i32) -> Self::Output {
        Tensor(0 as *mut i32)
    }
}
