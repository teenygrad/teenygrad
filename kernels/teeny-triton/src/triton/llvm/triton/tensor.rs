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

use std::ops::Add;

use super::super::super::types::{self as ty};
use super::{
    num::{I32, I64},
    types::Bool,
};

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

pub type BoolTensor = Tensor<Bool>;
impl ty::BoolTensor for BoolTensor {
    type Bool = Bool;
}

pub type I32Tensor = Tensor<I32>;

impl ty::I32Tensor for I32Tensor {
    type I32 = I32;
}

impl ty::Comparison<I32> for I32Tensor {
    type BoolTensor = BoolTensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn lt(&self, _other: I32) -> Self::BoolTensor {
        // dummy implementation not used in final output
        Tensor(0 as *mut Bool)
    }
}

// Blanket implementation for any type implementing I64, including <I32 as Mul<u32>>::Output
impl<R: ty::I32> Add<R> for I32Tensor {
    type Output = I32Tensor;

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn add(self, _rhs: R) -> Self::Output {
        // dummy implementation not used in final output
        Tensor(0 as *mut I32)
    }
}
