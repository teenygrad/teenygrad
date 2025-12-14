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

use crate::triton::{
    llvm::triton::{
        num::{I32, I64},
        types::Bool,
    },
    types::{self as ty},
};

/*--------------------------------- Tensor ---------------------------------*/

pub struct Tensor<D: ty::Dtype> {
    _phantom_1: std::marker::PhantomData<D>,
}

impl<D: ty::Dtype> ty::Tensor<D> for Tensor<D> {}
impl<D: ty::Dtype> ty::RankedTensor<D> for Tensor<D> {}

pub type BoolTensor = Tensor<Bool>;
impl ty::BoolTensor for BoolTensor {
    type B = Bool;
}

pub type I32Tensor = Tensor<I32>;

impl ty::I32Tensor for I32Tensor {
    type I32 = I32;
    type I64 = I64;
    type I64Tensor = I64Tensor;
}

impl ty::TensorComparison<I64> for I32Tensor {
    type BoolTensor = BoolTensor;

    fn less_than(&self, _other: I64) -> Self::BoolTensor {
        todo!()
    }
}

// Blanket implementation for any type implementing I64, including <I32 as Mul<u32>>::Output
impl<R: ty::I64> Add<R> for I32Tensor {
    type Output = I64Tensor;

    fn add(self, _rhs: R) -> Self::Output {
        todo!()
    }
}

pub type I64Tensor = Tensor<I64>;
impl ty::I64Tensor for I64Tensor {
    type I32 = I32;
    type I64 = I64;
}

impl ty::TensorComparison<I32> for I64Tensor {
    type BoolTensor = BoolTensor;

    fn less_than(&self, _other: I32) -> Self::BoolTensor {
        todo!()
    }
}
