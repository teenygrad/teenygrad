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

use crate::triton::{
    llvm::triton::{num::I32, types::Bool},
    types::{self as ty},
};

/*--------------------------------- Tensor ---------------------------------*/

pub struct Tensor<D: ty::Dtype> {
    _phantom_1: std::marker::PhantomData<D>,
}

impl<D: ty::Dtype> ty::Tensor<D> for Tensor<D> {}
impl<D: ty::Dtype> ty::RankedTensor<D> for Tensor<D> {}

pub type BoolTensor = Tensor<Bool>;
impl ty::BoolTensor<Bool> for BoolTensor {}

pub type I32Tensor = Tensor<I32>;
impl ty::I32Tensor<I32> for I32Tensor {}
