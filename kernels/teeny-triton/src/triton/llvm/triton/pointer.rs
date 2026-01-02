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
impl<D: ty::Dtype> ty::AddOffsets<D, I32, I32Tensor> for Pointer<D> {
    type Pointer = Pointer<D>;
    type Output = Tensor<Self::Pointer>;

    #[inline(never)]
    fn add_offsets(self, _offsets: I32Tensor) -> Self::Output {
        // dummy implementation not used in final output
        loop {}
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
