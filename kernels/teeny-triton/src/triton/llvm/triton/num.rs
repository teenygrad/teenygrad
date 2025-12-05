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

use std::ops::{Add, Mul};

use crate::triton::{
    llvm::triton::{BoolLike, IntLike, types::AnyType},
    types as ty,
};

#[derive(Copy, Clone)]
pub enum Int {}

impl ty::Int for Int {}

impl ty::Dtype for Int {}

/*--------------------------------- I1 ---------------------------------*/

#[derive(Copy, Clone)]
pub struct I1 {}

impl ty::I1<BoolLike, AnyType> for I1 {}
impl ty::Int for I1 {}
impl ty::Dtype for I1 {}

impl From<I1> for AnyType {
    fn from(_: I1) -> Self {
        todo!()
    }
}

impl From<I1> for BoolLike {
    fn from(_: I1) -> Self {
        todo!()
    }
}

/*--------------------------------- I32 ---------------------------------*/

#[derive(Copy, Clone)]
pub struct I32 {}

impl ty::I32<IntLike, AnyType, I64> for I32 {}
impl ty::Int for I32 {}
impl ty::Dtype for I32 {}

impl From<isize> for I32 {
    fn from(_: isize) -> Self {
        todo!()
    }
}

impl From<I32> for IntLike {
    fn from(_: I32) -> Self {
        todo!()
    }
}

impl From<I32> for I64 {
    fn from(_: I32) -> Self {
        todo!()
    }
}

impl From<I32> for AnyType {
    fn from(_: I32) -> Self {
        todo!()
    }
}

impl Add<I32> for I32 {
    type Output = I64;

    fn add(self, _other: I32) -> Self::Output {
        todo!()
    }
}

impl Mul<I32> for I32 {
    type Output = I64;

    fn mul(self, _other: I32) -> Self::Output {
        todo!()
    }
}

/*--------------------------------- I64 ---------------------------------*/

#[derive(Copy, Clone)]
pub struct I64 {}

impl ty::I64<IntLike, AnyType> for I64 {}
impl ty::Int for I64 {}
impl ty::Dtype for I64 {}

impl From<I64> for IntLike {
    fn from(_: I64) -> Self {
        todo!()
    }
}

impl From<I64> for AnyType {
    fn from(_: I64) -> Self {
        todo!()
    }
}
