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

use std::ops::Mul;

use crate::triton::types as ty;

/*--------------------------------- I1 ---------------------------------*/

#[derive(Copy, Clone)]
pub struct I1(bool);

impl ty::Dtype for I1 {}
impl ty::Int for I1 {}
impl ty::I1 for I1 {}

/*--------------------------------- I32 ---------------------------------*/

#[derive(Copy, Clone)]
pub struct I32(i32);
impl ty::Dtype for I32 {}
impl ty::Int for I32 {}
impl ty::I32 for I32 {}

impl Mul<u32> for I32 {
    type Output = I64;

    fn mul(self, rhs: u32) -> Self::Output {
        I64(self.0 as i64 * rhs as i64)
    }
}

impl From<u32> for I32 {
    fn from(value: u32) -> Self {
        Self(value as i32)
    }
}

impl From<i32> for I32 {
    fn from(value: i32) -> Self {
        Self(value as i32)
    }
}

/*--------------------------------- I64 ---------------------------------*/

#[derive(Copy, Clone)]
pub struct I64(i64);
impl ty::Dtype for I64 {}
impl ty::Int for I64 {}
impl ty::I64 for I64 {}
