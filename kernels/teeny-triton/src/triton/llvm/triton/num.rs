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

use std::ops::Mul;

use super::super::super::types as ty;

/*--------------------------------- I1 ---------------------------------*/

pub struct I1(pub bool);
impl Copy for I1 {}
impl Clone for I1 {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl ty::Dtype for I1 {}
impl ty::Num for I1 {}
impl ty::Int for I1 {}
impl ty::I1 for I1 {}

/*--------------------------------- I32 ---------------------------------*/

pub struct I32(pub i32);
impl Copy for I32 {}
impl Clone for I32 {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl ty::Dtype for I32 {}
impl ty::Num for I32 {}
impl ty::Int for I32 {}
impl ty::I32 for I32 {}

impl Mul<u32> for I32 {
    type Output = I32;

    #[inline(always)]
    fn mul(self, rhs: u32) -> Self::Output {
        I32(self.0 * rhs as i32)
    }
}

impl From<u32> for I32 {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self(value as i32)
    }
}

impl From<i32> for I32 {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self(value)
    }
}

/*--------------------------------- I64 ---------------------------------*/

pub struct I64(pub i64);
impl Copy for I64 {}
impl Clone for I64 {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl ty::Dtype for I64 {}
impl ty::Num for I64 {}
impl ty::Int for I64 {}
impl ty::I64 for I64 {}

/*--------------------------------- F32 ---------------------------------*/

pub struct F32(pub f32);
impl ty::Dtype for F32 {}
impl ty::Num for F32 {}
impl ty::Float for F32 {}
impl ty::F32 for F32 {}

impl Copy for F32 {}
impl Clone for F32 {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/*--------------------------------- BF16 ---------------------------------*/

pub struct BF16;
impl Copy for BF16 {}
impl Clone for BF16 {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl ty::Dtype for BF16 {}
impl ty::Num for BF16 {}
impl ty::Float for BF16 {}
impl ty::BF16 for BF16 {}
