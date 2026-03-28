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

// Dtype — base marker trait for all types that can flow through the system
pub trait Dtype: Copy + Clone {}

// Num — numeric scalars; BITS is used for device buffer allocation
pub trait Num: Dtype {
    const BITS: u8;
}

pub trait Float: Num {}
pub trait Int: Num {}
pub trait Bool: Dtype + Copy {}

// Floating-point specialisations
pub trait F8E4M3FN: Float {}
pub trait F8E4M3FNUZ: Float {}
pub trait F8E5M2: Float {}
pub trait F8E5M2FNUZ: Float {}
pub trait BF16: Float {}

// Integer specialisations
pub trait I4: Int {}

// Primitive impls
impl Dtype for bool {}

impl Dtype for i8 {}
impl Num for i8 {
    const BITS: u8 = 8;
}
impl Int for i8 {}

impl Dtype for i16 {}
impl Num for i16 {
    const BITS: u8 = 16;
}
impl Int for i16 {}

impl Dtype for i32 {}
impl Num for i32 {
    const BITS: u8 = 32;
}
impl Int for i32 {}

impl Dtype for i64 {}
impl Num for i64 {
    const BITS: u8 = 64;
}
impl Int for i64 {}

impl Dtype for u8 {}
impl Num for u8 {
    const BITS: u8 = 8;
}
impl Int for u8 {}

impl Dtype for u16 {}
impl Num for u16 {
    const BITS: u8 = 16;
}
impl Int for u16 {}

impl Dtype for u32 {}
impl Num for u32 {
    const BITS: u8 = 32;
}
impl Int for u32 {}

impl Dtype for u64 {}
impl Num for u64 {
    const BITS: u8 = 64;
}
impl Int for u64 {}

impl Dtype for f32 {}
impl Num for f32 {
    const BITS: u8 = 32;
}
impl Float for f32 {}

impl Dtype for f64 {}
impl Num for f64 {
    const BITS: u8 = 64;
}
impl Float for f64 {}

// Tensor
pub trait RankedTensor<D: Dtype>: Copy + Clone {}

pub trait Tensor<D: Dtype>: RankedTensor<D> {}

pub trait BoolTensor: Tensor<bool> {}

pub trait Comparison<I: Num> {
    type BoolTensor: BoolTensor;

    fn lt(self, other: I) -> Self::BoolTensor;
}

pub trait I32Tensor: Tensor<i32> + Add<i32> + Comparison<i32> {}

// Offsets trait for adding tensor offsets to pointers
pub trait AddOffsets<I: Int, T: Tensor<I>> {
    type Output;

    fn add_offsets(self, offsets: T) -> Self::Output;
}

// Pointer — Dtype itself (can be stored in tensors), no BITS needed
pub trait Pointer<D: Dtype>:
    Sized + Copy + Clone + Dtype + AddOffsets<i32, Self::I32Tensor> + Add<Self>
{
    type I32Tensor: I32Tensor;
}
