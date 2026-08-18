/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

/// Host-only float byte-conversion helpers (kept separate from [`Float`] so the DSL-embedded
/// `dtype/mod.rs` source doesn't need them — see `teeny-triton`'s crate docs).
pub mod bytes;
pub use bytes::FloatBytes;

/// Base marker trait for all types that can flow through the system as tensor elements.
pub trait Dtype: Copy + Clone {}

/// Numeric scalar dtypes; `BITS` is used for device buffer allocation sizing.
pub trait Num: Dtype {
    /// This type's bit width.
    const BITS: u8;
}

/// Floating-point dtypes.
pub trait Float: Num {
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;

    /// Materialize a scalar constant of this float type from an `f64` literal.
    ///
    /// Kernels are generic over `D: Float`, so they cannot use bare `f32`/`f64`
    /// literals when building constant tensors (e.g. `T::full(shape, value: D)`).
    /// `from_f64` provides the one hook needed to construct arbitrary constants
    /// generically; each concrete float type narrows/keeps the value as needed.
    fn from_f64(value: f64) -> Self;
}

/// Integer dtypes.
pub trait Int: Num {}
/// The boolean dtype.
pub trait Bool: Dtype + Copy {}

// Floating-point specialisations
/// 8-bit float, 4 exponent + 3 mantissa bits.
pub trait F8E4M3FN: Float {}
/// 8-bit float, 4 exponent + 3 mantissa bits, no infinities (unsigned zero variant).
pub trait F8E4M3FNUZ: Float {}
/// 8-bit float, 5 exponent + 2 mantissa bits.
pub trait F8E5M2: Float {}
/// 8-bit float, 5 exponent + 2 mantissa bits, no infinities (unsigned zero variant).
pub trait F8E5M2FNUZ: Float {}
/// `bfloat16`: 16-bit float with an 8-bit exponent (same range as f32, reduced precision).
pub trait BF16: Float {}

// Integer specialisations
/// 4-bit integer.
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
impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl Dtype for f64 {}
impl Num for f64 {
    const BITS: u8 = 64;
}
impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f64(value: f64) -> Self {
        value
    }
}

// Tensor
/// A tensor with a statically-known shape.
pub trait RankedTensor<D: Dtype, const RANK: usize>: Clone {
    /// This tensor's shape.
    const SHAPE: [usize; RANK];
}

/// A tensor of dtype `D` and rank `RANK`.
pub trait Tensor<D: Dtype, const RANK: usize>: RankedTensor<D, RANK> {}

/// Marker trait for eager (non-symbolic) tensors.
///
/// Implement this on any tensor type that computes eagerly. The generic
/// `Layer<T>` impls (Relu, Linear, Softmax, …) are gated on this marker so
/// that the specific `Layer<SymTensor>` impls in `nn::graph` don't conflict.
pub trait EagerTensor {}

/// A tensor of `bool`.
pub trait BoolTensor<const RANK: usize>: Tensor<bool, RANK> {}

/// Element-wise comparison against `Rhs`, producing a boolean-valued tensor.
pub trait Comparison<Rhs = Self> {
    /// The boolean-valued tensor type produced by these comparisons.
    type BoolTensor;

    /// Element-wise less-than.
    fn lt(self, other: Rhs) -> Self::BoolTensor;
    /// Element-wise less-than-or-equal.
    fn le(self, other: Rhs) -> Self::BoolTensor;
    /// Element-wise greater-than.
    fn gt(self, other: Rhs) -> Self::BoolTensor;
    /// Element-wise greater-than-or-equal.
    fn ge(self, other: Rhs) -> Self::BoolTensor;
    /// Element-wise equality.
    fn eq(self, other: Rhs) -> Self::BoolTensor;
    /// Element-wise inequality.
    fn ne(self, other: Rhs) -> Self::BoolTensor;
}

/// A tensor of `i32`.
pub trait I32Tensor<const RANK: usize>: Tensor<i32, RANK> + Add<i32> + Comparison<i32> {}

/// Adding a tensor of integer offsets to `Self` (e.g. a pointer).
pub trait AddOffsets<I: Int, const RANK: usize, T: Tensor<I, RANK>> {
    /// The result of adding offsets.
    type Output;

    /// Adds `offsets` to `self`.
    fn add_offsets(self, offsets: T) -> Self::Output;
}

/// A device pointer to elements of dtype `D`. A `Pointer` is itself a [`Dtype`] (so it can be
/// stored in tensors), but has no `BITS` requirement.
pub trait Pointer<D: Dtype, const RANK: usize>:
    Sized + Copy + Clone + Dtype + AddOffsets<i32, RANK, Self::I32Tensor> + Add<Self>
{
    /// The `i32` tensor type used to offset this pointer.
    type I32Tensor: I32Tensor<RANK>;
}
