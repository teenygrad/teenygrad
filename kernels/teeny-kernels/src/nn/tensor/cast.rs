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

//! Converting casts (ONNX `Cast`) between float, integer and bool tensors.
//!
//! Every element travels through one of three lane types:
//!
//! - `f32` for F32 tensors;
//! - `i32` words for I32 and I64 tensors. An I64 element is two words; read, its low
//!   word is the value (both backends are little-endian), exact for values that fit in
//!   an `i32`, which covers the masks, ids and positions exporters cast. Written, the
//!   high word is the low word's sign, so an I64 output holds the full value;
//! - `u8` for Bool tensors, one byte per element, `0` or `1`.
//!
//! A cast to Bool is `x != 0`. A cast from float to an integer truncates towards zero,
//! as ONNX's `Cast` does. Grid: `cdiv(n_elements, BLOCK_SIZE)`.

#![allow(non_snake_case)]

use teeny_core::dtype::{AddOffsets, Comparison, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// I32/I64 (as `i32` words, `in_words` per element) to F32.
#[kernel]
pub fn cast_words_to_f32_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<i32>>,
    y_ptr: OutPtr<T::Pointer<f32>>,
    n_elements: i32,
    in_words: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offs * in_words),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(offs),
        T::cast::<i32, f32>(x, None, false),
        Some(m),
        &[],
        None,
        None,
    );
}

/// Bool to F32: `1.0` or `0.0`.
#[kernel]
pub fn cast_bool_to_f32_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<u8>>,
    y_ptr: OutPtr<T::Pointer<f32>>,
    n_elements: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let t = T::ne(x, T::full::<u8>(&[BLOCK_SIZE], 0));
    T::store(
        y_ptr.add_offsets(offs),
        T::where_(
            t,
            T::full::<f32>(&[BLOCK_SIZE], 1.0),
            T::full::<f32>(&[BLOCK_SIZE], 0.0),
        ),
        Some(m),
        &[],
        None,
        None,
    );
}

/// I32/I64 (as `i32` words) to Bool: `x != 0`.
#[kernel]
pub fn cast_words_to_bool_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<i32>>,
    y_ptr: OutPtr<T::Pointer<u8>>,
    n_elements: i32,
    in_words: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offs * in_words),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(offs),
        T::where_(
            x.ne(0),
            T::full::<u8>(&[BLOCK_SIZE], 1),
            T::full::<u8>(&[BLOCK_SIZE], 0),
        ),
        Some(m),
        &[],
        None,
        None,
    );
}

/// F32 to Bool: `x != 0`.
#[kernel]
pub fn cast_f32_to_bool_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<u8>>,
    n_elements: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let t = T::ne(x, T::full::<f32>(&[BLOCK_SIZE], 0.0));
    T::store(
        y_ptr.add_offsets(offs),
        T::where_(
            t,
            T::full::<u8>(&[BLOCK_SIZE], 1),
            T::full::<u8>(&[BLOCK_SIZE], 0),
        ),
        Some(m),
        &[],
        None,
        None,
    );
}

/// F32 to I32/I64 (`out_words` words per element), truncating towards zero. For I64
/// the high word is written as the sign of the low one.
#[kernel]
pub fn cast_f32_to_words_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<i32>>,
    n_elements: i32,
    out_words: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let v = T::cast::<f32, i32>(x, None, false);
    T::store(
        y_ptr.add_offsets(offs * out_words),
        v,
        Some(m),
        &[],
        None,
        None,
    );
    // The high word, only when there is one (I64): out_words - 1 is 1 or 0, so the
    // mask covers every element or none.
    let izero = T::zeros::<i32>(&[BLOCK_SIZE]);
    let hi = T::where_(v.lt(0), izero + (-1), izero);
    T::store(
        y_ptr.add_offsets(offs * out_words + 1),
        hi,
        Some(offs.lt(n_elements * (out_words - 1))),
        &[],
        None,
        None,
    );
}

/// Bool to I32/I64: `1` or `0`, with a zero high word for I64.
#[kernel]
pub fn cast_bool_to_words_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<u8>>,
    y_ptr: OutPtr<T::Pointer<i32>>,
    n_elements: i32,
    out_words: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let izero = T::zeros::<i32>(&[BLOCK_SIZE]);
    let t = T::ne(x, T::full::<u8>(&[BLOCK_SIZE], 0));
    T::store(
        y_ptr.add_offsets(offs * out_words),
        T::where_(t, izero + 1, izero),
        Some(m),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs * out_words + 1),
        izero,
        Some(offs.lt(n_elements * (out_words - 1))),
        &[],
        None,
        None,
    );
}

/// I32/I64 to I32/I64: the low word copied, the high word (if any) its sign.
#[kernel]
pub fn cast_words_to_words_forward<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<i32>>,
    y_ptr: OutPtr<T::Pointer<i32>>,
    n_elements: i32,
    in_words: i32,
    out_words: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = offs.lt(n_elements);
    let v = T::load(
        x_ptr.add_offsets(offs * in_words),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(offs * out_words),
        v,
        Some(m),
        &[],
        None,
        None,
    );
    let izero = T::zeros::<i32>(&[BLOCK_SIZE]);
    T::store(
        y_ptr.add_offsets(offs * out_words + 1),
        T::where_(v.lt(0), izero + (-1), izero),
        Some(offs.lt(n_elements * (out_words - 1))),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

use teeny_core::device::program::ArgVisitor;
use teeny_core::model::{RawPtr, RuntimeOp};

/// Which conversion a [`CastRuntimeOp`] launches, with its word counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// I32/I64 to F32.
    WordsToF32 { in_words: i32 },
    /// Bool to F32.
    BoolToF32,
    /// I32/I64 to Bool.
    WordsToBool { in_words: i32 },
    /// F32 to Bool.
    F32ToBool,
    /// F32 to I32/I64.
    F32ToWords { out_words: i32 },
    /// Bool to I32/I64.
    BoolToWords { out_words: i32 },
    /// I32/I64 to I32/I64.
    WordsToWords { in_words: i32, out_words: i32 },
}

/// Runtime dispatch for the cast kernels: one activation, no parameters.
pub struct CastRuntimeOp {
    block_size: i32,
    kind: CastKind,
}

impl CastRuntimeOp {
    /// The op for `kind`.
    pub fn new(block_size: i32, kind: CastKind) -> Self {
        Self { block_size, kind }
    }
}

impl RuntimeOp for CastRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
        match self.kind {
            CastKind::WordsToF32 { in_words } | CastKind::WordsToBool { in_words } => {
                visitor.visit_i32(in_words)
            }
            CastKind::F32ToWords { out_words } | CastKind::BoolToWords { out_words } => {
                visitor.visit_i32(out_words)
            }
            CastKind::WordsToWords {
                in_words,
                out_words,
            } => {
                visitor.visit_i32(in_words);
                visitor.visit_i32(out_words);
            }
            CastKind::BoolToF32 | CastKind::F32ToBool => {}
        }
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}
