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

//! Boolean `And`, `Or` and `Xor`.
//!
//! ONNX defines these on bool tensors, and a bool tensor is one byte per element, so
//! the instantiation lowering picks is `u8`: the kernels read a non-zero byte as true
//! and write 1 or 0, never another non-zero value, so a chain of them stays
//! normalised. The constant folder reads a `Value`'s bool the same way, which is what
//! makes folding one of these and running it agree.
//!
//! They are separate from the arithmetic elementwise kernels because `bool` is
//! deliberately not a `Num` dtype: the kernels are generic over `Num` so that one
//! source serves every width, and `u8` is how a byte-wide tensor reaches them, but
//! what they compute is defined only for booleans.
//!
//! Like the rest of the elementwise kernels these are flat: both operands are read at
//! the output's own offsets, so the shapes have to match. ONNX would broadcast them,
//! and lowering refuses a mismatch rather than reading the shorter operand past its
//! end.

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Forward: `out = 1 if a != 0 and b != 0 else 0`
#[kernel]
pub fn logical_and_forward<T: Triton, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<u8>>,
    b_ptr: InPtr<T::Pointer<u8>>,
    out_ptr: OutPtr<T::Pointer<u8>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let a_true = T::ne(a, T::full::<u8>(&[BLOCK_SIZE], 0));
    let b_true = T::ne(b, T::full::<u8>(&[BLOCK_SIZE], 0));
    let one = T::full::<u8>(&[BLOCK_SIZE], 1);
    let both = T::where_(b_true, one, T::full::<u8>(&[BLOCK_SIZE], 0));
    T::store(
        out_ptr.add_offsets(offsets),
        T::where_(a_true, both, T::full::<u8>(&[BLOCK_SIZE], 0)),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Forward: `out = 1 if a != 0 or b != 0 else 0`
#[kernel]
pub fn logical_or_forward<T: Triton, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<u8>>,
    b_ptr: InPtr<T::Pointer<u8>>,
    out_ptr: OutPtr<T::Pointer<u8>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let a_true = T::ne(a, T::full::<u8>(&[BLOCK_SIZE], 0));
    let b_true = T::ne(b, T::full::<u8>(&[BLOCK_SIZE], 0));
    let one = T::full::<u8>(&[BLOCK_SIZE], 1);
    let either = T::where_(
        b_true,
        T::full::<u8>(&[BLOCK_SIZE], 1),
        T::full::<u8>(&[BLOCK_SIZE], 0),
    );
    T::store(
        out_ptr.add_offsets(offsets),
        T::where_(a_true, one, either),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Forward: `out = 1 if exactly one of a, b is non-zero else 0`
///
/// Both operands are normalised to 0 or 1 before they are compared, so that a true
/// byte of 1 and a true byte of 2 are the same value here, as they are for `And` and
/// `Or`.
#[kernel]
pub fn logical_xor_forward<T: Triton, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<u8>>,
    b_ptr: InPtr<T::Pointer<u8>>,
    out_ptr: OutPtr<T::Pointer<u8>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let a = T::load(
        a_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let a_norm = T::where_(
        T::ne(a, T::full::<u8>(&[BLOCK_SIZE], 0)),
        T::full::<u8>(&[BLOCK_SIZE], 1),
        T::full::<u8>(&[BLOCK_SIZE], 0),
    );
    let b_norm = T::where_(
        T::ne(b, T::full::<u8>(&[BLOCK_SIZE], 0)),
        T::full::<u8>(&[BLOCK_SIZE], 1),
        T::full::<u8>(&[BLOCK_SIZE], 0),
    );
    T::store(
        out_ptr.add_offsets(offsets),
        T::where_(
            T::ne(a_norm, b_norm),
            T::full::<u8>(&[BLOCK_SIZE], 1),
            T::full::<u8>(&[BLOCK_SIZE], 0),
        ),
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// `RuntimeOp` for the logical kernels: two operands, no parameters, no backward.
///
/// A boolean has no gradient, so there is nothing to differentiate; the arithmetic
/// elementwise ops get their impl from a macro that also wires a backward kernel.
macro_rules! impl_logical_runtime_op {
    ($Fwd:ident) => {
        impl teeny_core::model::RuntimeOp for $Fwd {
            fn n_activation_inputs(&self) -> usize {
                2
            }

            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
                vec![]
            }

            fn pack_args(
                &self,
                inputs: &[(teeny_core::model::RawPtr, &[usize])],
                _: &[teeny_core::model::RawPtr],
                output: teeny_core::model::RawPtr,
                output_shape: &[usize],
                _: i32,
                visitor: &mut dyn teeny_core::device::program::ArgVisitor,
            ) {
                let n: usize = output_shape.iter().product();
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(inputs[1].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n as i32);
            }

            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

impl_logical_runtime_op!(LogicalAndForward);
impl_logical_runtime_op!(LogicalOrForward);
impl_logical_runtime_op!(LogicalXorForward);
