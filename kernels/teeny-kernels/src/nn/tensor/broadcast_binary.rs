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

//! Elementwise arithmetic over operands of different shapes.
//!
//! The elementwise kernels in [`super::elemwise_binary`] are flat: they read both
//! operands at the output's own offsets, which is right only when all three shapes
//! agree. ONNX broadcasts, and an export leans on it heavily: a bias of `[1024]` added
//! to `[2, 129, 1024]`, a scalar divisor, a per-channel scale of `[1, 112, 1, 1]`. Read
//! flat, those are read past their ends, which is a segfault rather than a wrong
//! answer. 523 nodes of the FoundationStereo export at 96x128 are like that.
//!
//! So each lane works out where it is in the output and then where that is in each
//! operand. A length-1 axis gets a stride of 0 on the operand it is short in, which is
//! the whole of broadcasting: every coordinate along it reads the same element. The
//! per-axis extents and strides come from a parameter slot, three numbers per axis
//! innermost-first, so the rank is not capped by the kernel's arity.
//!
//! One kernel per operation rather than one kernel with a selector, because a branch
//! around a vector body is mis-lowered by the frontend.

#![allow(non_snake_case)]

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// How many numbers describe one axis in the `meta` slot.
pub const PER_AXIS: usize = 3;

/// Forward: `out = a + b`, with each operand read through its own strides.
#[kernel]
pub fn broadcast_add_forward<T, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    meta_ptr: InPtr<T::Pointer<i32>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    rank: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    let mut rest = p;
    let mut a_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    let mut b_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    for axis in 0..rank {
        let base = axis * 3;
        let at_ext = T::full::<i32>(&[BLOCK_SIZE], base);
        let at_a = T::full::<i32>(&[BLOCK_SIZE], base + 1);
        let at_b = T::full::<i32>(&[BLOCK_SIZE], base + 2);
        let out_ext = T::load(
            meta_ptr.add_offsets(at_ext),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let a_stride = T::load(
            meta_ptr.add_offsets(at_a),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let b_stride = T::load(
            meta_ptr.add_offsets(at_b),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // This axis' output coordinate, innermost first. There is no integer remainder
        // in the kernel DSL, so it is `a - (a / b) * b`.
        let q = rest / out_ext;
        let c = rest - q * out_ext;
        rest = q;
        a_off = a_off + c * a_stride;
        b_off = b_off + c * b_stride;
    }

    let a = T::load(
        a_ptr.add_offsets(a_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(b_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(p),
        a + b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Forward: `out = a - b`, with each operand read through its own strides.
#[kernel]
pub fn broadcast_sub_forward<T, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    meta_ptr: InPtr<T::Pointer<i32>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    rank: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    let mut rest = p;
    let mut a_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    let mut b_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    for axis in 0..rank {
        let base = axis * 3;
        let at_ext = T::full::<i32>(&[BLOCK_SIZE], base);
        let at_a = T::full::<i32>(&[BLOCK_SIZE], base + 1);
        let at_b = T::full::<i32>(&[BLOCK_SIZE], base + 2);
        let out_ext = T::load(
            meta_ptr.add_offsets(at_ext),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let a_stride = T::load(
            meta_ptr.add_offsets(at_a),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let b_stride = T::load(
            meta_ptr.add_offsets(at_b),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // This axis' output coordinate, innermost first. There is no integer remainder
        // in the kernel DSL, so it is `a - (a / b) * b`.
        let q = rest / out_ext;
        let c = rest - q * out_ext;
        rest = q;
        a_off = a_off + c * a_stride;
        b_off = b_off + c * b_stride;
    }

    let a = T::load(
        a_ptr.add_offsets(a_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(b_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(p),
        a - b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Forward: `out = a * b`, with each operand read through its own strides.
#[kernel]
pub fn broadcast_mul_forward<T, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    meta_ptr: InPtr<T::Pointer<i32>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    rank: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    let mut rest = p;
    let mut a_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    let mut b_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    for axis in 0..rank {
        let base = axis * 3;
        let at_ext = T::full::<i32>(&[BLOCK_SIZE], base);
        let at_a = T::full::<i32>(&[BLOCK_SIZE], base + 1);
        let at_b = T::full::<i32>(&[BLOCK_SIZE], base + 2);
        let out_ext = T::load(
            meta_ptr.add_offsets(at_ext),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let a_stride = T::load(
            meta_ptr.add_offsets(at_a),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let b_stride = T::load(
            meta_ptr.add_offsets(at_b),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // This axis' output coordinate, innermost first. There is no integer remainder
        // in the kernel DSL, so it is `a - (a / b) * b`.
        let q = rest / out_ext;
        let c = rest - q * out_ext;
        rest = q;
        a_off = a_off + c * a_stride;
        b_off = b_off + c * b_stride;
    }

    let a = T::load(
        a_ptr.add_offsets(a_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(b_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(p),
        a * b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Forward: `out = a / b`, with each operand read through its own strides.
#[kernel]
pub fn broadcast_div_forward<T, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    meta_ptr: InPtr<T::Pointer<i32>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    rank: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    let mut rest = p;
    let mut a_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    let mut b_off = T::full::<i32>(&[BLOCK_SIZE], 0);
    for axis in 0..rank {
        let base = axis * 3;
        let at_ext = T::full::<i32>(&[BLOCK_SIZE], base);
        let at_a = T::full::<i32>(&[BLOCK_SIZE], base + 1);
        let at_b = T::full::<i32>(&[BLOCK_SIZE], base + 2);
        let out_ext = T::load(
            meta_ptr.add_offsets(at_ext),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let a_stride = T::load(
            meta_ptr.add_offsets(at_a),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let b_stride = T::load(
            meta_ptr.add_offsets(at_b),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        // This axis' output coordinate, innermost first. There is no integer remainder
        // in the kernel DSL, so it is `a - (a / b) * b`.
        let q = rest / out_ext;
        let c = rest - q * out_ext;
        rest = q;
        a_off = a_off + c * a_stride;
        b_off = b_off + c * b_stride;
    }

    let a = T::load(
        a_ptr.add_offsets(a_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let b = T::load(
        b_ptr.add_offsets(b_off),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(p),
        a / b,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Dispatch for the broadcasting elementwise kernels.
///
/// `axes` describes the output innermost-axis-first: each entry is the output's extent
/// and the two operands' strides, with 0 where an operand is length 1 on that axis.
pub struct BroadcastBinaryRuntimeOp<K, D: Num + Send + Sync + 'static> {
    kernel: K,
    block_size: i32,
    meta: Vec<i32>,
    rank: usize,
    n_out: usize,
    _dtype: core::marker::PhantomData<D>,
}

impl<K, D: Num + Send + Sync + 'static> BroadcastBinaryRuntimeOp<K, D> {
    /// `out_shape`, `a_shape` and `b_shape` are outermost-first, as shapes are written,
    /// and the operands have already been aligned to the output's rank.
    ///
    /// `None` if the ranks disagree, if an axis has no extent, or if an operand's
    /// extent on an axis is neither the output's nor 1, which is not a broadcast.
    pub fn new(
        kernel: K,
        block_size: i32,
        out_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Option<Self> {
        let rank = out_shape.len();
        if rank == 0 || a_shape.len() != rank || b_shape.len() != rank {
            return None;
        }
        if out_shape.contains(&0) {
            return None;
        }

        let strides = |shape: &[usize]| -> Vec<usize> {
            let mut s = vec![1usize; rank];
            for ax in (0..rank.saturating_sub(1)).rev() {
                s[ax] = s[ax + 1] * shape[ax + 1];
            }
            s
        };
        let a_strides = strides(a_shape);
        let b_strides = strides(b_shape);

        let mut meta = Vec::with_capacity(rank * PER_AXIS);
        // Innermost axis first, which is the order the kernel peels coordinates in.
        for ax in (0..rank).rev() {
            for shape in [a_shape, b_shape] {
                if shape[ax] != out_shape[ax] && shape[ax] != 1 {
                    return None;
                }
            }
            meta.push(out_shape[ax] as i32);
            // A length-1 axis is read at the same element whatever the coordinate.
            meta.push(if a_shape[ax] == 1 {
                0
            } else {
                a_strides[ax] as i32
            });
            meta.push(if b_shape[ax] == 1 {
                0
            } else {
                b_strides[ax] as i32
            });
        }

        Some(Self {
            kernel,
            block_size,
            meta,
            rank,
            n_out: out_shape.iter().product(),
            _dtype: core::marker::PhantomData,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// The slot's contents, zero-padded to a whole number of the node's elements: a
    /// slot is allocated at the node's own dtype width.
    fn meta_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = self.meta.iter().flat_map(|v| v.to_le_bytes()).collect();
        let width = core::mem::size_of::<D>();
        let padded = bytes.len().div_ceil(width) * width;
        bytes.resize(padded, 0);
        bytes
    }
}

impl<K: Send + Sync + 'static, D: Num + Send + Sync + 'static> RuntimeOp
    for BroadcastBinaryRuntimeOp<K, D>
{
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![
            self.meta_bytes().len().div_ceil(core::mem::size_of::<D>()),
        ]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["axes"]
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        (param_idx == 0).then(|| self.meta_bytes())
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n_out as i32);
        visitor.visit_i32(self.rank as i32);
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n_out.div_ceil(self.block_size as usize) as u32).max(1),
            1,
            1,
        ]
    }
}
