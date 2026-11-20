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

//! A tensor of one repeated value (ONNX `ConstantOfShape`).
//!
//! The same shape as [`crate::nn::tensor::constant`], and deliberately not the same
//! op. A constant carries its whole contents as a parameter slot, which is right for
//! a literal the exporter wrote out element by element; a fill knows only a shape and
//! one value, and the exports that use it ask for big ones. FoundationStereo's largest
//! is 55.9 million elements, so carrying the contents would mean 224 MB of graph data
//! for a number that is already known, and the constant folder rightly refuses to
//! evaluate it.
//!
//! So the parameter slot here is a single element, the value, and the kernel writes it
//! across the output. Every lane reads element 0 of the slot, which is a broadcast read
//! rather than a scalar argument because the value's type is the tensor's: passing it
//! as an argument would mean one kernel signature per dtype.

#![allow(non_snake_case)]

use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Forward: `y[i] = v`, with `v` the single element in parameter slot 0.
#[kernel]
pub fn fill_forward<
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    D: Num,
    const BLOCK_SIZE: i32,
>(
    v_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);

    // Every lane at element 0 of the slot: one value, read into a whole tile. The
    // read needs no mask, since element 0 is always there.
    let at_value = T::zeros::<i32>(&[BLOCK_SIZE]);
    let v = T::load(
        v_ptr.add_offsets(at_value),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(offsets),
        v,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Dispatch for [`fill_forward`], carrying the value's bytes.
///
/// `bytes` is the value's little-endian representation in the tensor's dtype, exactly
/// as it appeared in the model, so nothing on the path from file to device converts it.
pub struct FillRuntimeOp<D: Num> {
    kernel: FillForward<D>,
    bytes: Vec<u8>,
}

impl<D: Num> FillRuntimeOp<D> {
    /// `bytes` is one element's little-endian value, or empty for a zero fill, which
    /// is what ONNX uses when `ConstantOfShape` has no `value` attribute.
    ///
    /// Returns `None` if the bytes are neither empty nor exactly one element wide:
    /// `param_init_data`'s contract is that the byte count matches the slot, and a
    /// short slot would be read past its end on the device.
    pub fn new(block_size: i32, bytes: &[u8]) -> Option<Self> {
        let width = core::mem::size_of::<D>();
        let bytes = if bytes.is_empty() {
            vec![0u8; width]
        } else if bytes.len() == width {
            bytes.to_vec()
        } else {
            return None;
        };
        Some(Self {
            kernel: FillForward::new(block_size),
            bytes,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &FillForward<D> {
        &self.kernel
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for FillRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        0
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![1]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        // Named for the same reason a constant's slot is: it is already populated and
        // nothing needs to bind it, but an unnamed slot is invisible to anything
        // walking the model's parameters.
        &["value"]
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        (param_idx == 0).then(|| self.bytes.clone())
    }

    fn pack_args(
        &self,
        _inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.kernel.block_size as usize) as u32, 1, 1]
    }
}
