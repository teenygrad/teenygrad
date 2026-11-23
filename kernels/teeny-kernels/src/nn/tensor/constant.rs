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

//! A constant tensor whose values live in the graph rather than in a weight file.
//!
//! Most constants in an imported model are either weights, bound by name from a
//! checkpoint, or shape metadata, which lowering reads and elides (see
//! [`crate::graph::elidable_constants`]). What is left is a third kind: a literal the
//! exporter wrote into the graph that genuinely flows data, such as a bias folded out
//! of a layer or an attention mask built at trace time. Nothing outside the model file
//! knows its value, so it cannot be bound by name, and something has to put it on the
//! device.
//!
//! [`ConstantRuntimeOp`] does that by declaring one parameter slot and handing its
//! bytes to [`RuntimeOp::param_init_data`], which both drivers already copy into the
//! slot right after allocating it. The kernel then copies the slot to the node's
//! output, for the same reason the shape-only operators lower to a copy: the
//! executable DAG has no node type that aliases another buffer, so one node's output
//! cannot simply *be* a parameter buffer.

#![allow(non_snake_case)]

use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Forward: `y = c`, with `c` the constant in parameter slot 0.
#[kernel]
pub fn constant_forward<T: Triton, D: Num, const BLOCK_SIZE: i32>(
    c_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let c = T::load(
        c_ptr.add_offsets(offsets),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(offsets),
        c,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Dispatch for [`constant_forward`], carrying the constant's bytes.
///
/// The kernel is the same copy for every constant; what differs per node is the data,
/// which is why this is a wrapper around the generated kernel struct rather than the
/// kernel struct itself. `bytes` is the tensor's little-endian row-major contents,
/// exactly as it appeared in the model, so no dtype conversion happens anywhere on the
/// path from file to device.
pub struct ConstantRuntimeOp<D: Num> {
    kernel: ConstantForward<D>,
    /// Little-endian element bytes of the constant.
    bytes: Vec<u8>,
    /// The constant's static shape.
    shape: Vec<usize>,
}

impl<D: Num> ConstantRuntimeOp<D> {
    /// `shape` is the constant's static shape and `bytes` its little-endian contents.
    ///
    /// Returns `None` if the two disagree, which would otherwise surface as a buffer
    /// over-read on the device: `param_init_data`'s contract is that the byte count
    /// matches the slot exactly.
    pub fn new(block_size: i32, shape: Vec<usize>, bytes: Vec<u8>) -> Option<Self> {
        let n: usize = shape.iter().product();
        if n * core::mem::size_of::<D>() != bytes.len() {
            return None;
        }
        Some(Self {
            kernel: ConstantForward::new(block_size),
            bytes,
            shape,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &ConstantForward<D> {
        &self.kernel
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for ConstantRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        0
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![self.shape.clone()]
    }

    fn param_names(&self) -> &'static [&'static str] {
        // Named so the slot shows up in `param_info_named`, which makes a constant
        // visible to anything walking a model's parameters. The value is already
        // populated, so nothing needs to bind it, but leaving it unnamed would hide it.
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
