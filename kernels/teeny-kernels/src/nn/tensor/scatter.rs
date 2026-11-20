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

//! `ScatterND`: a copy of one tensor with some positions overwritten.
//!
//! The op is two passes over the output, and this is one kernel that serves as both.
//! The first copies `data` across, the second writes the updates where the indices
//! say. The difference between them is arguments alone: a coordinate width of zero
//! makes the index loop run no iterations and the destination of a lane its own
//! position, which is exactly a copy. There is no flag and no branch, because a scalar
//! `if` around a vector body is mis-lowered by the kernel frontend.
//!
//! The output's strides and extents for the indexed axes live in a parameter slot
//! rather than in the argument list, so the coordinate width is not capped by the
//! kernel's arity: the export's own scatters index five axes. The extents are there to
//! resolve a negative index against its own axis, as ONNX defines it.
//!
//! Indices arrive as a buffer rather than a parameter slot, unlike `Gather`'s: they are
//! computed in the export rather than constant, so there is nothing to bake in.
//! Duplicate coordinates are undefined in ONNX and stay undefined here, where the
//! last lane to write a position wins.

#![allow(non_snake_case)]

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Forward: `out[coord(lane)] = src[lane]`, one lane per element this launch writes.
///
/// `k` is the coordinate width: 0 copies `src` straight across, which is the pass that
/// seeds the output from `data`. `slice_len` is how many elements one index row
/// addresses, 1 for that copy. `meta_ptr` holds `k` output strides followed by `k`
/// output extents.
#[kernel]
pub fn scatter_nd_forward<T, D: Num, const BLOCK_SIZE: i32>(
    src_ptr: InPtr<T::Pointer<D>>,
    idx_ptr: InPtr<T::Pointer<i64>>,
    meta_ptr: InPtr<T::Pointer<i32>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    k: i32,
    slice_len: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
    T::Pointer<i64>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i64>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // Which index row this lane belongs to, and where it sits in that row's slice.
    // There is no integer remainder in the kernel DSL, so it is `a - (a / b) * b`.
    let row = p / slice_len;
    let within = p - row * slice_len;

    let mut dst = within;
    for j in 0..k {
        // One stride and one extent, read into every lane: the slot holds strides
        // first, then extents.
        let at_stride = T::full::<i32>(&[BLOCK_SIZE], j);
        let at_extent = T::full::<i32>(&[BLOCK_SIZE], k + j);
        let stride = T::load(
            meta_ptr.add_offsets(at_stride),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let extent = T::load(
            meta_ptr.add_offsets(at_extent),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        let coord = T::load(
            idx_ptr.add_offsets(row * k + j),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let coord = T::cast::<i64, i32>(coord, None, false);
        // ONNX counts a negative coordinate from the end of its own axis.
        let wrapped = T::where_(coord.lt(0), extent, T::full::<i32>(&[BLOCK_SIZE], 0));
        dst = dst + (coord + wrapped) * stride;
    }

    let v = T::load(
        src_ptr.add_offsets(p),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(dst),
        v,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Forward: `ScatterElements`, one index per element along a single axis.
///
/// `out[o, idx[o, j, n], n] = src[o, j, n]`, writing `o` for the coordinates in front
/// of the axis and `n` for those behind it. As with [`scatter_nd_forward`] the same
/// kernel runs the copy that seeds the output: `use_index` of 0 drops the index term
/// and the axis arithmetic collapses to the identity, and it also pins the index read
/// to element 0 so the copy pass, which runs over more elements than the index tensor
/// holds, never reads past its end.
#[kernel]
pub fn scatter_elements_forward<T, D: Num, const BLOCK_SIZE: i32>(
    src_ptr: InPtr<T::Pointer<D>>,
    idx_ptr: InPtr<T::Pointer<i64>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    mid: i32,
    inner: i32,
    axis_len: i32,
    use_index: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i64>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i64>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // Split the lane's position into (before the axis, along it, behind it). There is
    // no integer remainder in the kernel DSL, so each is `a - (a / b) * b`.
    let block = mid * inner;
    let o = p / block;
    let rest = p - o * block;
    let j = rest / inner;
    let n = rest - j * inner;

    let coord = T::load(
        idx_ptr.add_offsets(p * use_index),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let coord = T::cast::<i64, i32>(coord, None, false);
    // ONNX counts a negative index from the end of the axis.
    let wrapped = T::where_(
        coord.lt(0),
        T::full::<i32>(&[BLOCK_SIZE], axis_len),
        T::full::<i32>(&[BLOCK_SIZE], 0),
    );
    let along = (coord + wrapped) * use_index + j * (1 - use_index);

    let dst = o * (axis_len * inner) + along * inner + n;
    let v = T::load(
        src_ptr.add_offsets(p),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        out_ptr.add_offsets(dst),
        v,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Dispatch for [`scatter_elements_forward`]: the copy pass, then the scatter pass.
///
/// The copy pass walks the output as one flat run: `mid` of 1 and `inner` of 1 make a
/// lane's position its own, as long as `axis_len` is 1 there too, which is why the
/// two passes pass different values for all of them.
pub struct ScatterElementsRuntimeOp<D: Num + Send + Sync + 'static> {
    kernel: ScatterElementsForward<D>,
    /// Elements along the scattered axis of the output.
    axis_len: usize,
    /// Elements along that axis in the updates, which is where a lane's own
    /// coordinate comes from.
    updates_mid: usize,
    /// Elements behind the axis, the same in both.
    inner: usize,
    n_data: usize,
    n_updates: usize,
}

impl<D: Num + Send + Sync + 'static> ScatterElementsRuntimeOp<D> {
    /// `None` if the shapes give nothing to index: an empty axis, or no elements
    /// behind it, either of which makes the lane arithmetic a division by zero.
    pub fn new(
        block_size: i32,
        axis_len: usize,
        updates_mid: usize,
        inner: usize,
        n_data: usize,
        n_updates: usize,
    ) -> Option<Self> {
        if axis_len == 0 || inner == 0 {
            return None;
        }
        Some(Self {
            kernel: ScatterElementsForward::<D>::new(block_size),
            axis_len,
            updates_mid,
            inner,
            n_data,
            n_updates,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &ScatterElementsForward<D> {
        &self.kernel
    }

    fn tiles(&self, n: usize) -> u32 {
        (n.div_ceil(self.kernel.block_size as usize) as u32).max(1)
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for ScatterElementsRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        3
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn n_launches(&self) -> usize {
        2
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        self.pack_args_for_launch(
            0,
            inputs,
            params,
            output,
            output_shape,
            output_row_stride,
            visitor,
        );
    }

    fn pack_args_for_launch(
        &self,
        launch_idx: usize,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let copy = launch_idx == 0;
        visitor.visit_ptr(if copy { inputs[0].0 } else { inputs[2].0 });
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(if copy { self.n_data } else { self.n_updates } as i32);
        visitor.visit_i32(if copy { 1 } else { self.updates_mid } as i32);
        visitor.visit_i32(if copy { 1 } else { self.inner } as i32);
        visitor.visit_i32(if copy { 1 } else { self.axis_len } as i32);
        visitor.visit_i32(i32::from(!copy));
    }

    fn grid_for_launch(
        &self,
        launch_idx: usize,
        _input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> [u32; 3] {
        let n = if launch_idx == 0 {
            self.n_data
        } else {
            self.n_updates
        };
        [self.tiles(n), 1, 1]
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [self.tiles(self.n_data), 1, 1]
    }
}

/// Dispatch for [`scatter_nd_forward`]: the copy pass, then the scatter pass.
///
/// `strides` and `extents` describe the output's indexed axes, and go in the parameter
/// slot the kernel reads them from. `slice_len` is the number of elements one index row
/// addresses, and `n_data` and `n_updates` are the two passes' element counts.
pub struct ScatterNdRuntimeOp<D: Num + Send + Sync + 'static> {
    kernel: ScatterNdForward<D>,
    meta: Vec<i32>,
    k: usize,
    slice_len: usize,
    n_data: usize,
    n_updates: usize,
}

impl<D: Num + Send + Sync + 'static> ScatterNdRuntimeOp<D> {
    /// `None` if the pieces do not describe one scatter: a coordinate width that does
    /// not match the number of strides and extents given, or a slice length of zero,
    /// which would make the lane-to-row arithmetic a division by zero.
    pub fn new(
        block_size: i32,
        strides: &[usize],
        extents: &[usize],
        slice_len: usize,
        n_data: usize,
        n_updates: usize,
    ) -> Option<Self> {
        if strides.len() != extents.len() || strides.is_empty() || slice_len == 0 {
            return None;
        }
        let mut meta: Vec<i32> = Vec::with_capacity(strides.len() * 2);
        meta.extend(strides.iter().map(|&s| s as i32));
        meta.extend(extents.iter().map(|&e| e as i32));
        Some(Self {
            kernel: ScatterNdForward::<D>::new(block_size),
            k: strides.len(),
            meta,
            slice_len,
            n_data,
            n_updates,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &ScatterNdForward<D> {
        &self.kernel
    }

    fn tiles(&self, n: usize) -> u32 {
        n.div_ceil(self.kernel.block_size as usize) as u32
    }

    /// The slot's contents, zero-padded to a whole number of the node's elements.
    fn meta_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = self.meta.iter().flat_map(|v| v.to_le_bytes()).collect();
        let width = core::mem::size_of::<D>();
        let padded = bytes.len().div_ceil(width) * width;
        bytes.resize(padded, 0);
        bytes
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for ScatterNdRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        3
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        // A parameter slot is allocated in the node's own dtype, so its length is in
        // those elements however wide they are, while what goes in it here is i32.
        // Rounding up keeps the slot at least as large as the bytes written to it: an
        // I64 node would otherwise be handed half the bytes its slot expects.
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

    fn n_launches(&self) -> usize {
        2
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        self.pack_args_for_launch(
            0,
            inputs,
            params,
            output,
            output_shape,
            output_row_stride,
            visitor,
        );
    }

    fn pack_args_for_launch(
        &self,
        launch_idx: usize,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        // Launch 0 copies `data` across with a coordinate width of 0; launch 1 writes
        // the updates where the indices say.
        let copy = launch_idx == 0;
        visitor.visit_ptr(if copy { inputs[0].0 } else { inputs[2].0 });
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(if copy { self.n_data } else { self.n_updates } as i32);
        visitor.visit_i32(if copy { 0 } else { self.k } as i32);
        // The copy pass is one row covering the whole tensor: that makes a lane's
        // row 0 and its position within the row its own index, which is the identity
        // mapping. A slice length of 1 would instead make every lane position 0 and
        // every write land on the first element.
        visitor.visit_i32(if copy {
            self.n_data.max(1)
        } else {
            self.slice_len
        } as i32);
    }

    fn grid_for_launch(
        &self,
        launch_idx: usize,
        _input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> [u32; 3] {
        let n = if launch_idx == 0 {
            self.n_data
        } else {
            self.n_updates
        };
        [self.tiles(n).max(1), 1, 1]
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [self.tiles(self.n_data).max(1), 1, 1]
    }
}
