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

//! Select slices along one axis by index (ONNX `Gather`).
//!
//! `Gather` replaces one axis of its input with the shape of its index tensor:
//!
//! ```text
//! out[outer, k, inner] = data[outer, indices[k], inner]
//! ```
//!
//! where `outer` ranges over the axes before the gathered one and `inner` over those
//! after. Only three numbers are needed to index that, whatever the ranks involved:
//! how many indices there are, how many elements follow the gathered axis, and how far
//! apart two outer slices are. So unlike
//! [`strided_copy`](crate::nn::tensor::strided_copy) and
//! [`tile`](crate::nn::tensor::tile), this kernel does not need the shapes padded to a
//! fixed rank: collapsing the leading and trailing axes loses nothing, because the
//! mapping is flat in both.
//!
//! What it does need is to *read* an index, which neither of those kernels does. For
//! constant indices ([`gather_forward`]) the lowering writes them, normalised, as `i32`
//! into a parameter slot. Indices known only at run time, such as token ids into an
//! embedding, use [`gather_dyn_forward`], which reads the index tensor's own buffer: the
//! executor sizes it from the tensor's dtype, and the kernel reads an `int64` index as
//! the low word of each pair.
//!
//! Indices are normalised and range-checked by the runtime op before they reach the
//! device, so the kernel does no bounds arithmetic: ONNX permits negative indices, and
//! resolving them once on the host is cheaper and safer than per-element on the device.
//!
//! Grid: `cdiv(n_elements, BLOCK_SIZE)`, one CTA per block of output elements.

#![allow(non_snake_case)]

use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Copy `y[o, k, i] = x[o * outer_stride + idx[k] * inner + i]`.
///
/// `n_idx` is the number of indices, `inner` the number of elements after the gathered
/// axis, `out_block` = `n_idx * inner` the output elements per outer slice, and
/// `outer_stride` = `dim * inner` the input elements per outer slice.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gather_forward<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    i_ptr: InPtr<T::Pointer<i32>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    inner: i32,
    out_block: i32,
    outer_stride: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // Split the flat output index into (outer, k, inner). There is no integer
    // remainder in the kernel DSL, so each is `a - (a / b) * b`.
    let o = p / out_block;
    let rest = p - o * out_block;
    let k = rest / inner;
    let i = rest - k * inner;

    // The index for this output element. Lanes past n_elements are masked, so `k` may
    // be out of range for them and is never dereferenced.
    let idx = T::load(
        i_ptr.add_offsets(k),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    let src = o * outer_stride + idx * inner + i;
    let v = T::load(
        x_ptr.add_offsets(src),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(p), v, Some(in_bounds), &[], None, None);
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use teeny_core::device::program::ArgVisitor;
use teeny_core::model::{RawPtr, RuntimeOp};

/// Why a set of `Gather` arguments cannot be lowered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatherError {
    /// The axis is outside the input's rank.
    AxisOutOfRange {
        /// The axis asked for, already normalised if it was negative.
        axis: i64,
        /// The input's rank.
        rank: usize,
    },
    /// An index is outside `[-dim, dim)`, so it selects nothing.
    IndexOutOfRange {
        /// The offending index, as given.
        index: i64,
        /// The extent of the gathered axis.
        dim: usize,
    },
    /// The gathered axis has extent 0, so there is nothing to select.
    EmptyAxis,
}

impl core::fmt::Display for GatherError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GatherError::AxisOutOfRange { axis, rank } => {
                write!(f, "axis {axis} is out of range for a rank-{rank} input")
            }
            GatherError::IndexOutOfRange { index, dim } => write!(
                f,
                "index {index} is out of range for an axis of extent {dim} \
                 (valid: -{dim}..{dim})"
            ),
            GatherError::EmptyAxis => {
                write!(f, "the gathered axis has extent 0, so no index is valid")
            }
        }
    }
}

/// Runtime dispatch for [`gather_forward`].
pub struct GatherRuntimeOp<D: Num + Send + Sync + 'static> {
    block_size: i32,
    inner: i32,
    out_block: i32,
    outer_stride: i32,
    /// Normalised indices, little-endian `i32`, padded to fill the parameter slot.
    index_bytes: Vec<u8>,
    /// Parameter slot length in elements of `D`.
    slot_elems: usize,
    _phantom: PhantomData<D>,
}

impl<D: Num + Send + Sync + 'static> GatherRuntimeOp<D> {
    /// Builds the op for `Gather(data, indices, axis)`.
    ///
    /// `axis` may be negative, as ONNX allows, and so may the indices; both are
    /// normalised here. Indices are range-checked, so an out-of-range one is a
    /// lowering error rather than an out-of-bounds device read.
    pub fn new(
        block_size: i32,
        in_shape: &[usize],
        axis: i64,
        indices: &[i64],
    ) -> Result<Self, GatherError> {
        let rank = in_shape.len() as i64;
        let axis_norm = if axis < 0 { axis + rank } else { axis };
        if axis_norm < 0 || axis_norm >= rank {
            return Err(GatherError::AxisOutOfRange {
                axis,
                rank: rank as usize,
            });
        }
        let ax = axis_norm as usize;
        let dim = in_shape[ax];
        if dim == 0 {
            return Err(GatherError::EmptyAxis);
        }

        let inner: usize = in_shape[ax + 1..].iter().product();
        let n_idx = indices.len();

        let mut idx_i32: Vec<i32> = Vec::with_capacity(n_idx);
        for &i in indices {
            let norm = if i < 0 { i + dim as i64 } else { i };
            if norm < 0 || norm >= dim as i64 {
                return Err(GatherError::IndexOutOfRange { index: i, dim });
            }
            idx_i32.push(norm as i32);
        }

        // The slot is allocated in elements of `D`, so size it to hold `n_idx` i32s
        // and pad the tail. `param_init_data`'s contract is an exact byte count.
        let elem = core::mem::size_of::<D>().max(1);
        let slot_elems = (n_idx * 4).div_ceil(elem).max(1);
        let mut index_bytes = Vec::with_capacity(slot_elems * elem);
        for v in &idx_i32 {
            index_bytes.extend_from_slice(&v.to_le_bytes());
        }
        index_bytes.resize(slot_elems * elem, 0);

        Ok(Self {
            block_size,
            inner: inner as i32,
            out_block: (n_idx * inner) as i32,
            outer_stride: (dim * inner) as i32,
            index_bytes,
            slot_elems,
            _phantom: PhantomData,
        })
    }

    /// The output shape for `Gather(data, indices, axis)`: the gathered axis is
    /// replaced by the index tensor's own shape.
    ///
    /// A scalar index tensor therefore *removes* the axis, which is how a single-slice
    /// select reduces rank.
    pub fn output_shape(in_shape: &[usize], axis: i64, index_shape: &[usize]) -> Vec<usize> {
        let rank = in_shape.len() as i64;
        let ax = (if axis < 0 { axis + rank } else { axis }) as usize;
        let mut out = Vec::with_capacity(in_shape.len() - 1 + index_shape.len());
        out.extend_from_slice(&in_shape[..ax]);
        out.extend_from_slice(index_shape);
        out.extend_from_slice(&in_shape[ax + 1..]);
        out
    }

    /// The kernel's scalar arguments after the three pointers, in order:
    /// `n_elements`, `inner`, `out_block`, `outer_stride`. Exposed so callers and
    /// tests can see the mapping without going through an [`ArgVisitor`].
    pub fn kernel_scalars(&self, n_elements: i32) -> [i32; 4] {
        [n_elements, self.inner, self.out_block, self.outer_stride]
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for GatherRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.slot_elems]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["indices"]
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        (param_idx == 0).then(|| self.index_bytes.clone())
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        for v in self.kernel_scalars(n as i32) {
            visitor.visit_i32(v);
        }
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

// ---------------------------------------------------------------------------
// Runtime indices
// ---------------------------------------------------------------------------

/// [`gather_forward`] with its indices read from an activation buffer at run time: an
/// embedding lookup of token ids, say, which no amount of constant folding can resolve.
///
/// The index buffer is the index tensor's own, sized from its dtype by the executor, and
/// read as `i32` words: `idx_words` is 1 for an `int32` tensor and 2 for an `int64` one,
/// whose low word (both backends are little-endian) holds the index exactly whenever it
/// fits in an `i32`, negative ones included. Negative indices wrap by `dim` as ONNX
/// defines; an index still outside `[0, dim)` is clamped rather than read out of bounds.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn gather_dyn_forward<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    i_ptr: InPtr<T::Pointer<i32>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    inner: i32,
    out_block: i32,
    outer_stride: i32,
    dim: i32,
    idx_words: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    let o = p / out_block;
    let rest = p - o * out_block;
    let k = rest / inner;
    let i = rest - k * inner;

    let raw = T::load(
        i_ptr.add_offsets(k * idx_words),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let izero = T::zeros::<i32>(&[BLOCK_SIZE]);
    let wrapped = T::where_(raw.lt(0), raw + dim, raw);
    let idx = T::maximum(T::minimum(wrapped, izero + (dim - 1)), izero);

    let src = o * outer_stride + idx * inner + i;
    let v = T::load(
        x_ptr.add_offsets(src),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(y_ptr.add_offsets(p), v, Some(in_bounds), &[], None, None);
}

/// Runtime dispatch for [`gather_dyn_forward`]: activations `(data, indices)`, no
/// parameters.
pub struct GatherDynRuntimeOp<D: Num + Send + Sync + 'static> {
    block_size: i32,
    inner: i32,
    out_block: i32,
    outer_stride: i32,
    dim: i32,
    idx_words: i32,
    _phantom: PhantomData<D>,
}

impl<D: Num + Send + Sync + 'static> GatherDynRuntimeOp<D> {
    /// Builds the op for `Gather(data, indices, axis)` with `n_idx` indices of
    /// `idx_bytes` bytes each (4 or 8).
    pub fn new(
        block_size: i32,
        in_shape: &[usize],
        axis: i64,
        n_idx: usize,
        idx_bytes: usize,
    ) -> Result<Self, GatherError> {
        let rank = in_shape.len() as i64;
        let axis_norm = if axis < 0 { axis + rank } else { axis };
        if axis_norm < 0 || axis_norm >= rank {
            return Err(GatherError::AxisOutOfRange {
                axis,
                rank: rank as usize,
            });
        }
        let ax = axis_norm as usize;
        let dim = in_shape[ax];
        if dim == 0 {
            return Err(GatherError::EmptyAxis);
        }
        let inner: usize = in_shape[ax + 1..].iter().product();
        Ok(Self {
            block_size,
            inner: inner as i32,
            out_block: (n_idx * inner) as i32,
            outer_stride: (dim * inner) as i32,
            dim: dim as i32,
            idx_words: (idx_bytes / 4).max(1) as i32,
            _phantom: PhantomData,
        })
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for GatherDynRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
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
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        for v in [
            n as i32,
            self.inner,
            self.out_block,
            self.outer_stride,
            self.dim,
            self.idx_words,
        ] {
            visitor.visit_i32(v);
        }
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}
