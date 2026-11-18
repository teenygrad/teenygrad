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

//! `Resize` over up to three trailing axes, by a table of weights.
//!
//! The kernel knows nothing about interpolation. Which input elements an output
//! element reads, and how much of each, is worked out on the host by
//! [`teeny_core::graph::resize_axis_plan`] and handed over as two parameter slots: an
//! index and a weight per (axis, output coordinate, tap). Nearest, linear and cubic
//! differ only in how many taps they have and what the weights are, and half-pixel,
//! align-corners and asymmetric coordinates differ only in where the taps point, so
//! none of that reaches the device.
//!
//! That division is deliberate. The coordinate transformations are where a resize goes
//! subtly wrong — a half-pixel offset dropped, a rounding that goes the wrong way at
//! exactly .5 — and in a host function they can be tested against the arithmetic
//! directly instead of through a device.
//!
//! Up to three axes are resized, which covers the 2-D and 3-D resizes in the exports;
//! an axis that is not resized is given an extent of 1 and one tap, so it costs
//! nothing. The taps of all three axes are walked as one flat loop rather than three
//! nested ones.

#![allow(non_snake_case)]

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::{FloatBytes, Num};
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// How many trailing axes one `resize_forward` can resize.
pub const MAX_AXES: usize = 3;

/// Forward: `out[o] = sum over taps of weight * src[tap]`.
///
/// `idx_ptr` and `wts_ptr` hold, per axis and then per output coordinate and tap, the
/// input index to read and the weight to give it. `*_off` is where an axis' table
/// starts, `*_taps` how many taps it has, `*_ext` its output extent and `*_stride` its
/// stride in the input.
#[kernel]
pub fn resize_forward<T, D: Num, const BLOCK_SIZE: i32>(
    src_ptr: InPtr<T::Pointer<D>>,
    idx_ptr: InPtr<T::Pointer<i32>>,
    wts_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    in_block: i32,
    a_ext: i32,
    a_stride: i32,
    a_off: i32,
    a_taps: i32,
    b_ext: i32,
    b_stride: i32,
    b_off: i32,
    b_taps: i32,
    c_ext: i32,
    c_stride: i32,
    c_off: i32,
    c_taps: i32,
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
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // The lane's output coordinates, innermost axis first, and whatever leading axes
    // the resize does not touch. There is no integer remainder in the kernel DSL, so
    // each is `a - (a / b) * b`.
    let q_c = p / c_ext;
    let coord_c = p - q_c * c_ext;
    let q_b = q_c / b_ext;
    let coord_b = q_c - q_b * b_ext;
    let q_a = q_b / a_ext;
    let coord_a = q_b - q_a * a_ext;
    let outer = q_a;

    let mut acc = T::zeros::<D>(&[BLOCK_SIZE]);
    let n_taps = a_taps * b_taps * c_taps;
    for t in 0..n_taps {
        // Which tap of each axis this iteration is, peeled the same way.
        let t_rest = t / c_taps;
        let t_c = t - t_rest * c_taps;
        let t_a = t_rest / b_taps;
        let t_b = t_rest - t_a * b_taps;

        let at_a = coord_a * a_taps + t_a + a_off;
        let at_b = coord_b * b_taps + t_b + b_off;
        let at_c = coord_c * c_taps + t_c + c_off;

        let i_a = T::load(
            idx_ptr.add_offsets(at_a),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let i_b = T::load(
            idx_ptr.add_offsets(at_b),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let i_c = T::load(
            idx_ptr.add_offsets(at_c),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_a = T::load(
            wts_ptr.add_offsets(at_a),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_b = T::load(
            wts_ptr.add_offsets(at_b),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_c = T::load(
            wts_ptr.add_offsets(at_c),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        let src = outer * in_block + i_a * a_stride + i_b * b_stride + i_c * c_stride;
        let v = T::load(
            src_ptr.add_offsets(src),
            Some(in_bounds),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + v * w_a * w_b * w_c;
    }

    T::store(
        out_ptr.add_offsets(p),
        acc,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// One resized axis, as the kernel reads it.
pub struct ResizeAxis {
    /// The output's extent along this axis.
    pub out_ext: usize,
    /// The input's stride along it, in elements.
    pub stride: usize,
    /// Taps per output coordinate.
    pub taps: usize,
    /// Clamped input index per (coordinate, tap).
    pub indices: Vec<i32>,
    /// Weight per (coordinate, tap), in the same order.
    pub weights: Vec<f64>,
}

/// Dispatch for [`resize_forward`].
pub struct ResizeRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    kernel: ResizeForward<D>,
    /// Per axis: extent, stride, table offset, taps. Always [`MAX_AXES`] long, with
    /// the unused axes filled in so they cost one tap of weight 1 on index 0.
    axes: Vec<(i32, i32, i32, i32)>,
    indices: Vec<i32>,
    weights: Vec<f64>,
    in_block: usize,
    n_out: usize,
}

impl<D: FloatBytes + Send + Sync + 'static> ResizeRuntimeOp<D> {
    /// `axes` describes the resized axes outermost-first, at most [`MAX_AXES`] of
    /// them; `in_block` is the product of the input's extents over those axes, which
    /// is how far one step of the untouched leading axes moves.
    ///
    /// `None` if there are too many axes, if an axis' table does not hold one entry
    /// per coordinate and tap, or if an axis has no taps.
    pub fn new(
        block_size: i32,
        axes: &[ResizeAxis],
        in_block: usize,
        n_out: usize,
    ) -> Option<Self> {
        if axes.len() > MAX_AXES || axes.is_empty() {
            return None;
        }
        let mut indices: Vec<i32> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut packed: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(MAX_AXES);

        // The axes the resize does not use come first, so that the ones it does stay
        // innermost: the kernel peels coordinates from the innermost axis outwards.
        for _ in axes.len()..MAX_AXES {
            let off = indices.len() as i32;
            indices.push(0);
            weights.push(1.0);
            packed.push((1, 0, off, 1));
        }
        for axis in axes {
            if axis.taps == 0
                || axis.indices.len() != axis.out_ext * axis.taps
                || axis.weights.len() != axis.indices.len()
            {
                return None;
            }
            let off = indices.len() as i32;
            indices.extend_from_slice(&axis.indices);
            weights.extend_from_slice(&axis.weights);
            packed.push((
                axis.out_ext as i32,
                axis.stride as i32,
                off,
                axis.taps as i32,
            ));
        }

        Some(Self {
            kernel: ResizeForward::<D>::new(block_size),
            axes: packed,
            indices,
            weights,
            in_block,
            n_out,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &ResizeForward<D> {
        &self.kernel
    }

    /// The index table, zero-padded to a whole number of the node's elements: a
    /// parameter slot is allocated at the node's own dtype width.
    fn index_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = self.indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let width = core::mem::size_of::<D>();
        let padded = bytes.len().div_ceil(width) * width;
        bytes.resize(padded, 0);
        bytes
    }

    /// The weight table in the node's own dtype, which is what the kernel loads it as.
    ///
    /// The weights are planned in f64 so that a half-pixel offset or a cubic
    /// coefficient is not rounded twice, and narrowed once, here.
    fn weight_bytes(&self) -> Vec<u8> {
        self.weights
            .iter()
            .flat_map(|&w| D::from_f64(w).to_le_bytes())
            .collect()
    }
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ResizeRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        let width = core::mem::size_of::<D>();
        vec![
            vec![self.index_bytes().len().div_ceil(width)],
            vec![self.weights.len()],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["taps", "weights"]
    }

    fn param_init_data(&self, param_idx: usize) -> Option<Vec<u8>> {
        match param_idx {
            0 => Some(self.index_bytes()),
            1 => Some(self.weight_bytes()),
            _ => None,
        }
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
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n_out as i32);
        visitor.visit_i32(self.in_block as i32);
        for &(ext, stride, off, taps) in &self.axes {
            visitor.visit_i32(ext);
            visitor.visit_i32(stride);
            visitor.visit_i32(off);
            visitor.visit_i32(taps);
        }
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n_out.div_ceil(self.kernel.block_size as usize) as u32).max(1),
            1,
            1,
        ]
    }
}
