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

//! `ConvTranspose`: a convolution run backwards, as an upsampling layer.
//!
//! The name describes the arithmetic rather than the intent. A convolution with a
//! stride spreads each input element over a window of the output; a transposed one
//! does the reverse, so each input element contributes to a window of a *larger*
//! output. Written that way it wants atomics, because several input elements land on
//! the same output element.
//!
//! This runs it from the output instead: an output element at `(oh, ow)` collects from
//! every `(kh, kw)` of the kernel whose source input element exists, which is where
//! `oh + pad - kh` divides by the stride and lands inside the input. No two lanes write
//! the same place, so no atomics, and the condition is arithmetic rather than a branch.
//!
//! One lane per output element, and the kernel's taps and input channels are walked as
//! one flat loop. The weight layout is ONNX's, `[C_in, C_out, kH, kW]`, which is the
//! transpose of a forward convolution's.

#![allow(non_snake_case)]

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Forward: `out[n, co, oh, ow] = sum over ci, kh, kw of x[n, ci, ih, iw] * w[ci, co, kh, kw]`.
///
/// `ih` and `iw` are `(oh + pad - kh) / stride` and its horizontal twin, and a tap
/// counts only where those divide exactly and land inside the input.
#[kernel]
pub fn conv_transpose2d_forward<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    C_in: i32,
    C_out: i32,
    H_in: i32,
    W_in: i32,
    H_out: i32,
    W_out: i32,
    KH: i32,
    KW: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    use_bias: i32,
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
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // The output is [N, C_out, H_out, W_out]. There is no integer remainder in the
    // kernel DSL, so each coordinate is `a - (a / b) * b`.
    let q_w = p / W_out;
    let ow = p - q_w * W_out;
    let q_h = q_w / H_out;
    let oh = q_w - q_h * H_out;
    let q_c = q_h / C_out;
    let co = q_h - q_c * C_out;
    let n = q_c;

    let mut acc = T::zeros::<D>(&[BLOCK_SIZE]);
    let taps = C_in * KH * KW;
    for t in 0..taps {
        // Peel (ci, kh, kw) out of the flat tap index.
        let t_kw_rest = t / KW;
        let kw = t - t_kw_rest * KW;
        let ci = t_kw_rest / KH;
        let kh = t_kw_rest - ci * KH;

        // Which input element feeds this output through this tap, if any.
        let th = oh + pad_h - kh;
        let tw = ow + pad_w - kw;
        let ih = th / stride_h;
        let iw = tw / stride_w;

        // It counts only where the stride divides exactly and the element exists.
        // `th - ih * stride_h` is the remainder, and a negative `th` gives a non-zero
        // one or an index the range test rejects.
        let exact_h = T::where_(
            T::eq(th - ih * stride_h, T::full::<i32>(&[BLOCK_SIZE], 0)),
            T::full::<i32>(&[BLOCK_SIZE], 1),
            T::full::<i32>(&[BLOCK_SIZE], 0),
        );
        let exact_w = T::where_(
            T::eq(tw - iw * stride_w, T::full::<i32>(&[BLOCK_SIZE], 0)),
            T::full::<i32>(&[BLOCK_SIZE], 1),
            T::full::<i32>(&[BLOCK_SIZE], 0),
        );
        let ok = T::ne(exact_h * exact_w, T::full::<i32>(&[BLOCK_SIZE], 0));
        let take = in_bounds
            .bitand(ok)
            .bitand(ih.ge(0))
            .bitand(ih.lt(H_in))
            .bitand(iw.ge(0))
            .bitand(iw.lt(W_in));

        let x_at = ((n * C_in + ci) * H_in + ih) * W_in + iw;
        // `co` is the lane's channel and the rest of this is scalar, so the tensor
        // comes first: a scalar on the left of a tensor does not type.
        let w_at = (co + ci * C_out) * (KH * KW) + kh * KW + kw;
        let xv = T::load(
            x_ptr.add_offsets(x_at),
            Some(take),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let wv = T::load(
            w_ptr.add_offsets(w_at),
            Some(take),
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + xv * wv;
    }

    // The bias, when there is one: `use_bias` of 0 pins the read to element 0 and
    // multiplies it out, so a model without a bias needs no second kernel.
    let bias = T::load(
        b_ptr.add_offsets(co * use_bias),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let scaled = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], use_bias), None, false);
    T::store(
        out_ptr.add_offsets(p),
        acc + bias * scaled,
        Some(in_bounds),
        &[],
        None,
        None,
    );
}

/// Dispatch for [`conv_transpose2d_forward`].
pub struct ConvTranspose2dRuntimeOp<D: Num + Send + Sync + 'static> {
    kernel: ConvTranspose2dForward<D>,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    has_bias: bool,
    n_out: usize,
}

impl<D: Num + Send + Sync + 'static> ConvTranspose2dRuntimeOp<D> {
    /// `None` if a stride or a kernel extent is zero, which would make the tap
    /// arithmetic a division by zero or leave nothing to sum.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_size: i32,
        n: usize,
        c_in: usize,
        c_out: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        has_bias: bool,
    ) -> Option<Self> {
        if stride_h == 0 || stride_w == 0 || kh == 0 || kw == 0 || c_in == 0 {
            return None;
        }
        Some(Self {
            kernel: ConvTranspose2dForward::<D>::new(block_size),
            c_in,
            c_out,
            h_in,
            w_in,
            h_out,
            w_out,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            has_bias,
            n_out: n * c_out * h_out * w_out,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &ConvTranspose2dForward<D> {
        &self.kernel
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for ConvTranspose2dRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        // The input, the weights, and the bias when the model has one. They arrive as
        // activations because an exported weight is an Input node bound by name.
        if self.has_bias { 3 } else { 2 }
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        // With no bias the kernel still needs a pointer it may read element 0 of, and
        // multiplies the result out; the weights are the one buffer always there.
        visitor.visit_ptr(if self.has_bias {
            inputs[2].0
        } else {
            inputs[1].0
        });
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n_out as i32);
        visitor.visit_i32(self.c_in as i32);
        visitor.visit_i32(self.c_out as i32);
        visitor.visit_i32(self.h_in as i32);
        visitor.visit_i32(self.w_in as i32);
        visitor.visit_i32(self.h_out as i32);
        visitor.visit_i32(self.w_out as i32);
        visitor.visit_i32(self.kh as i32);
        visitor.visit_i32(self.kw as i32);
        visitor.visit_i32(self.stride_h as i32);
        visitor.visit_i32(self.stride_w as i32);
        visitor.visit_i32(self.pad_h as i32);
        visitor.visit_i32(self.pad_w as i32);
        visitor.visit_i32(i32::from(self.has_bias));
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n_out.div_ceil(self.kernel.block_size as usize) as u32).max(1),
            1,
            1,
        ]
    }
}
