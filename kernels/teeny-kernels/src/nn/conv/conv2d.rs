/*
 * Copyright (c) 2026 Teenygrad.
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

#![allow(non_snake_case)]

use core::ops::BitAnd;

use teeny_core::dtype::Num;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// 2-D convolution forward pass (supports grouped and depthwise conv).
///
/// Grid: one CTA per (b, c_out, oh, ow-tile):
///   `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Each CTA computes a BLOCK_OW-wide strip of output columns by iterating over
/// `(C_IN/G) * KH * KW` combinations for its assigned group.  The weight for
/// each `(c_in_local, kh, kw)` is loaded as a [1] tensor and broadcast to
/// `[BLOCK_OW]`.
///
/// `G` is the number of groups (1 = standard conv, G = C_IN = C_OUT for depthwise).
/// Weight layout: `[C_OUT, C_IN/G, KH, KW]`.
///
/// Zero-padding of `PAD_H` / `PAD_W` elements is applied on each spatial side.
/// `OH = (H + 2*PAD_H - KH) / STRIDE_H + 1`, `OW = (W + 2*PAD_W - KW) / STRIDE_W + 1`.
#[kernel]
pub fn conv2d_forward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const G: i32,
    const BLOCK_OW: i32,
>(
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    _B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    // Decode flat pid → (b, c_out, oh, ow_tile).
    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let out_bc_base = (b * C_OUT + c_out) * OH * OW;

    // Group index for this output channel and its input-channel window.
    let c_in_per_group = C_IN / G;
    let g_idx = c_out / (C_OUT / G);
    let c_in_start = g_idx * c_in_per_group;

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    // Flat loop over (C_IN/G) * KH * KW combinations for this group.
    let loop_bound = c_in_per_group * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH; // index within the group
        let c_in = c_in_start + c_in_local; // absolute input channel

        // Compute padded input coordinates; OOB height rows contribute zero via mask.
        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        // `ow_range * 0` is the only way to splat scalar ih into an I32Tensor.
        // A scalar `if`/`continue` here triggers a compiler phi-node bug.
        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let load_mask = ow_mask & h_in_bounds & w_in_bounds;

        let x_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(load_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        // Weight layout [C_OUT, C_IN/G, KH, KW]: load scalar and broadcast.
        let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        acc = acc + x_tile * w_tile;
    }

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        y_ptr.add_offsets(out_offsets),
        acc,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

/// 2-D convolution backward pass — gradient with respect to input (`dx`).
///
/// Uses the same grid as the forward pass (over output positions) and scatters
/// the gradient back to the input via `atomic_add`, which correctly handles
/// overlapping receptive fields when `STRIDE < kernel size`.
///
/// `G` — number of groups (1 = standard conv, G = C_IN = C_OUT for depthwise).
/// Weight layout: `[C_OUT, C_IN/G, KH, KW]`.
///
/// Grid: `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// Padding positions (those that correspond to out-of-bounds input locations)
/// are skipped.
#[kernel]
pub fn conv2d_backward_dx<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const G: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    _B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    // Load the upstream gradient tile for this output row.
    let dy_offsets = ow_range + ((b * C_OUT + c_out) * OH * OW + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );

    // Group index for this output channel and its input-channel window.
    let c_in_per_group = C_IN / G;
    let g_idx = c_out / (C_OUT / G);
    let c_in_start = g_idx * c_in_per_group;

    // Scatter gradient to input via flat loop over (C_IN/G) * KH * KW.
    let loop_bound = c_in_per_group * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH;
        let c_in = c_in_start + c_in_local;

        let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        let grad_tile = dy_tile * w_tile;

        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);

        let dx_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad_tile,
            Some(ow_mask & h_in_bounds & w_in_bounds),
            None,
            None,
        );
    }
}

/// 2-D convolution backward pass — gradient with respect to weights (`dw`).
///
/// Uses the same grid as the forward pass.  Each CTA at `(b, c_out, oh, ow_tile)`
/// accumulates partial sums into `dw` via `atomic_add` (one per `(c_in_local, kh, kw)`
/// combination within the group).
///
/// `G` — number of groups. Weight layout: `[C_OUT, C_IN/G, KH, KW]`.
/// Grid: `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// `dw` must be zero-initialised before launch.
#[kernel]
pub fn conv2d_backward_dw<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const G: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    dw_ptr: T::Pointer<D>,
    _B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    // Load dy tile for this output row.
    let dy_offsets = ow_range + ((b * C_OUT + c_out) * OH * OW + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );

    let ih_base = oh * STRIDE_H;

    // Group index for this output channel and its input-channel window.
    let c_in_per_group = C_IN / G;
    let g_idx = c_out / (C_OUT / G);
    let c_in_start = g_idx * c_in_per_group;

    // Flat loop over (C_IN/G) * KH * KW to compute partial weight gradients.
    let loop_bound = c_in_per_group * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH;
        let c_in = c_in_start + c_in_local;

        let ih = ih_base + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let load_mask = ow_mask & h_in_bounds & w_in_bounds;

        let x_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(load_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        // Partial sum for this weight element: scalar dot product over the ow tile.
        let partial = T::sum(dy_tile * x_tile, Some(0), false);
        let partial_1 = T::expand_dims(partial, 0);

        let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
        let dw_off = T::arange(0, 1) + w_idx;
        T::atomic_add(dw_ptr.add_offsets(dw_off), partial_1, None, None, None);
    }
}

/// 2-D convolution combined backward pass — computes both `dx` and `dw` in one
/// kernel launch, using the same grid as the forward pass.
///
/// Grid: `pid = ((b * C_OUT + c_out) * OH + oh) * num_ow_tiles + ow_tile`
///
/// For each `(c_in_local, kh, kw)` combination:
///   - `dx`: `atomic_add(dx_ptr[b, c_in, ih, iw], dy * w)`
///   - `dw`: `atomic_add(dw_ptr[c_out, c_in_local, kh, kw], sum(dy * x))`
///
/// Both `dx_ptr` and `dw_ptr` must be zero-initialised before launch.
#[kernel]
pub fn conv2d_backward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const G: i32,
    const BLOCK_OW: i32,
>(
    dy_ptr: T::Pointer<D>,
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    dx_ptr: T::Pointer<D>,
    dw_ptr: T::Pointer<D>,
    _B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    // Load the upstream gradient tile for this output row.
    let dy_offsets = ow_range + ((b * C_OUT + c_out) * OH * OW + oh * OW);
    let dy_tile = T::load(
        dy_ptr.add_offsets(dy_offsets),
        Some(ow_mask),
        Some(T::zeros::<D>(&[BLOCK_OW])),
        &[],
        None,
        None,
        None,
        false,
    );

    let c_in_per_group = C_IN / G;
    let g_idx = c_out / (C_OUT / G);
    let c_in_start = g_idx * c_in_per_group;

    let loop_bound = c_in_per_group * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH;
        let c_in = c_in_start + c_in_local;

        // Load weight scalar and broadcast.
        let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let in_mask = ow_mask & h_in_bounds & w_in_bounds;

        let dx_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);

        // dx: scatter dy * w to input positions.
        let grad_tile = dy_tile * w_tile;
        T::atomic_add(
            dx_ptr.add_offsets(dx_offsets),
            grad_tile,
            Some(in_mask),
            None,
            None,
        );

        // dw: load x tile, compute partial = sum(dy * x), scatter to weight.
        let x_tile = T::load(
            x_ptr.add_offsets(dx_offsets),
            Some(in_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );
        let partial = T::sum(dy_tile * x_tile, Some(0), false);
        let partial_1 = T::expand_dims(partial, 0);
        let dw_off = T::arange(0, 1) + w_idx;
        T::atomic_add(dw_ptr.add_offsets(dw_off), partial_1, None, None, None);
    }
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for Conv2dForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c_in = input_shapes[0][1];
        let c_out = output_shape[1];
        vec![vec![
            c_out,
            c_in / self.g as usize,
            self.kh as usize,
            self.kw as usize,
        ]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(input_shape[0] as i32);
        visitor.visit_i32(input_shape[1] as i32);
        visitor.visit_i32(output_shape[1] as i32);
        visitor.visit_i32(input_shape[2] as i32);
        visitor.visit_i32(input_shape[3] as i32);
        visitor.visit_i32(output_shape[2] as i32);
        visitor.visit_i32(output_shape[3] as i32);
    }

    fn block(&self) -> [u32; 3] {
        [128, 1, 1]
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32,
            1,
            1,
        ]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }

    /// kernel args: dy, x, w, dx, dw, B, C_IN, C_OUT, H, W, OH, OW
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let in_shape = inputs[0].1; // [B, C_IN, H, W]
        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(params[0]); // w_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_ptr(grad_params[0]); // dw_ptr
        visitor.visit_i32(in_shape[0] as i32); // B
        visitor.visit_i32(in_shape[1] as i32); // C_IN
        visitor.visit_i32(output_shape[1] as i32); // C_OUT
        visitor.visit_i32(in_shape[2] as i32); // H
        visitor.visit_i32(in_shape[3] as i32); // W
        visitor.visit_i32(output_shape[2] as i32); // OH
        visitor.visit_i32(output_shape[3] as i32); // OW
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] {
        [128, 1, 1]
    }

    /// Grid over forward-output positions (same formula as forward).
    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32,
            1,
            1,
        ]
    }
}

pub struct Conv2dOp<'a, T: Num> {
    pub forward: Conv2dForward<T>,
    pub backward_dx: Conv2dBackwardDx<T>,
    pub backward_dw: Conv2dBackwardDw<T>,
    _marker: core::marker::PhantomData<&'a ()>,
}

/// 2-D convolution forward pass fused with a per-output-channel bias add.
///
/// Identical to [`conv2d_forward`] (same grid, same masked-load accumulation loop —
/// see its doc comment) with one addition: `acc + bias[c_out]` before the store.
/// `bias` broadcasts the same "load as a [1] tensor, then broadcast_to" pattern
/// `conv2d_forward` already uses for weights, applied once after the loop instead
/// of once per (c_in, kh, kw) tap.
///
/// For `Conv2d(has_bias=true)` with no downstream BatchNorm/SiLU to fuse into (the
/// `conv2d_bn_silu` family), this replaces what would otherwise lower to two
/// separate kernel launches — [`conv2d_forward`] then a standalone NCHW bias-add —
/// with one. See spinorml-ia5.
///
/// Inference-only; no backward pass (training still uses the two-kernel path via
/// `conv2d_forward` + a separate bias-add, whose backward is a plain per-channel sum
/// over the output gradient — fusing that isn't this kernel's job).
#[kernel]
pub fn conv2d_bias_forward<
    T: Triton,
    D: Num,
    const KH: i32,
    const KW: i32,
    const STRIDE_H: i32,
    const STRIDE_W: i32,
    const PAD_H: i32,
    const PAD_W: i32,
    const G: i32,
    const BLOCK_OW: i32,
>(
    x_ptr: T::Pointer<D>,
    w_ptr: T::Pointer<D>,
    bias_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    _B: i32,
    C_IN: i32,
    C_OUT: i32,
    H: i32,
    W: i32,
    OH: i32,
    OW: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let num_ow_tiles = T::cdiv(OW, BLOCK_OW);

    let ow_tile = pid % num_ow_tiles;
    let bco = pid / num_ow_tiles;
    let oh = bco % OH;
    let bc = bco / OH;
    let c_out = bc % C_OUT;
    let b = bc / C_OUT;

    let ow_start = ow_tile * BLOCK_OW;
    let ow_range = T::arange(0, BLOCK_OW) + ow_start;
    let ow_mask = ow_range.lt(OW);

    let out_bc_base = (b * C_OUT + c_out) * OH * OW;

    let c_in_per_group = C_IN / G;
    let g_idx = c_out / (C_OUT / G);
    let c_in_start = g_idx * c_in_per_group;

    let mut acc = T::zeros::<D>(&[BLOCK_OW]);

    let loop_bound = c_in_per_group * KH * KW;
    for idx in 0..loop_bound {
        let kw = idx % KW;
        let kh_cin = idx / KW;
        let kh = kh_cin % KH;
        let c_in_local = kh_cin / KH;
        let c_in = c_in_start + c_in_local;

        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw_range = ow_range * STRIDE_W + kw - PAD_W;

        #[allow(clippy::erasing_op)]
        let ih_t = ow_range * 0 + ih;
        let h_in_bounds = ih_t.ge(0) & ih_t.lt(H);
        let w_in_bounds = iw_range.ge(0) & iw_range.lt(W);
        let load_mask = ow_mask & h_in_bounds & w_in_bounds;

        let x_offsets = iw_range + ((b * C_IN + c_in) * H * W + ih * W);
        let x_tile = T::load(
            x_ptr.add_offsets(x_offsets),
            Some(load_mask),
            Some(T::zeros::<D>(&[BLOCK_OW])),
            &[],
            None,
            None,
            None,
            false,
        );

        let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
        let w_off = T::arange(0, 1) + w_idx;
        let w_1 = T::load(
            w_ptr.add_offsets(w_off),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let w_tile = T::broadcast_to(w_1, &[BLOCK_OW]);

        acc = acc + x_tile * w_tile;
    }

    // ── Bias epilog ──────────────────────────────────────────────────────────
    let bias_off = T::arange(0, 1) + c_out;
    let bias_1 = T::load(
        bias_ptr.add_offsets(bias_off),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let bias_tile = T::broadcast_to(bias_1, &[BLOCK_OW]);
    acc = acc + bias_tile;

    let out_offsets = ow_range + (out_bc_base + oh * OW);
    T::store(
        y_ptr.add_offsets(out_offsets),
        acc,
        Some(ow_mask),
        &[],
        None,
        None,
    );
}

impl<D: Num + Send + Sync + 'static> teeny_core::model::RuntimeOp for Conv2dBiasForward<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c_in = input_shapes[0][1];
        let c_out = output_shape[1];
        vec![
            vec![
                c_out,
                c_in / self.g as usize,
                self.kh as usize,
                self.kw as usize,
            ],
            vec![c_out],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias"]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let input_shape = inputs[0].1;
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(params[0]); // w_ptr
        visitor.visit_ptr(params[1]); // bias_ptr
        visitor.visit_ptr(output); // y_ptr
        visitor.visit_i32(input_shape[0] as i32); // B
        visitor.visit_i32(input_shape[1] as i32); // C_IN
        visitor.visit_i32(output_shape[1] as i32); // C_OUT
        visitor.visit_i32(input_shape[2] as i32); // H
        visitor.visit_i32(input_shape[3] as i32); // W
        visitor.visit_i32(output_shape[2] as i32); // OH
        visitor.visit_i32(output_shape[3] as i32); // OW
    }

    fn block(&self) -> [u32; 3] {
        [128, 1, 1]
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let num_ow_tiles = output_shape[3].div_ceil(self.block_ow as usize);
        [
            (output_shape[0] * output_shape[1] * output_shape[2] * num_ow_tiles) as u32,
            1,
            1,
        ]
    }
}
