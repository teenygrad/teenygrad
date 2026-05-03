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

//! PSA attention helper kernels and RuntimeOp wrappers.
//!
//! Implements the data-rearrangement passes surrounding Flash Attention 2 for
//! the PSABlock used in YOLO26 C2PSA layers.
//!
//! Assumptions (derived from the ultralytics PSABlock / Attention module):
//!   - `head_dim = c / num_heads = 2 * key_dim`
//!   - QKV conv output has `qkv_h = num_heads * 4 * KEY_DIM` channels
//!     stored as NCHW `[B, qkv_h, H, W]`.
//!   - Per head `h`, channels `[h*4*KEY_DIM : +KEY_DIM]` are Q,
//!     `[+KEY_DIM : +2*KEY_DIM]` are K, `[+2*KEY_DIM : +3*KEY_DIM]` are V_lo,
//!     `[+3*KEY_DIM : +4*KEY_DIM]` are V_hi.
//!
//! The V split trick lets us run FA2 with `HEAD_DIM = key_dim` twice (once for
//! V_lo, once for V_hi) instead of needing `HEAD_DIM = head_dim = 2*key_dim`.

#![allow(non_snake_case)]

use core::ffi::c_void;

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

use super::flash_attn2::FlashAttention2Forward;

// ── psa_pack_qkv ─────────────────────────────────────────────────────────────

/// Rearranges the NCHW QKV tensor into packed FA2 format.
///
/// Input:  `qkv_ptr`  — `[B, qkv_h, H, W]` NCHW, `qkv_h = num_heads * 4 * KEY_DIM`
/// Output: `out_ptr`  — flat `[4, BH, N, KEY_DIM]` buffer
///   - Section 0: Q, Section 1: K, Section 2: V_lo, Section 3: V_hi
///
/// Grid: `[4 * BH * N, 1, 1]` — one CTA per (section, bh, n) triple.
/// Block: `[KEY_DIM, 1, 1]`
#[kernel]
pub fn psa_pack_qkv<T: Triton, const KEY_DIM: i32>(
    qkv_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    qkv_h: i32,    // num_heads * 4 * KEY_DIM
    H: i32,
    W: i32,
    B: i32,
    num_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, 4 * BH * N)
    let BH: i32 = B * num_heads;
    let N: i32 = H * W;

    let section: i32 = pid / (BH * N);
    let bh: i32 = (pid / N) % BH;
    let n: i32 = pid % N;
    let b: i32 = bh / num_heads;
    let h: i32 = bh % num_heads;

    let d = T::arange(0, KEY_DIM);

    // NCHW source: offset = (h*4*KEY_DIM + section*KEY_DIM + d) * H*W + b*qkv_h*H*W + n
    let chan_base: i32 = h * 4 * KEY_DIM + section * KEY_DIM;
    let src_off = (d + chan_base) * (H * W) + (b * qkv_h * H * W + n);

    let x = T::load(qkv_ptr.add_offsets(src_off), None, None, &[], None, None, None, false);

    // Output flat [4, BH, N, KEY_DIM]
    let dst_base: i32 = section * BH * N * KEY_DIM + bh * N * KEY_DIM + n * KEY_DIM;
    let dst_off = d + dst_base;

    T::store(out_ptr.add_offsets(dst_off), x, None, &[], None, None);
}

// ── psa_extract_v_nchw ────────────────────────────────────────────────────────

/// Extracts V channels from QKV NCHW into V NCHW.
///
/// Input:  `qkv_ptr`  — `[B, qkv_h, H, W]` NCHW
/// Output: `v_ptr`    — `[B, c, H, W]` NCHW  (c = num_heads * 2 * KEY_DIM)
///
/// Grid: `[BH * N, 1, 1]`.  Block: `[KEY_DIM, 1, 1]`.
#[kernel]
pub fn psa_extract_v_nchw<T: Triton, const KEY_DIM: i32>(
    qkv_ptr: T::Pointer<f32>,
    v_ptr: T::Pointer<f32>,
    qkv_h: i32,
    c: i32,        // num_heads * 2 * KEY_DIM
    H: i32,
    W: i32,
    num_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, BH * N)
    let N: i32 = H * W;
    let bh: i32 = pid / N;
    let n: i32 = pid % N;
    let b: i32 = bh / num_heads;
    let h: i32 = bh % num_heads;

    let d = T::arange(0, KEY_DIM);

    let src_lo_base: i32 = h * 4 * KEY_DIM + 2 * KEY_DIM;
    let src_hi_base: i32 = h * 4 * KEY_DIM + 3 * KEY_DIM;
    let src_off_lo = (d + src_lo_base) * (H * W) + (b * qkv_h * H * W + n);
    let src_off_hi = (d + src_hi_base) * (H * W) + (b * qkv_h * H * W + n);

    let x_lo = T::load(qkv_ptr.add_offsets(src_off_lo), None, None, &[], None, None, None, false);
    let x_hi = T::load(qkv_ptr.add_offsets(src_off_hi), None, None, &[], None, None, None, false);

    let dst_lo_base: i32 = h * 2 * KEY_DIM;
    let dst_hi_base: i32 = h * 2 * KEY_DIM + KEY_DIM;
    let dst_off_lo = (d + dst_lo_base) * (H * W) + (b * c * H * W + n);
    let dst_off_hi = (d + dst_hi_base) * (H * W) + (b * c * H * W + n);

    T::store(v_ptr.add_offsets(dst_off_lo), x_lo, None, &[], None, None);
    T::store(v_ptr.add_offsets(dst_off_hi), x_hi, None, &[], None, None);
}

// ── psa_merge_attn_nchw ───────────────────────────────────────────────────────

/// Merges two FA2 outputs (V_lo and V_hi attention results) into NCHW.
///
/// Inputs: `lo_ptr`, `hi_ptr` — each `[BH * N * KEY_DIM]` flat
/// Output: `out_ptr` — `[B, c, H, W]` NCHW  (c = num_heads * 2 * KEY_DIM)
///
/// Grid: `[BH * N, 1, 1]`.  Block: `[KEY_DIM, 1, 1]`.
#[kernel]
pub fn psa_merge_attn_nchw<T: Triton, const KEY_DIM: i32>(
    lo_ptr: T::Pointer<f32>,
    hi_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    c: i32,
    H: i32,
    W: i32,
    num_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, BH * N)
    let N: i32 = H * W;
    let bh: i32 = pid / N;
    let n: i32 = pid % N;
    let b: i32 = bh / num_heads;
    let h: i32 = bh % num_heads;

    let d = T::arange(0, KEY_DIM);

    // Source: flat [BH, N, KEY_DIM]
    let src_base: i32 = bh * N * KEY_DIM + n * KEY_DIM;
    let src_off = d + src_base;

    let x_lo = T::load(lo_ptr.add_offsets(src_off), None, None, &[], None, None, None, false);
    let x_hi = T::load(hi_ptr.add_offsets(src_off), None, None, &[], None, None, None, false);

    let dst_lo_base: i32 = h * 2 * KEY_DIM;
    let dst_hi_base: i32 = h * 2 * KEY_DIM + KEY_DIM;
    let dst_off_lo = (d + dst_lo_base) * (H * W) + (b * c * H * W + n);
    let dst_off_hi = (d + dst_hi_base) * (H * W) + (b * c * H * W + n);

    T::store(out_ptr.add_offsets(dst_off_lo), x_lo, None, &[], None, None);
    T::store(out_ptr.add_offsets(dst_off_hi), x_hi, None, &[], None, None);
}

// ── psa_pack_qkv_backward ─────────────────────────────────────────────────────

/// Backward of `psa_pack_qkv`: scatters `d_packed` back to `d_qkv`.
///
/// Grid / block matches the forward pass: `[4 * BH * N, 1, 1]`, block `[KEY_DIM, 1, 1]`.
///
/// Uses `atomic_add` because both `psa_pack_qkv_backward` (all four sections)
/// and `psa_extract_v_backward` (V_lo/V_hi sections) write to the same
/// `d_qkv` buffer.
#[kernel]
pub fn psa_pack_qkv_backward<T: Triton, const KEY_DIM: i32>(
    d_packed_ptr: T::Pointer<f32>,  // [4, BH, N, KEY_DIM] gradient of the packed output
    d_qkv_ptr:   T::Pointer<f32>,  // [B, qkv_h, H, W]    gradient accumulation target
    qkv_h: i32,
    H: i32,
    W: i32,
    B: i32,
    num_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, 4 * BH * N)
    let BH = B * num_heads;
    let N = H * W;
    let section = pid / (BH * N);
    let bh = (pid / N) % BH;
    let n = pid % N;
    let b = bh / num_heads;
    let h = bh % num_heads;

    let d = T::arange(0, KEY_DIM);

    // Load from d_packed: flat [4, BH, N, KEY_DIM]
    let src_base = section * BH * N * KEY_DIM + bh * N * KEY_DIM + n * KEY_DIM;
    let dx = T::load(d_packed_ptr.add_offsets(d + src_base), None, None, &[], None, None, None, false);

    // atomic_add to d_qkv at the NCHW channel position.
    let chan_base = h * 4 * KEY_DIM + section * KEY_DIM;
    let dst_off = (d + chan_base) * (H * W) + (b * qkv_h * H * W + n);
    T::atomic_add(d_qkv_ptr.add_offsets(dst_off), dx, None, None, None);
}

// ── psa_extract_v_backward ────────────────────────────────────────────────────

/// Backward of `psa_extract_v_nchw`: scatters `d_v` back to V_lo / V_hi
/// channels of `d_qkv`.
///
/// Grid / block matches the forward: `[BH * N, 1, 1]`, block `[KEY_DIM, 1, 1]`.
/// Uses `atomic_add` because V_lo / V_hi channels of `d_qkv` are also updated
/// by `psa_pack_qkv_backward`.
#[kernel]
pub fn psa_extract_v_backward<T: Triton, const KEY_DIM: i32>(
    d_v_ptr:   T::Pointer<f32>,  // [B, c, H, W]    gradient of the extracted V output
    d_qkv_ptr: T::Pointer<f32>,  // [B, qkv_h, H, W] gradient accumulation target
    qkv_h: i32,
    c: i32,
    H: i32,
    W: i32,
    num_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, BH * N)
    let N = H * W;
    let bh = pid / N;
    let n = pid % N;
    let b = bh / num_heads;
    let h = bh % num_heads;

    let d = T::arange(0, KEY_DIM);

    // Load from d_v NCHW [B, c, H, W] at V_lo and V_hi channel offsets.
    let v_lo_src_base = h * 2 * KEY_DIM;
    let v_hi_src_base = h * 2 * KEY_DIM + KEY_DIM;
    let src_off_lo = (d + v_lo_src_base) * (H * W) + (b * c * H * W + n);
    let src_off_hi = (d + v_hi_src_base) * (H * W) + (b * c * H * W + n);
    let dx_lo = T::load(d_v_ptr.add_offsets(src_off_lo), None, None, &[], None, None, None, false);
    let dx_hi = T::load(d_v_ptr.add_offsets(src_off_hi), None, None, &[], None, None, None, false);

    // atomic_add to d_qkv at the corresponding QKV channel positions (sections 2 and 3).
    let qkv_lo_base = h * 4 * KEY_DIM + 2 * KEY_DIM;
    let qkv_hi_base = h * 4 * KEY_DIM + 3 * KEY_DIM;
    let dst_off_lo = (d + qkv_lo_base) * (H * W) + (b * qkv_h * H * W + n);
    let dst_off_hi = (d + qkv_hi_base) * (H * W) + (b * qkv_h * H * W + n);
    T::atomic_add(d_qkv_ptr.add_offsets(dst_off_lo), dx_lo, None, None, None);
    T::atomic_add(d_qkv_ptr.add_offsets(dst_off_hi), dx_hi, None, None, None);
}

// ── psa_merge_attn_backward ───────────────────────────────────────────────────

/// Backward of `psa_merge_attn_nchw`: scatters `d_merged` back to `d_lo` and
/// `d_hi` flat buffers.
///
/// Grid / block matches the forward: `[BH * N, 1, 1]`, block `[KEY_DIM, 1, 1]`.
/// Regular stores are safe: each `(bh, n, d)` position maps to a unique merged
/// channel, so `d_lo` and `d_hi` receive no overlapping writes.
#[kernel]
pub fn psa_merge_attn_backward<T: Triton, const KEY_DIM: i32>(
    d_merged_ptr: T::Pointer<f32>,  // [B, c, H, W]    gradient of the merged output
    d_lo_ptr:     T::Pointer<f32>,  // [BH, N, KEY_DIM] gradient for FA2_lo output
    d_hi_ptr:     T::Pointer<f32>,  // [BH, N, KEY_DIM] gradient for FA2_hi output
    c: i32,
    H: i32,
    W: i32,
    num_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, BH * N)
    let N = H * W;
    let bh = pid / N;
    let n = pid % N;
    let b = bh / num_heads;
    let h = bh % num_heads;

    let d = T::arange(0, KEY_DIM);

    // Load lo and hi slices from d_merged NCHW.
    let src_lo_base = h * 2 * KEY_DIM;
    let src_hi_base = h * 2 * KEY_DIM + KEY_DIM;
    let src_off_lo = (d + src_lo_base) * (H * W) + (b * c * H * W + n);
    let src_off_hi = (d + src_hi_base) * (H * W) + (b * c * H * W + n);
    let dx_lo = T::load(d_merged_ptr.add_offsets(src_off_lo), None, None, &[], None, None, None, false);
    let dx_hi = T::load(d_merged_ptr.add_offsets(src_off_hi), None, None, &[], None, None, None, false);

    // Store to d_lo and d_hi flat [BH, N, KEY_DIM].
    let dst_base = bh * N * KEY_DIM + n * KEY_DIM;
    let dst_off  = d + dst_base;
    T::store(d_lo_ptr.add_offsets(dst_off), dx_lo, None, &[], None, None);
    T::store(d_hi_ptr.add_offsets(dst_off), dx_hi, None, &[], None, None);
}

// ── RuntimeOp: PsaPackQkvRuntimeOp ───────────────────────────────────────────

pub struct PsaPackQkvRuntimeOp {
    fwd: PsaPackQkv,
    bwd: PsaPackQkvBackward,
    num_heads: usize,
}

impl PsaPackQkvRuntimeOp {
    pub fn new(key_dim: i32, num_heads: usize) -> Self {
        Self { fwd: PsaPackQkv::new(key_dim), bwd: PsaPackQkvBackward::new(key_dim), num_heads }
    }

    pub fn kernel_name(&self) -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for PsaPackQkvRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // input: [B, qkv_h, H, W]
        let b = inputs[0].1[0] as i32;
        let qkv_h = inputs[0].1[1] as i32;
        let h = inputs[0].1[2] as i32;
        let w = inputs[0].1[3] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(qkv_h);
        visitor.visit_i32(h);
        visitor.visit_i32(w);
        visitor.visit_i32(b);
        visitor.visit_i32(self.num_heads as i32);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [4, BH, N, KEY_DIM]
        [(output_shape[0] * output_shape[1] * output_shape[2]) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    /// kernel args: d_packed, d_qkv, qkv_h, H, W, B, num_heads
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0].1 = [B, qkv_h, H, W]; output_shape = [4, BH, N, KEY_DIM]
        let b     = inputs[0].1[0] as i32;
        let qkv_h = inputs[0].1[1] as i32;
        let h     = inputs[0].1[2] as i32;
        let w     = inputs[0].1[3] as i32;
        let _ = output_shape;
        visitor.visit_ptr(grad_output);    // d_packed_ptr
        visitor.visit_ptr(grad_inputs[0]); // d_qkv_ptr
        visitor.visit_i32(qkv_h);
        visitor.visit_i32(h);
        visitor.visit_i32(w);
        visitor.visit_i32(b);
        visitor.visit_i32(self.num_heads as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    /// Grid = `[4 * BH * N, 1, 1]` — same layout as the forward pass.
    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * output_shape[1] * output_shape[2]) as u32, 1, 1]
    }
}

// ── RuntimeOp: PsaExtractVRuntimeOp ──────────────────────────────────────────

pub struct PsaExtractVRuntimeOp {
    fwd: PsaExtractVNchw,
    bwd: PsaExtractVBackward,
    num_heads: usize,
}

impl PsaExtractVRuntimeOp {
    pub fn new(key_dim: i32, num_heads: usize) -> Self {
        Self { fwd: PsaExtractVNchw::new(key_dim), bwd: PsaExtractVBackward::new(key_dim), num_heads }
    }

    pub fn kernel_name(&self) -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for PsaExtractVRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // input: [B, qkv_h, H, W]; output: [B, c, H, W]
        let qkv_h = inputs[0].1[1] as i32;
        let h = inputs[0].1[2] as i32;
        let w = inputs[0].1[3] as i32;
        let c = output_shape[1] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(qkv_h);
        visitor.visit_i32(c);
        visitor.visit_i32(h);
        visitor.visit_i32(w);
        visitor.visit_i32(self.num_heads as i32);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, c, H, W]; grid = [BH * N, 1, 1]
        let bh = output_shape[0] * self.num_heads;
        let n = output_shape[2] * output_shape[3];
        [(bh * n) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    /// kernel args: d_v, d_qkv, qkv_h, c, H, W, num_heads
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0].1 = [B, qkv_h, H, W]; output_shape = [B, c, H, W]
        let qkv_h = inputs[0].1[1] as i32;
        let h     = inputs[0].1[2] as i32;
        let w     = inputs[0].1[3] as i32;
        let c     = output_shape[1] as i32;
        visitor.visit_ptr(grad_output);    // d_v_ptr
        visitor.visit_ptr(grad_inputs[0]); // d_qkv_ptr
        visitor.visit_i32(qkv_h);
        visitor.visit_i32(c);
        visitor.visit_i32(h);
        visitor.visit_i32(w);
        visitor.visit_i32(self.num_heads as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    /// Grid = `[BH * N, 1, 1]` — same layout as the forward pass.
    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let bh = output_shape[0] * self.num_heads;
        let n = output_shape[2] * output_shape[3];
        [(bh * n) as u32, 1, 1]
    }
}

// ── RuntimeOp: PsaMergeAttnRuntimeOp ─────────────────────────────────────────

pub struct PsaMergeAttnRuntimeOp {
    fwd: PsaMergeAttnNchw,
    bwd: PsaMergeAttnBackward,
    num_heads: usize,
}

impl PsaMergeAttnRuntimeOp {
    pub fn new(key_dim: i32, num_heads: usize) -> Self {
        Self { fwd: PsaMergeAttnNchw::new(key_dim), bwd: PsaMergeAttnBackward::new(key_dim), num_heads }
    }

    pub fn kernel_name(&self) -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for PsaMergeAttnRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 2 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0] = o_lo [BH, N, KEY_DIM], inputs[1] = o_hi [BH, N, KEY_DIM]
        // output_shape = [B, c, H, W]
        let c = output_shape[1] as i32;
        let h = output_shape[2] as i32;
        let w = output_shape[3] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(c);
        visitor.visit_i32(h);
        visitor.visit_i32(w);
        visitor.visit_i32(self.num_heads as i32);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, c, H, W]; grid = [BH * N, 1, 1]
        let bh = output_shape[0] * self.num_heads;
        let n = output_shape[2] * output_shape[3];
        [(bh * n) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    /// kernel args: d_merged, d_lo, d_hi, c, H, W, num_heads
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        _inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // output_shape = [B, c, H, W]; grad_inputs = [d_lo, d_hi]
        let c = output_shape[1] as i32;
        let h = output_shape[2] as i32;
        let w = output_shape[3] as i32;
        visitor.visit_ptr(grad_output);    // d_merged_ptr
        visitor.visit_ptr(grad_inputs[0]); // d_lo_ptr
        visitor.visit_ptr(grad_inputs[1]); // d_hi_ptr
        visitor.visit_i32(c);
        visitor.visit_i32(h);
        visitor.visit_i32(w);
        visitor.visit_i32(self.num_heads as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    /// Grid = `[BH * N, 1, 1]` — same layout as the forward pass.
    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let bh = output_shape[0] * self.num_heads;
        let n = output_shape[2] * output_shape[3];
        [(bh * n) as u32, 1, 1]
    }
}

// ── psa_fa2_backward ─────────────────────────────────────────────────────────

/// Combined PSA Flash Attention 2 backward pass.
///
/// Each CTA handles one `(n, bh)` pair and computes both dQ_n (by iterating
/// over all K rows) and dK_n / dV_n (by iterating over all Q rows).
/// This is valid in PSA because `N_CTX_Q == N_CTX_K == N` (self-attention).
///
/// **Atomicity**: `dq_ptr` and `dk_ptr` are written with `atomic_add` because
/// both FA2_lo and FA2_hi backward passes contribute to shared sections 0 and 1
/// of the d_packed buffer.  `dv_ptr` uses a regular store (each FA2 call owns
/// an exclusive V section so no overlap exists).
///
/// Grid: `(N, BH, 1)` — same shape as the FA2 forward pass.
#[kernel]
pub fn psa_fa2_backward<T: Triton, const HEAD_DIM: i32>(
    q_ptr:  T::Pointer<f32>,  // [BH, N, HEAD_DIM] Q — section 0 of packed forward buffer
    k_ptr:  T::Pointer<f32>,  // [BH, N, HEAD_DIM] K — section 1 of packed forward buffer
    v_ptr:  T::Pointer<f32>,  // [BH, N, HEAD_DIM] V — section v_section of packed forward buffer
    o_ptr:  T::Pointer<f32>,  // [BH, N, HEAD_DIM] FA2 forward output
    do_ptr: T::Pointer<f32>,  // [BH, N, HEAD_DIM] upstream gradient
    l_ptr:  T::Pointer<f32>,  // [BH * N]          logsumexp saved from forward
    dq_ptr: T::Pointer<f32>,  // [BH, N, HEAD_DIM] atomic_add target for dQ
    dk_ptr: T::Pointer<f32>,  // [BH, N, HEAD_DIM] atomic_add target for dK
    dv_ptr: T::Pointer<f32>,  // [BH, N, HEAD_DIM] store target for dV
    N: i32,                   // N_CTX (== N_CTX_Q == N_CTX_K in PSA self-attention)
    softmax_scale: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_n  = T::program_id(Axis::X); // spatial token [0, N)
    let pid_bh = T::program_id(Axis::Y); // (batch, head)  [0, BH)

    let row_base = pid_bh * N * HEAD_DIM + pid_n * HEAD_DIM;
    let l_base   = pid_bh * N + pid_n;
    let bh_base  = pid_bh * N * HEAD_DIM;
    let l_bh     = pid_bh * N;

    let d       = T::arange(0, HEAD_DIM);
    let scale_t = T::full::<f32>(&[HEAD_DIM], softmax_scale);

    // Load this row's Q, O, dO and compute D_n = rowsum(O * dO).
    let q_vec  = T::load(q_ptr.add_offsets(d + row_base),  None, None, &[], None, None, None, false);
    let o_vec  = T::load(o_ptr.add_offsets(d + row_base),  None, None, &[], None, None, None, false);
    let do_vec = T::load(do_ptr.add_offsets(d + row_base), None, None, &[], None, None, None, false);

    let d_n = T::sum(o_vec * do_vec, Some(0), false); // scalar

    let l_n_raw = T::load(l_ptr.add_offsets(T::arange(0, 1) + l_base), None, None, &[], None, None, None, false);
    let l_n     = T::sum(l_n_raw, Some(0), false); // scalar

    // Phase 1: accumulate dQ_n by iterating over all K rows.
    let mut dq_acc = T::zeros::<f32>(&[HEAD_DIM]);
    for k_row in 0..N {
        let kv_row_base  = bh_base + k_row * HEAD_DIM;
        let k_vec        = T::load(k_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);
        let v_vec        = T::load(v_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);
        let qk           = T::sum(q_vec * k_vec, Some(0), false) * scale_t;
        let p            = T::exp(qk - l_n);
        let do_dot_v     = T::sum(do_vec * v_vec, Some(0), false);
        let ds           = p * (do_dot_v - d_n);
        dq_acc = dq_acc + ds * k_vec * scale_t;
    }
    T::atomic_add(dq_ptr.add_offsets(d + row_base), dq_acc, None, None, None);

    // Phase 2: accumulate dK_n and dV_n by iterating over all Q rows.
    let k_vec_n = T::load(k_ptr.add_offsets(d + row_base), None, None, &[], None, None, None, false);
    let v_vec_n = T::load(v_ptr.add_offsets(d + row_base), None, None, &[], None, None, None, false);
    let mut dk_acc = T::zeros::<f32>(&[HEAD_DIM]);
    let mut dv_acc = T::zeros::<f32>(&[HEAD_DIM]);
    for q_row in 0..N {
        let q_row_base_m  = bh_base + q_row * HEAD_DIM;
        let l_row_base_m  = l_bh + q_row;
        let q_vec_m  = T::load(q_ptr.add_offsets(d + q_row_base_m),  None, None, &[], None, None, None, false);
        let o_vec_m  = T::load(o_ptr.add_offsets(d + q_row_base_m),  None, None, &[], None, None, None, false);
        let do_vec_m = T::load(do_ptr.add_offsets(d + q_row_base_m), None, None, &[], None, None, None, false);
        let l_m_raw  = T::load(l_ptr.add_offsets(T::arange(0, 1) + l_row_base_m), None, None, &[], None, None, None, false);
        let l_m      = T::sum(l_m_raw, Some(0), false);
        let d_m      = T::sum(o_vec_m * do_vec_m, Some(0), false);
        let qk       = T::sum(q_vec_m * k_vec_n, Some(0), false) * scale_t;
        let p        = T::exp(qk - l_m);
        dv_acc = dv_acc + p * do_vec_m;
        let do_dot_v_m = T::sum(do_vec_m * v_vec_n, Some(0), false);
        let ds_m = p * (do_dot_v_m - d_m);
        dk_acc = dk_acc + ds_m * q_vec_m * scale_t;
    }
    T::atomic_add(dk_ptr.add_offsets(d + row_base), dk_acc, None, None, None);
    T::store(dv_ptr.add_offsets(d + row_base), dv_acc, None, &[], None, None);
}

// ── RuntimeOp: FlashAttn2PsaRuntimeOp ────────────────────────────────────────

/// RuntimeOp wrapping `flash_attention2_forward` for PSA attention.
///
/// Input:  packed QKV buffer, shape `[4, BH, N, KEY_DIM]`
/// Output: attention result, shape `[BH, N, KEY_DIM]`
/// Params: `[BH * N]` scratch for FA2 logsumexp `l_ptr`.
pub struct FlashAttn2PsaRuntimeOp {
    fwd: FlashAttention2Forward,
    bwd: PsaFa2Backward,
    v_section: usize,
}

impl FlashAttn2PsaRuntimeOp {
    /// Attention on V_lo (section index 2).
    pub fn new_lo(key_dim: i32) -> Self {
        Self { fwd: FlashAttention2Forward::new(key_dim), bwd: PsaFa2Backward::new(key_dim), v_section: 2 }
    }

    /// Attention on V_hi (section index 3).
    pub fn new_hi(key_dim: i32) -> Self {
        Self { fwd: FlashAttention2Forward::new(key_dim), bwd: PsaFa2Backward::new(key_dim), v_section: 3 }
    }

    pub fn kernel_name(&self) -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for FlashAttn2PsaRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, input_shapes: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        // input_shapes[0] = [4, BH, N, KEY_DIM]
        let bh = input_shapes[0][1];
        let n = input_shapes[0][2];
        vec![vec![bh * n]] // l_ptr scratch
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let bh = inputs[0].1[1];
        let n = inputs[0].1[2];
        let kd = inputs[0].1[3];
        let section_elems = bh * n * kd;

        let base = inputs[0].0 as *mut f32;
        let q_ptr = base as *mut c_void;
        let k_ptr = unsafe { base.add(section_elems) } as *mut c_void;
        let v_ptr = unsafe { base.add(self.v_section * section_elems) } as *mut c_void;
        let softmax_scale = 1.0_f32 / (kd as f32).sqrt();

        visitor.visit_ptr(q_ptr);
        visitor.visit_ptr(k_ptr);
        visitor.visit_ptr(v_ptr);
        visitor.visit_ptr(output);
        visitor.visit_ptr(params[0]);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(softmax_scale);
        visitor.visit_f32(f32::NEG_INFINITY);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [BH, N, KEY_DIM]; FA2 grid = (N, BH, 1)
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    /// kernel args: q, k, v, o, do, l, dq, dk, dv, N, softmax_scale
    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let bh = inputs[0].1[1];
        let n  = inputs[0].1[2];
        let kd = inputs[0].1[3];
        let section_elems = bh * n * kd;
        let softmax_scale = 1.0_f32 / (kd as f32).sqrt();

        // Forward Q, K, V pointers from the packed input buffer.
        let fwd_base = inputs[0].0 as *mut f32;
        let q_ptr = fwd_base as *mut c_void;
        let k_ptr = unsafe { fwd_base.add(section_elems) } as *mut c_void;
        let v_ptr = unsafe { fwd_base.add(self.v_section * section_elems) } as *mut c_void;

        // Gradient pointers into d_packed (same layout as packed input).
        let d_base = grad_inputs[0] as *mut f32;
        let dq_ptr = d_base as *mut c_void;
        let dk_ptr = unsafe { d_base.add(section_elems) } as *mut c_void;
        let dv_ptr = unsafe { d_base.add(self.v_section * section_elems) } as *mut c_void;

        visitor.visit_ptr(q_ptr);
        visitor.visit_ptr(k_ptr);
        visitor.visit_ptr(v_ptr);
        visitor.visit_ptr(output);       // o_ptr
        visitor.visit_ptr(grad_output);  // do_ptr
        visitor.visit_ptr(params[0]);    // l_ptr
        visitor.visit_ptr(dq_ptr);
        visitor.visit_ptr(dk_ptr);
        visitor.visit_ptr(dv_ptr);
        visitor.visit_i32(n as i32);     // N
        visitor.visit_f32(softmax_scale);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    /// Grid over `(N, BH, 1)` — same shape as the forward pass.
    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }
}
