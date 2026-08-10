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

//! BatchNorm1d Triton kernels.
//!
//! Layout: input `x` is `[N, C]` row-major. Element `x[n, c]` lives at flat
//! offset `n * C + c`.
//!
//! Parallelism: **one CTA per channel**. Each CTA iterates over all N batch
//! elements in `BLOCK_N`-wide tiles. This avoids cross-CTA synchronisation
//! entirely — C channels execute concurrently across SMs.
//!
//! Training requires two sequential kernel launches separated by a host sync:
//!   1. `batch_norm_stats_forward`   — computes per-channel mean + rstd, updates
//!      running stats.
//!   2. `batch_norm_normalize_forward` — normalises x using the saved stats.
//!
//! Inference uses a single kernel that reads the frozen running statistics.

#![allow(non_snake_case)]

use teeny_core::dtype::Float;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ─── Inference: single kernel, frozen running statistics ─────────────────────

/// Normalises input `x` using the frozen `running_mean` / `running_var`.
///
/// Grid: `[C]` — one CTA per channel.
#[kernel]
pub fn batch_norm_forward_inference<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    running_mean_ptr: InPtr<T::Pointer<D>>,
    running_var_ptr: InPtr<T::Pointer<D>>,
    N: i32,
    C: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars (shape [1]) and broadcast to [BLOCK_N].
    let mean = T::broadcast_to(
        T::load(
            running_mean_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let var = T::load(
        running_var_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let rstd = T::broadcast_to(
        T::rsqrt(var + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false)),
        &[BLOCK_N],
    );
    let gamma = T::broadcast_to(
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let beta = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );

    // Normalise all N elements for this channel.
    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let mut n_start: i32 = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let y_tile = gamma * (x_tile - mean) * rstd + beta;

        T::store(
            y_ptr.add_offsets(elem_offsets),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );

        n_start += BLOCK_N;
    }
}

// ─── Training: kernel 1 — compute per-channel statistics ─────────────────────

/// Computes per-channel mean and rstd from the current mini-batch, saves them
/// for the normalisation kernel and the backward pass, and updates the running
/// statistics with exponential moving average.
///
/// Grid: `[C]` — one CTA per channel.
///
/// **Must complete (host sync) before `batch_norm_normalize_forward` is launched.**
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_stats_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    mean_ptr: OutPtr<T::Pointer<D>>,
    rstd_ptr: OutPtr<T::Pointer<D>>,
    running_mean_ptr: InOutPtr<T::Pointer<D>>,
    running_var_ptr: InOutPtr<T::Pointer<D>>,
    N: i32,
    C: i32,
    eps: f32,
    momentum: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);

    // Accumulate sum(x) and sum(x²) over all N elements for this channel.
    // Triton idiom: accumulate BLOCK_N-wide tiles inside the loop, reduce once
    // outside — tt.reduce inside a loop body is not supported by Triton's lowering.
    let zeros_blk = T::zeros::<D>(&[BLOCK_N]);
    let mut acc_sum = zeros_blk;
    let mut acc_sum_sq = zeros_blk;
    let mut n_start: i32 = 0;

    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros_blk),
            &[],
            None,
            None,
            None,
            false,
        );
        acc_sum = acc_sum + x_tile;
        acc_sum_sq = acc_sum_sq + x_tile * x_tile;

        n_start += BLOCK_N;
    }

    // Single reduce outside the loop — shape [BLOCK_N] → [1].
    let sum = T::sum(acc_sum, None, true);
    let sum_sq = T::sum(acc_sum_sq, None, true);

    // Derive mean, biased variance, and rstd (all shape [1]).
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false);
    let mean_1 = sum * n_inv;
    let var_1 = sum_sq * n_inv - mean_1 * mean_1;
    let rstd_1 = T::rsqrt(var_1 + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false));

    // Save for the normalisation and backward kernels.
    let c_idx = T::arange(0, 1) + c;
    T::store(mean_ptr.add_offsets(c_idx), mean_1, None, &[], None, None);
    T::store(rstd_ptr.add_offsets(c_idx), rstd_1, None, &[], None, None);

    // Exponential moving average: running = (1 - m) * running + m * batch.
    let m = T::cast::<f32, D>(T::full::<f32>(&[1], momentum), None, false);
    let one_m = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 - momentum), None, false);
    let running_mean_old = T::load(
        running_mean_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let running_var_old = T::load(
        running_var_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    T::store(
        running_mean_ptr.add_offsets(c_idx),
        one_m * running_mean_old + m * mean_1,
        None,
        &[],
        None,
        None,
    );
    T::store(
        running_var_ptr.add_offsets(c_idx),
        one_m * running_var_old + m * var_1,
        None,
        &[],
        None,
        None,
    );
}

// ─── Training: kernel 2 — normalise using saved statistics ───────────────────

/// Normalises x using the mean and rstd produced by `batch_norm_stats_forward`.
///
/// Grid: `[C]` — one CTA per channel.
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_normalize_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    mean_ptr: InPtr<T::Pointer<D>>,
    rstd_ptr: InPtr<T::Pointer<D>>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars and broadcast to [BLOCK_N].
    let mean = T::broadcast_to(
        T::load(
            mean_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let rstd = T::broadcast_to(
        T::load(
            rstd_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let gamma = T::broadcast_to(
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let beta = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let mut n_start: i32 = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let y_tile = gamma * (x_tile - mean) * rstd + beta;

        T::store(
            y_ptr.add_offsets(elem_offsets),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );

        n_start += BLOCK_N;
    }
}

// ─── Training RuntimeOp implementations ──────────────────────────────────────

/// RuntimeOp for the stats kernel node in a training BatchNorm graph.
///
/// Stores `eps` and `momentum` (not in the macro-generated struct) and handles
/// the packed `[2*C]` output layout: first C elements = mean, last C = rstd.
#[cfg(feature = "training")]
pub struct BatchNormStatsRuntimeOp<D: teeny_core::dtype::Float + Send + Sync + 'static> {
    pub block_n: i32,
    pub eps: f32,
    pub momentum: f32,
    _phantom: core::marker::PhantomData<D>,
}

#[cfg(feature = "training")]
impl<D: teeny_core::dtype::Float + Send + Sync + 'static> BatchNormStatsRuntimeOp<D> {
    pub fn new(block_n: i32, eps: f32, momentum: f32) -> Self {
        Self {
            block_n,
            eps,
            momentum,
            _phantom: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "training")]
impl<D: teeny_core::dtype::Float + Send + Sync + 'static> teeny_core::model::RuntimeOp
    for BatchNormStatsRuntimeOp<D>
{
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c = input_shapes[0][1];
        vec![vec![c], vec![c]]
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
        let c = output_shape[0] / 2;
        let n_total: usize = inputs[0].1.iter().product();
        let n = (n_total / c) as i32;
        let mean_ptr = output;
        let rstd_ptr = unsafe { (output as *mut D).add(c) } as teeny_core::model::RawPtr;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(mean_ptr);
        visitor.visit_ptr(rstd_ptr);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_i32(n);
        visitor.visit_i32(c as i32);
        visitor.visit_f32(self.eps);
        visitor.visit_f32(self.momentum);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let c = output_shape[0] / 2;
        [c as u32, 1, 1]
    }
}

/// RuntimeOp for the normalize kernel node in a training BatchNorm graph.
///
/// Expects two activation inputs: `inputs[0]` = x, `inputs[1]` = packed stats
/// `[2*C]` from the stats node (first C = mean, last C = rstd).
#[cfg(feature = "training")]
pub struct BatchNormNormalizeRuntimeOp<D: teeny_core::dtype::Float + Send + Sync + 'static> {
    pub block_n: i32,
    bwd_source: String,
    _phantom: core::marker::PhantomData<D>,
}

#[cfg(feature = "training")]
impl<D: teeny_core::dtype::Float + Send + Sync + 'static> BatchNormNormalizeRuntimeOp<D> {
    pub fn new(block_n: i32) -> Self {
        Self {
            block_n,
            bwd_source: BatchNormBackward::<D>::new(block_n).source,
            _phantom: core::marker::PhantomData,
        }
    }

    pub fn backward_source(&self) -> &str {
        &self.bwd_source
    }
}

#[cfg(feature = "training")]
impl<D: teeny_core::dtype::Float + Send + Sync + 'static> teeny_core::model::RuntimeOp
    for BatchNormNormalizeRuntimeOp<D>
{
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c = input_shapes[1][0] / 2;
        vec![vec![c], vec![c]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias"]
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
        let c = inputs[1].1[0] / 2;
        let n_total: usize = inputs[0].1.iter().product();
        let n = (n_total / c) as i32;
        let mean_ptr = inputs[1].0;
        let rstd_ptr = unsafe { (inputs[1].0 as *mut D).add(c) } as teeny_core::model::RawPtr;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(mean_ptr);
        visitor.visit_ptr(rstd_ptr);
        visitor.visit_i32(n);
        visitor.visit_i32(c as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let c = output_shape.get(1).copied().unwrap_or(output_shape[0]);
        [c as u32, 1, 1]
    }

    fn has_backward(&self) -> bool {
        true
    }

    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // kernel args: dy, x, dx, weight, mean, rstd, dweight, dbias, N, C
        // inputs[0] = x,     inputs[1] = stats [2*C] (mean at [0..C], rstd at [C..2C])
        // params[0] = weight, params[1] = bias (unused for dx)
        // grad_inputs[0] = dx,  grad_params[0] = dweight,  grad_params[1] = dbias
        let c = inputs[1].1[0] / 2;
        let n_total: usize = inputs[0].1.iter().product();
        let n = (n_total / c) as i32;
        let mean_ptr = inputs[1].0;
        let rstd_ptr = unsafe { (inputs[1].0 as *mut D).add(c) } as teeny_core::model::RawPtr;
        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_ptr(params[0]); // weight_ptr
        visitor.visit_ptr(mean_ptr); // mean_ptr
        visitor.visit_ptr(rstd_ptr); // rstd_ptr
        visitor.visit_ptr(grad_params[0]); // dweight_ptr
        visitor.visit_ptr(grad_params[1]); // dbias_ptr
        visitor.visit_i32(n);
        visitor.visit_i32(c as i32);
    }

    fn backward_grid(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> [u32; 3] {
        // input_shapes[1] = [2*C] (stats); one CTA per channel
        let c = input_shapes[1][0] / 2;
        [c as u32, 1, 1]
    }
}

// ─── Inference (NCHW): NCHW-native BatchNorm2d ───────────────────────────────

/// Normalises NCHW input `x` using frozen running statistics.
///
/// Input layout: [B, C, H, W] row-major. Element `x[b, c, h, w]` lives at
/// offset `b*C*HW + c*HW + h*W + w`.
///
/// Grid: `[C, B]` — one CTA per (channel, batch) pair; each CTA iterates over
/// H*W spatial positions in `BLOCK_HW`-wide tiles.
#[kernel]
pub fn batch_norm_2d_nchw_forward_inference<T: Triton, D: Float, const BLOCK_HW: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: InPtr<T::Pointer<D>>,
    running_mean_ptr: InPtr<T::Pointer<D>>,
    running_var_ptr: InPtr<T::Pointer<D>>,
    C: i32,
    HW: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let b = T::program_id(Axis::Y);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars and broadcast to [BLOCK_HW].
    let mean = T::broadcast_to(
        T::load(
            running_mean_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_HW],
    );
    let var = T::load(
        running_var_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let rstd = T::broadcast_to(
        T::rsqrt(var + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false)),
        &[BLOCK_HW],
    );
    let gamma = T::broadcast_to(
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_HW],
    );
    let beta = T::broadcast_to(
        T::load(
            bias_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_HW],
    );

    // Flat start offset for (b, c, hw=0) in NCHW: b*C*HW + c*HW
    let batch_channel_offset: i32 = b * C * HW + c * HW;
    let zeros = T::zeros::<D>(&[BLOCK_HW]);
    let mut hw_start: i32 = 0;
    while hw_start < HW {
        let offsets = T::arange(0, BLOCK_HW) + hw_start;
        let mask = offsets.lt(HW);
        let elem_offsets = offsets + batch_channel_offset;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let y_tile = gamma * (x_tile - mean) * rstd + beta;
        T::store(
            y_ptr.add_offsets(elem_offsets),
            y_tile,
            Some(mask),
            &[],
            None,
            None,
        );

        hw_start += BLOCK_HW;
    }
}

// ─── Inference (NCHW) RuntimeOp ──────────────────────────────────────────────

/// RuntimeOp for NCHW BatchNorm2d inference.
///
/// Parameter layout (4 params): `[weight, bias, running_mean, running_var]`,
/// each of shape `[C]`.
pub struct BatchNorm2dNchwInferenceRuntimeOp<D: Float + Send + Sync + 'static> {
    fwd: BatchNorm2dNchwForwardInference<D>,
    block_hw: i32,
    eps: f32,
}

impl<D: Float + Send + Sync + 'static> BatchNorm2dNchwInferenceRuntimeOp<D> {
    pub fn new(block_hw: i32, eps: f32) -> Self {
        Self {
            fwd: BatchNorm2dNchwForwardInference::<D>::new(block_hw),
            block_hw,
            eps,
        }
    }

    pub fn forward_source(&self) -> &str {
        &self.fwd.source
    }
    pub fn kernel_name(&self) -> &str {
        self.fwd.name
    }

    #[cfg(feature = "training")]
    pub fn backward_source(&self) -> String {
        BatchNorm2dNchwBackward::<D>::new(self.block_hw).source
    }
}

impl<D: Float + Send + Sync + 'static> teeny_core::model::RuntimeOp
    for BatchNorm2dNchwInferenceRuntimeOp<D>
{
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        let c = input_shapes[0][1];
        vec![vec![c], vec![c], vec![c], vec![c]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "bias", "running_mean", "running_var"]
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
        let c = output_shape[1] as i32;
        let hw = (output_shape[2] * output_shape[3]) as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_ptr(params[0]); // weight
        visitor.visit_ptr(params[1]); // bias
        visitor.visit_ptr(params[2]); // running_mean
        visitor.visit_ptr(params[3]); // running_var
        visitor.visit_i32(c);
        visitor.visit_i32(hw);
        visitor.visit_f32(self.eps);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool {
        true
    }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // kernel args: dy, x, dx, weight, running_mean, running_var, dweight, dbias, B, C, HW, eps
        let in_shape = inputs[0].1; // [B, C, H, W]
        let b = in_shape[0] as i32;
        let c = in_shape[1] as i32;
        let hw = (in_shape[2] * in_shape[3]) as i32;
        visitor.visit_ptr(grad_output); // dy_ptr
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_ptr(params[0]); // weight_ptr
        visitor.visit_ptr(params[2]); // running_mean_ptr
        visitor.visit_ptr(params[3]); // running_var_ptr
        visitor.visit_ptr(grad_params[0]); // dweight_ptr
        visitor.visit_ptr(grad_params[1]); // dbias_ptr
        visitor.visit_i32(b);
        visitor.visit_i32(c);
        visitor.visit_i32(hw);
        visitor.visit_f32(self.eps);
    }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _output_shape: &[usize]) -> [u32; 3] {
        // input_shapes[0] = [B, C, H, W]; one CTA per channel
        [input_shapes[0][1] as u32, 1, 1]
    }
}

// ─── Training (NCHW): backward pass ──────────────────────────────────────────

/// Computes gradients for NCHW BatchNorm2d inference.
///
/// Since `running_mean` and `running_var` are frozen constants, the backward of
/// `y = gamma * (x - mean) * rstd + beta` (with respect to `x`) is simply:
/// ```text
/// dx[b,c,h,w]   = gamma[c] * rstd[c] * dy[b,c,h,w]
/// dweight[c]     = Σ_{b,h,w} dy * xhat
/// dbias[c]       = Σ_{b,h,w} dy
/// ```
///
/// A single loop over (b, hw) computes all three simultaneously.
///
/// Grid: `[C]` — one CTA per channel.
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_2d_nchw_backward<T: Triton, D: Float, const BLOCK_HW: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    running_mean_ptr: InPtr<T::Pointer<D>>,
    running_var_ptr: InPtr<T::Pointer<D>>,
    dweight_ptr: OutPtr<T::Pointer<D>>,
    dbias_ptr: OutPtr<T::Pointer<D>>,
    B: i32,
    C: i32,
    HW: i32,
    eps: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars as [1]-shaped tensors to match element-wise loop.
    let mean = T::load(
        running_mean_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let var = T::load(
        running_var_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let rstd = T::rsqrt(var + T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false));
    let gamma = T::load(
        weight_ptr.add_offsets(c_idx),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );

    // Use a single flat loop over B*HW with one element per iteration.
    // Scalar b_idx/hw_idx arithmetic avoids tensor-level division/modulo,
    // and a single-level loop with [1] accumulators is correctly lowered
    // (same pattern as batch_norm_stats_forward).
    let mut sum_dy = T::zeros::<D>(&[1]);
    let mut sum_dy_xhat = T::zeros::<D>(&[1]);
    let total_bhw = B * HW;
    let mut n: i32 = 0;
    while n < total_bhw {
        let b_idx = n / HW; // scalar division — always valid
        let hw_idx = n % HW; // scalar modulo
        let offset: i32 = b_idx * C * HW + c * HW + hw_idx;
        let off_1 = T::arange(0, 1) + offset; // [1] pointing at this element

        let x_elem = T::load(
            x_ptr.add_offsets(off_1),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_elem = T::load(
            dy_ptr.add_offsets(off_1),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );

        let xhat = (x_elem - mean) * rstd;
        sum_dy = sum_dy + dy_elem;
        sum_dy_xhat = sum_dy_xhat + dy_elem * xhat;

        // dx = gamma * rstd * dy (frozen-stats: mean/rstd are constants)
        let dx_elem = gamma * rstd * dy_elem;
        T::store(dx_ptr.add_offsets(off_1), dx_elem, None, &[], None, None);

        n += 1;
    }

    T::store(
        dweight_ptr.add_offsets(c_idx),
        sum_dy_xhat,
        None,
        &[],
        None,
        None,
    );
    T::store(dbias_ptr.add_offsets(c_idx), sum_dy, None, &[], None, None);
}

// ─── Training (NC): backward pass ─────────────────────────────────────────────

/// Computes gradients for BatchNorm.
///
/// Given saved `mean` and `rstd` from the forward pass:
/// ```text
/// xhat      = (x - mean) * rstd
/// dbias[c]  = Σ_n dy[n,c]
/// dweight[c]= Σ_n dy[n,c] * xhat[n,c]
/// dx[n,c]   = weight[c] * rstd[c] * (dy[n,c]
///               - dbias[c] / N
///               - xhat[n,c] * dweight[c] / N)
/// ```
///
/// Uses two sequential passes over N within the same CTA to avoid storing
/// the full xhat tensor.
///
/// Grid: `[C]` — one CTA per channel.
#[cfg(feature = "training")]
#[kernel]
pub fn batch_norm_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr: InPtr<T::Pointer<D>>,
    x_ptr: InPtr<T::Pointer<D>>,
    dx_ptr: OutPtr<T::Pointer<D>>,
    weight_ptr: InPtr<T::Pointer<D>>,
    mean_ptr: InPtr<T::Pointer<D>>,
    rstd_ptr: InPtr<T::Pointer<D>>,
    dweight_ptr: OutPtr<T::Pointer<D>>,
    dbias_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::program_id(Axis::X);
    let c_idx = T::arange(0, 1) + c;

    // Load per-channel scalars; broadcast to [BLOCK_N] for element-wise ops.
    let mean = T::broadcast_to(
        T::load(
            mean_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let rstd = T::broadcast_to(
        T::load(
            rstd_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );
    let weight = T::broadcast_to(
        T::load(
            weight_ptr.add_offsets(c_idx),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ),
        &[BLOCK_N],
    );

    // Pass 1: accumulate dbias (= Σ dy) and dweight (= Σ dy * xhat).
    // Triton idiom: accumulate BLOCK_N-wide tiles inside the loop, reduce once
    // outside — tt.reduce inside a loop body is not supported by Triton's lowering.
    let zeros_blk = T::zeros::<D>(&[BLOCK_N]);
    let mut acc_dy = zeros_blk;
    let mut acc_dy_xhat = zeros_blk;
    let mut n_start: i32 = 0;

    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros_blk),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros_blk),
            &[],
            None,
            None,
            None,
            false,
        );
        let xhat = (x_tile - mean) * rstd;

        acc_dy = acc_dy + dy_tile;
        acc_dy_xhat = acc_dy_xhat + dy_tile * xhat;

        n_start += BLOCK_N;
    }

    // Single reduce outside the loop — shape [BLOCK_N] → [1].
    let sum_dy = T::sum(acc_dy, None, true);
    let sum_dy_xhat = T::sum(acc_dy_xhat, None, true);

    // Save dweight and dbias (shape [1] → stored as scalars).
    T::store(
        dweight_ptr.add_offsets(c_idx),
        sum_dy_xhat,
        None,
        &[],
        None,
        None,
    );
    T::store(dbias_ptr.add_offsets(c_idx), sum_dy, None, &[], None, None);

    // Broadcast reduction results and 1/N for pass 2.
    let n_inv = T::broadcast_to(
        T::cast::<f32, D>(T::full::<f32>(&[1], 1.0f32 / (N as f32)), None, false),
        &[BLOCK_N],
    );
    let sum_dy_bcast = T::broadcast_to(sum_dy, &[BLOCK_N]);
    let sum_dy_xhat_bcast = T::broadcast_to(sum_dy_xhat, &[BLOCK_N]);

    // Pass 2: compute dx.
    n_start = 0;
    while n_start < N {
        let offsets_n = T::arange(0, BLOCK_N) + n_start;
        let mask = offsets_n.lt(N);
        let elem_offsets = offsets_n * C + c;

        let x_tile = T::load(
            x_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros_blk),
            &[],
            None,
            None,
            None,
            false,
        );
        let dy_tile = T::load(
            dy_ptr.add_offsets(elem_offsets),
            Some(mask),
            Some(zeros_blk),
            &[],
            None,
            None,
            None,
            false,
        );
        let xhat = (x_tile - mean) * rstd;

        let dx_tile =
            weight * rstd * (dy_tile - sum_dy_bcast * n_inv - xhat * sum_dy_xhat_bcast * n_inv);

        T::store(
            dx_ptr.add_offsets(elem_offsets),
            dx_tile,
            Some(mask),
            &[],
            None,
            None,
        );

        n_start += BLOCK_N;
    }
}
