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

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── BCELoss ───────────────────────────────────────────────────────────────────

/// Element-wise binary cross-entropy forward.
///
/// Assumes `input` is already a probability (sigmoid output), i.e. in `(0, 1)`.
///
/// ```text
/// out = -(target * log(input) + (1 - target) * log(1 - input))
/// ```
///
/// Grid: `[ceil(n / BLOCK_SIZE), 1, 1]`, block `[128, 1, 1]`.
#[kernel]
pub fn bce_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    // Clamp to (eps, 1-eps) to avoid log(0)
    let eps = T::full(&[BLOCK_SIZE], 1e-7_f32);
    let one_minus_eps = T::full(&[BLOCK_SIZE], 1.0_f32 - 1e-7_f32);
    let inp_c = T::clamp(inp, eps, one_minus_eps);

    let loss = T::full(&[BLOCK_SIZE], -1.0_f32) * (tgt * T::log(inp_c) + (one - tgt) * T::log(one - inp_c));
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise BCE backward.
///
/// ```text
/// dx = -(target / input - (1 - target) / (1 - input))
/// ```
#[kernel]
pub fn bce_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    let eps = T::full(&[BLOCK_SIZE], 1e-7_f32);
    let one_minus_eps = T::full(&[BLOCK_SIZE], 1.0_f32 - 1e-7_f32);
    let inp_c = T::clamp(inp, eps, one_minus_eps);

    // dx = -(t/x - (1-t)/(1-x))
    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    let dx_raw = neg_one * (tgt / inp_c - (one - tgt) / (one - inp_c));
    let dx = dx_raw * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── BCEWithLogitsLoss ─────────────────────────────────────────────────────────

/// Element-wise BCE-with-logits forward.
///
/// Numerically stable implementation:
/// ```text
/// out = max(x, 0) - x*t + log(1 + exp(-|x|))
/// ```
#[kernel]
pub fn bce_with_logits_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    // Numerically stable: max(x,0) - x*t + log(1+exp(-|x|))
    let relu_x = T::maximum(inp, zeros);
    let neg_abs_x = T::full(&[BLOCK_SIZE], -1.0_f32) * T::abs(inp);
    let loss = relu_x - inp * tgt + T::log(one + T::exp(neg_abs_x));
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise BCE-with-logits backward.
///
/// `dx = sigmoid(x) - target`
#[kernel]
pub fn bce_with_logits_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    // sigmoid(x) = 1 / (1 + exp(-x))
    let sig = one / (one + T::exp(neg_one * inp));
    let dx = (sig - tgt) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── SoftMarginLoss ────────────────────────────────────────────────────────────

/// Element-wise soft margin loss forward.
///
/// ```text
/// out = log(1 + exp(-target * input))
/// ```
///
/// Numerically stable via: `log(1 + exp(-t*x)) = max(-t*x, 0) + log(1 + exp(-|t*x|))`
#[kernel]
pub fn soft_margin_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    // log(1 + exp(-t*x)) — numerically stable softplus of (-t*x)
    let neg_tx = T::full(&[BLOCK_SIZE], -1.0_f32) * tgt * inp;
    let loss = T::maximum(neg_tx, zeros) + T::log(one + T::exp(T::full(&[BLOCK_SIZE], -1.0_f32) * T::abs(tgt * inp)));
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise soft margin loss backward.
///
/// ```text
/// dx = -target * sigmoid(-target * input)
/// ```
#[kernel]
pub fn soft_margin_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    let one = T::full(&[BLOCK_SIZE], 1.0_f32);
    // sigmoid(-t*x) = 1 / (1 + exp(t*x))
    let neg_tx = neg_one * tgt * inp;
    let sig_neg_tx = one / (one + T::exp(neg_one * neg_tx));
    // dx = -t * sigmoid(-t*x) * dy
    let dx = neg_one * tgt * sig_neg_tx * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── KLDivLoss ─────────────────────────────────────────────────────────────────

/// Element-wise KL-divergence forward.
///
/// PyTorch convention: `input` is log-probability, `target` is probability.
///
/// ```text
/// out = target * (log(target) - input)
/// ```
///
/// Masked: `out = 0` where `target <= 0`.
#[kernel]
pub fn kl_div_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    // out = target * (log(target) - input), masked to 0 where target <= 0
    let eps = T::full(&[BLOCK_SIZE], 1e-10_f32);
    let tgt_safe = T::maximum(tgt, eps);
    let loss_raw = tgt * (T::log(tgt_safe) - inp);
    let positive = T::gt(tgt, zeros);
    let loss = T::where_(positive, loss_raw, zeros);
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise KL-divergence backward w.r.t. log-input.
///
/// `dx = -target`
#[kernel]
pub fn kl_div_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let neg_one = T::full(&[BLOCK_SIZE], -1.0_f32);
    let dx = neg_one * tgt * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── PoissonNLLLoss ────────────────────────────────────────────────────────────

/// Element-wise Poisson NLL loss forward with `log_input=True` (default).
///
/// ```text
/// out = exp(input) - target * input
/// ```
#[kernel]
pub fn poisson_nll_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    // loss = exp(input) - target * input  (log_input mode)
    let loss = T::exp(inp) - tgt * inp;
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise Poisson NLL loss backward (log_input=True).
///
/// `dx = (exp(input) - target) * dy`
#[kernel]
pub fn poisson_nll_loss_backward<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let dx = (T::exp(inp) - tgt) * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}

// ── GaussianNLLLoss ───────────────────────────────────────────────────────────

/// Element-wise Gaussian NLL loss forward.
///
/// ```text
/// out = 0.5 * (log(var) + (input - target)^2 / var)
/// ```
///
/// `var` is clamped to `eps_var` from below for numerical stability.
#[kernel]
pub fn gaussian_nll_loss_forward<T: Triton, const BLOCK_SIZE: i32>(
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    var_ptr: T::Pointer<f32>,
    out_ptr: T::Pointer<f32>,
    n_elements: i32,
    eps_var: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let var = T::load(var_ptr.add_offsets(offsets),    Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let eps_t = T::full(&[BLOCK_SIZE], eps_var);
    let half  = T::full(&[BLOCK_SIZE], 0.5_f32);
    let var_c = T::maximum(var, eps_t);
    let diff  = inp - tgt;
    let loss  = half * (T::log(var_c) + diff * diff / var_c);
    T::store(out_ptr.add_offsets(offsets), loss, Some(in_bounds), &[], None, None);
}

/// Element-wise Gaussian NLL backward w.r.t. input (mean prediction).
///
/// `dx = (input - target) / var * dy`
#[kernel]
pub fn gaussian_nll_loss_backward_input<T: Triton, const BLOCK_SIZE: i32>(
    dy_ptr: T::Pointer<f32>,
    input_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    var_ptr: T::Pointer<f32>,
    dx_ptr: T::Pointer<f32>,
    n_elements: i32,
    eps_var: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let in_bounds = offsets.lt(n_elements);
    let zeros = T::zeros::<f32>(&[BLOCK_SIZE]);

    let dy  = T::load(dy_ptr.add_offsets(offsets),     Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let inp = T::load(input_ptr.add_offsets(offsets),  Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let tgt = T::load(target_ptr.add_offsets(offsets), Some(in_bounds), Some(zeros), &[], None, None, None, false);
    let var = T::load(var_ptr.add_offsets(offsets),    Some(in_bounds), Some(zeros), &[], None, None, None, false);

    let eps_t = T::full(&[BLOCK_SIZE], eps_var);
    let var_c = T::maximum(var, eps_t);
    let dx = (inp - tgt) / var_c * dy;
    T::store(dx_ptr.add_offsets(offsets), dx, Some(in_bounds), &[], None, None);
}
