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

//! Unit tests for `#[kernel]`'s `fusion_core()` codegen (teenygrad-3w0.9):
//! the splice-ready "core" extracted from single-input, single-axis
//! `#[tile(...)]` kernels whose body ends in a plain `T::store(...)` call.
//!
//! No CUDA / kernel compilation involved — this only checks the
//! macro-generated `fusion_core()` associated function produces the
//! declared metadata.

use teeny_kernels::nn::activation::elu::{EluBackward, EluForward, SeluForward};
use teeny_kernels::nn::activation::relu::ReluForward;
use teeny_kernels::nn::activation::sigmoid::SigmoidForward;
use teeny_kernels::nn::activation::tanh::TanhForward;
use teeny_kernels::nn::conv::conv2d::Conv2dForward;
use teeny_kernels::nn::tensor::elemwise_unary::ElemwiseExpForward;

#[test]
fn elu_forward_fusion_core_captures_body_minus_trailing_store() {
    let core = EluForward::<f32>::fusion_core()
        .expect("elu_forward is single-input, single-axis, trailing-store");
    assert_eq!(core.input_ident, "x");
    assert_eq!(core.output_ident, "y");
    assert!(!core.body_source.contains("store"));
    assert!(core.body_source.contains("where_"));
    // alpha: f32 is elu_forward's one extra scalar param.
    assert_eq!(core.extra_params, &[("alpha", "f32")]);
}

#[test]
fn relu_forward_fusion_core_has_no_extra_params() {
    let core = ReluForward::<f32>::fusion_core()
        .expect("relu_forward is single-input, single-axis, trailing-store");
    assert_eq!(core.input_ident, "x");
    assert_eq!(core.output_ident, "relu");
    assert!(core.extra_params.is_empty());
}

#[test]
fn sigmoid_tanh_exp_fusion_core_resolve() {
    assert!(SigmoidForward::<f32>::fusion_core().is_some());
    assert!(TanhForward::<f32>::fusion_core().is_some());
    assert!(ElemwiseExpForward::<f32>::fusion_core().is_some());
}

#[test]
fn selu_forward_fusion_core_resolves_with_no_extra_params() {
    let core = SeluForward::<f32>::fusion_core()
        .expect("selu_forward is single-input, single-axis, trailing-store");
    assert_eq!(core.input_ident, "x");
    assert!(core.extra_params.is_empty());
}

#[test]
fn elu_backward_fusion_core_is_none_multi_input() {
    // Two #[tile(...)]-tagged InPtr params (dy_ptr, x_ptr) sharing the same
    // axis -- fusion_core is scoped to single-input chains only.
    assert!(EluBackward::<f32>::fusion_core().is_none());
}

#[test]
fn conv2d_fusion_core_is_none_multi_axis() {
    // conv2d_forward's tags are multi-axis / prelude = false -- no shared
    // single-axis prelude group, so fusion_core must be None.
    assert!(Conv2dForward::<f32>::fusion_core().is_none());
}
