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

use core::marker::PhantomData;

use crate::{
    dtype::{Dtype, EagerTensor, Tensor},
    nn::Layer,
};

/// 2-D average pooling layer.
///
/// Input shape:  `[N, C, H_in,  W_in ]`
/// Output shape: `[N, C, H_out, W_out]`
///
/// where:
///   `H_out = (H_in - kernel_h) / stride_h + 1`
///   `W_out = (W_in - kernel_w) / stride_w + 1`
///
/// Type parameters:
/// - `D`    — element dtype
/// - `IT`   — input tensor type  (rank 4)
/// - `OT`   — output tensor type (rank 4)
/// - `RANK` — tensor rank; must be 4 for a valid 2-D pooling operation
///
/// Tensor bounds are on impls, not the struct, so `SymTensor` can have its
/// own `Layer` impl without a coherence conflict.
pub struct AvgPool2d<D: Dtype, IT, OT, const RANK: usize> {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> AvgPool2d<D, IT, OT, RANK> {
    /// Create a new AvgPool2d layer.
    ///
    /// - `kernel_size` — `(height, width)` of the pooling window
    /// - `stride`      — `(height, width)` step between pooling windows
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: stride.0,
            stride_w: stride.1,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
    Layer<IT> for AvgPool2d<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
