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

/// 2-D convolution layer: `output = conv2d(input, weight) [+ b]`
///
/// Input shape:  `[N, C_in,  H_in,  W_in ]`
/// Output shape: `[N, C_out, H_out, W_out]`
///
/// where:
///   `H_out = (H_in + 2*padding_h - kernel_h) / stride_h + 1`
///   `W_out = (W_in + 2*padding_w - kernel_w) / stride_w + 1`
///
/// Type parameters:
/// - `D`    — element dtype
/// - `IT`   — input tensor type  (rank 4: `[N, C_in,  H, W]`)
/// - `OT`   — output tensor type (rank 4: `[N, C_out, H, W]`)
/// - `RANK` — tensor rank; must be 4 for a valid 2-D convolution
///
/// Tensor bounds are on impls, not the struct, so `SymTensor` can have its
/// own `Layer` impl without a coherence conflict.
pub struct Conv2d<D: Dtype, IT, OT, const RANK: usize> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub groups: usize,
    pub has_bias: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> Conv2d<D, IT, OT, RANK> {
    /// Create a new Conv2d layer.
    ///
    /// - `kernel_size` — `(height, width)` of the convolution kernel
    /// - `stride`      — `(height, width)` step between kernel applications
    /// - `padding`     — `(height, width)` zero-padding added to each spatial side
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        has_bias: bool,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: stride.0,
            stride_w: stride.1,
            padding_h: padding.0,
            padding_w: padding.1,
            groups: 1,
            has_bias,
            _pd: PhantomData,
        }
    }

    pub fn new_grouped(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        has_bias: bool,
        groups: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: stride.0,
            stride_w: stride.1,
            padding_h: padding.0,
            padding_w: padding.1,
            groups,
            has_bias,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
    Layer<IT> for Conv2d<D, IT, OT, RANK>
{
    type Output = OT;

    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
