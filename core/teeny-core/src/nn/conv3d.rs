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

/// 3-D convolution layer: `output = conv3d(input, weight) [+ b]`
///
/// Input shape:  `[N, C_in,  D_in,  H_in,  W_in ]`
/// Output shape: `[N, C_out, D_out, H_out, W_out]`
pub struct Conv3d<D: Dtype, IT, OT, const RANK: usize> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_d: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_d: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_d: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub has_bias: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> Conv3d<D, IT, OT, RANK> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        has_bias: bool,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_d: kernel_size.0,
            kernel_h: kernel_size.1,
            kernel_w: kernel_size.2,
            stride_d: stride.0,
            stride_h: stride.1,
            stride_w: stride.2,
            padding_d: padding.0,
            padding_h: padding.1,
            padding_w: padding.2,
            has_bias,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
    Layer<IT> for Conv3d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
