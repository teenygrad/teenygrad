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

/// 1-D convolution layer: `output = conv1d(input, weight) [+ b]`
///
/// Input shape:  `[N, C_in,  L_in ]`
/// Output shape: `[N, C_out, L_out]`
///
/// where: `L_out = (L_in + 2*padding - kernel_l) / stride + 1`
pub struct Conv1d<D: Dtype, IT, OT, const RANK: usize> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_l: usize,
    pub stride: usize,
    pub padding: usize,
    pub has_bias: bool,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> Conv1d<D, IT, OT, RANK> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_l: kernel_size,
            stride,
            padding,
            has_bias,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
    Layer<IT> for Conv1d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
