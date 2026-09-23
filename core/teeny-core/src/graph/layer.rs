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

//! `Layer` implementations for [`SymTensor`].
//!
//! Each one records its op on the graph instead of computing anything: the
//! symbolic counterpart to the eager implementations in `crate::nn`. They are
//! fallible because shape inference is — a conv window that cannot fit is
//! reported here rather than at execution time.

use super::{Op, SymTensor};
use crate::errors::Result;
use crate::{
    dtype::{Dtype, Float},
    nn::{
        Layer,
        activation::{
            elu::{Celu, Elu, Selu},
            gelu::{Gelu, Mish},
            hard::{Hardshrink, Hardsigmoid, Hardswish, Hardtanh, Relu6},
            misc::{LeakyRelu, Softplus, Softshrink, Softsign, Threshold},
            relu::Relu,
            sigmoid::{LogSigmoid, Sigmoid, Silu},
            softmax::Softmax,
            tanh::{Tanh, Tanhshrink},
        },
        batchnorm::{BatchNorm1d, BatchNorm2d, BatchNorm3d},
        conv1d::Conv1d,
        conv2d::Conv2d,
        conv3d::Conv3d,
        flatten::Flatten,
        groupnorm::GroupNorm,
        instancenorm::{InstanceNorm1d, InstanceNorm2d, InstanceNorm3d},
        layernorm::LayerNorm,
        linear::Linear,
        pad::{
            CircularPad1d, CircularPad2d, CircularPad3d, ConstantPad1d, ConstantPad2d,
            ConstantPad3d, ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d,
            ReplicationPad2d, ReplicationPad3d,
        },
        pool::{
            AvgPool1d, AvgPool2d, AvgPool3d, LpPool1d, LpPool2d, LpPool3d, MaxPool1d, MaxPool2d,
            MaxPool3d,
        },
        rmsnorm::RmsNorm,
    },
};

// --- Linear / MLP ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Linear<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Linear {
            in_features: self.in_features,
            out_features: self.out_features,
            has_bias: self.has_bias,
        })
    }
}

impl<D: Dtype> Layer<SymTensor> for Flatten<D, SymTensor, SymTensor> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Flatten)
    }
}

// --- Normalisation ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for BatchNorm1d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::BatchNorm1d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for BatchNorm2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::BatchNorm2d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for BatchNorm3d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::BatchNorm3d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for LayerNorm<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::LayerNorm {
            normalized_shape: self.normalized_shape.clone(),
            eps: self.eps,
            affine: self.affine,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for RmsNorm<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::RmsNorm {
            normalized_shape: self.normalized_shape.clone(),
            eps: self.eps,
            affine: self.affine,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for GroupNorm<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::GroupNorm {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            eps: self.eps,
            affine: self.affine,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for InstanceNorm1d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::InstanceNorm1d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for InstanceNorm2d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::InstanceNorm2d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for InstanceNorm3d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::InstanceNorm3d {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            track_running_stats: self.track_running_stats,
        })
    }
}

// --- Convolution ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Conv1d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Conv1d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_l: self.kernel_l,
            stride: self.stride,
            padding: self.padding,
            has_bias: self.has_bias,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Conv2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Conv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            padding_h: self.padding_h,
            padding_w: self.padding_w,
            groups: self.groups,
            has_bias: self.has_bias,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Conv3d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Conv3d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_d: self.kernel_d,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_d: self.stride_d,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            padding_d: self.padding_d,
            padding_h: self.padding_h,
            padding_w: self.padding_w,
            has_bias: self.has_bias,
        })
    }
}

// --- Pooling ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for AvgPool1d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::AvgPool1d {
            kernel_l: self.kernel_l,
            stride: self.stride,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for AvgPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::AvgPool2d {
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for AvgPool3d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::AvgPool3d {
            kernel_d: self.kernel_d,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_d: self.stride_d,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for MaxPool1d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::MaxPool1d {
            kernel_l: self.kernel_l,
            stride: self.stride,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for MaxPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::MaxPool2d {
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            pad_h: self.padding_h,
            pad_w: self.padding_w,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for MaxPool3d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::MaxPool3d {
            kernel_d: self.kernel_d,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_d: self.stride_d,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for LpPool1d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::LpPool1d {
            kernel_l: self.kernel_l,
            stride: self.stride,
            p: self.p,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for LpPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::LpPool2d {
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            p: self.p,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for LpPool3d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::LpPool3d {
            kernel_d: self.kernel_d,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_d: self.stride_d,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            p: self.p,
        })
    }
}

// --- Padding ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ConstantPad1d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ConstantPad1d {
            pad_left: self.pad_left,
            pad_right: self.pad_right,
            value: self.value,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ConstantPad2d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ConstantPad2d {
            pad_l: self.pad_l,
            pad_r: self.pad_r,
            pad_t: self.pad_t,
            pad_b: self.pad_b,
            value: self.value,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ConstantPad3d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ConstantPad3d {
            pad_d1: self.pad_d1,
            pad_d2: self.pad_d2,
            pad_h1: self.pad_h1,
            pad_h2: self.pad_h2,
            pad_w1: self.pad_w1,
            pad_w2: self.pad_w2,
            value: self.value,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ReflectionPad1d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ReflectionPad1d {
            pad_left: self.pad_left,
            pad_right: self.pad_right,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ReflectionPad2d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ReflectionPad2d {
            pad_l: self.pad_l,
            pad_r: self.pad_r,
            pad_t: self.pad_t,
            pad_b: self.pad_b,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ReflectionPad3d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ReflectionPad3d {
            pad_d1: self.pad_d1,
            pad_d2: self.pad_d2,
            pad_h1: self.pad_h1,
            pad_h2: self.pad_h2,
            pad_w1: self.pad_w1,
            pad_w2: self.pad_w2,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ReplicationPad1d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ReplicationPad1d {
            pad_left: self.pad_left,
            pad_right: self.pad_right,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ReplicationPad2d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ReplicationPad2d {
            pad_l: self.pad_l,
            pad_r: self.pad_r,
            pad_t: self.pad_t,
            pad_b: self.pad_b,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for ReplicationPad3d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::ReplicationPad3d {
            pad_d1: self.pad_d1,
            pad_d2: self.pad_d2,
            pad_h1: self.pad_h1,
            pad_h2: self.pad_h2,
            pad_w1: self.pad_w1,
            pad_w2: self.pad_w2,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for CircularPad1d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::CircularPad1d {
            pad_left: self.pad_left,
            pad_right: self.pad_right,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for CircularPad2d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::CircularPad2d {
            pad_l: self.pad_l,
            pad_r: self.pad_r,
            pad_t: self.pad_t,
            pad_b: self.pad_b,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor>
    for CircularPad3d<D, SymTensor, SymTensor, RANK>
{
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::CircularPad3d {
            pad_d1: self.pad_d1,
            pad_d2: self.pad_d2,
            pad_h1: self.pad_h1,
            pad_h2: self.pad_h2,
            pad_w1: self.pad_w1,
            pad_w2: self.pad_w2,
        })
    }
}

// --- Activation ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Relu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Relu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Elu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Elu { alpha: self.alpha })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Selu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Selu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Celu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Celu { alpha: self.alpha })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Gelu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Gelu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Mish<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Mish)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardtanh<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Hardtanh {
            min_val: self.min_val,
            max_val: self.max_val,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Relu6<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Relu6)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardsigmoid<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Hardsigmoid)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardswish<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Hardswish)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardshrink<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Hardshrink {
            lambda: self.lambda,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for LeakyRelu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::LeakyRelu {
            negative_slope: self.negative_slope,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Threshold<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Threshold {
            threshold: self.threshold,
            value: self.value,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softsign<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Softsign)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softshrink<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Softshrink {
            lambda: self.lambda,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softplus<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Softplus {
            beta: self.beta,
            threshold: self.threshold,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Sigmoid<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Sigmoid)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Silu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Silu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for LogSigmoid<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::LogSigmoid)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Tanh<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Tanh)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Tanhshrink<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Tanhshrink)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softmax<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> Result<SymTensor> {
        input.record(Op::Softmax { dim: self.dim })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    use crate::graph::DtypeRepr;
    use crate::sequential;

    #[test]
    fn test_sequential_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);

        let model = sequential![
            Linear::<f32, SymTensor, SymTensor, 2>::new(784, 128, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(128, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input).unwrap();

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 5);
        assert!(matches!(g.nodes[0].op, Op::Input));
        assert_eq!(g.nodes[0].shape, vec![None, Some(784)]);

        assert!(matches!(
            g.nodes[1].op,
            Op::Linear {
                in_features: 784,
                out_features: 128,
                ..
            }
        ));
        assert_eq!(g.nodes[1].shape, vec![None, Some(128)]);

        assert!(matches!(g.nodes[2].op, Op::Relu));
        assert_eq!(g.nodes[2].shape, vec![None, Some(128)]);

        assert!(matches!(
            g.nodes[3].op,
            Op::Linear {
                in_features: 128,
                out_features: 10,
                ..
            }
        ));
        assert_eq!(g.nodes[3].shape, vec![None, Some(10)]);

        assert!(matches!(g.nodes[4].op, Op::Softmax { dim: 1 }));
        assert_eq!(g.nodes[4].shape, vec![None, Some(10)]);
    }

    #[test]
    fn test_conv2d_graph_extraction() {
        let (input, graph) =
            SymTensor::input(DtypeRepr::F32, vec![None, Some(3), Some(32), Some(32)]);

        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(3, 64, (3, 3), (1, 1), (1, 1), true);
        let _out = Layer::call(&conv, input).unwrap();

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 2);
        assert!(matches!(
            g.nodes[1].op,
            Op::Conv2d {
                in_channels: 3,
                out_channels: 64,
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                padding_h: 1,
                padding_w: 1,
                has_bias: true,
                ..
            }
        ));
        assert_eq!(g.nodes[1].shape, vec![None, Some(64), Some(32), Some(32)]);
    }

    /// A kernel wider than its padded input used to underflow `usize` inside
    /// `infer_output_shape`, surfacing as a bare "attempt to subtract with
    /// overflow" in debug and a wrapped, enormous extent in release.
    #[test]
    fn test_conv2d_kernel_larger_than_padded_input_errors_with_context() {
        let (input, _graph) =
            SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(4), Some(4)]);
        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(3, 8, (7, 7), (1, 1), (1, 1), false);
        let err = Layer::call(&conv, input)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(
            err.contains("Conv2d: height kernel 7 does not fit its input"),
            "unexpected error: {err}"
        );
    }

    /// The message names the extents involved and both ways out, rather than
    /// just the failing line.
    #[test]
    fn test_conv2d_window_error_reports_padded_extent_and_remedies() {
        let (input, _graph) =
            SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(4), Some(4)]);
        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(3, 8, (7, 7), (1, 1), (1, 1), false);
        let err = Layer::call(&conv, input)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(
            err.contains("widens it to only 6, so no window position is valid"),
            "unexpected error: {err}"
        );
    }

    /// Exactly-fitting windows are still legal: kernel == padded extent gives
    /// a single window, so the guard must not be off by one.
    #[test]
    fn test_conv2d_kernel_exactly_filling_padded_input_is_allowed() {
        let (input, graph) =
            SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(4), Some(4)]);
        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(3, 8, (6, 6), (1, 1), (1, 1), false);
        Layer::call(&conv, input).unwrap();

        let g = graph.borrow();
        assert_eq!(g.nodes[1].shape, vec![Some(1), Some(8), Some(1), Some(1)]);
    }

    /// A zero stride divided by zero one line below the subtraction; it now
    /// says which axis and what to set it to.
    #[test]
    fn test_conv2d_zero_stride_errors_with_context() {
        let (input, _graph) =
            SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(8), Some(8)]);
        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(3, 8, (3, 3), (1, 0), (0, 0), false);
        let err = Layer::call(&conv, input)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(
            err.contains("Conv2d: width stride is 0"),
            "unexpected error: {err}"
        );
    }

    /// Pooling shares the same guard, and the combined `AvgPool*`/`MaxPool*`
    /// match arms must still name the op the caller actually used.
    #[test]
    fn test_maxpool2d_window_error_names_the_right_op() {
        let (input, _graph) =
            SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(3), Some(4), Some(4)]);
        let pool = MaxPool2d::<f32, SymTensor, SymTensor, 4>::new((5, 5), (1, 1));
        let err = Layer::call(&pool, input)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(
            err.contains("MaxPool2d: height kernel 5 does not fit its input"),
            "unexpected error: {err}"
        );
    }
}
