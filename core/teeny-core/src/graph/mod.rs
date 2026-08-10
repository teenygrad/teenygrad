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

use alloc::{collections::BTreeMap, rc::Rc, string::String, sync::Arc, vec, vec::Vec};
use core::{any::Any, cell::RefCell};

use crate::{
    dtype::{Dtype, Float, RankedTensor, Tensor},
    nn::{
        Layer,
        activation::{
            elu::{Celu, Elu, Selu},
            gelu::{Gelu, Mish},
            hard::{Hardshrink, Hardsigmoid, Hardswish, Hardtanh, Relu6},
            misc::{LeakyRelu, Softplus, Softshrink, Softsign, Threshold},
            relu::Relu,
            sigmoid::{Logsigmoid, Sigmoid, Silu},
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

/// Graph-to-FXGraph lowering.
pub mod compiler;

// ---------------------------------------------------------------------------
// Shape — dynamic tensor shape used throughout the graph IR
// ---------------------------------------------------------------------------

/// A dynamic shape vector. Each element is either a known size (`Some(n)`) or a
/// dynamic/unknown dimension (`None`), e.g. a batch axis whose size is determined
/// at runtime.
pub type Shape = Vec<Option<usize>>;

// ---------------------------------------------------------------------------
// Runtime dtype tag — used in the graph since D is erased at the node level
// ---------------------------------------------------------------------------

/// Runtime dtype tag: the graph-level (type-erased) representation of a tensor's dtype,
/// mirroring `dtype::Dtype`'s implementors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DtypeRepr {
    /// `bool`.
    Bool,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// 16-bit float.
    F16,
    /// `bfloat16`.
    BF16,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

// ---------------------------------------------------------------------------
// Graph IR
// ---------------------------------------------------------------------------

/// Trait implemented by user-defined ops.
pub trait CustomOp: Any + Send + Sync {
    /// Identifier used in error messages and debug output.
    fn name(&self) -> &str;

    /// Compute the output shape given the shapes of all input tensors in order.
    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape;

    /// Expose `self` as `&dyn Any` so the custom lowering can downcast to the
    /// concrete op type.  Implement as `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn Any;

    /// Return kernel lowering info so `TritonLowering` can compile this op
    /// without a project-specific middleware.  Return `None` to keep the
    /// existing middleware / error behaviour.
    ///
    /// Tuple layout: `(name, kernel_source, entry_point_name, runtime_op)`.
    /// `entry_point_name` is the PTX symbol name, conventionally `"{name}_entry_point"`.
    fn lower(&self) -> Option<(String, String, String, Arc<dyn crate::model::RuntimeOp>)> {
        None
    }

    /// Return the backward kernel source for this op (used in training mode).
    /// Return an empty string if this op has no backward pass.
    fn lower_backward_source(&self) -> String {
        String::new()
    }
}

/// Wrapper around `Arc<dyn CustomOp>` that implements `Debug` for [`Op`].
#[derive(Clone)]
pub struct CustomData(pub Arc<dyn CustomOp>);

impl CustomData {
    /// Wraps a [`CustomOp`] implementation.
    pub fn new<T: CustomOp>(op: T) -> Self {
        Self(Arc::new(op))
    }

    /// The wrapped op's name.
    pub fn name(&self) -> &str {
        self.0.name()
    }

    /// The wrapped op's inferred output shape.
    pub fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        self.0.infer_output_shape(input_shapes)
    }

    /// Downcast to a concrete op type via [`CustomOp::as_any`].
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref::<T>()
    }
}

impl core::fmt::Debug for CustomData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Custom({})", self.0.name())
    }
}

/// A single computational-graph operation. Each variant corresponds to one `nn` layer or
/// primitive op; variant fields are that op's configuration (mirroring the corresponding
/// `nn::*` layer struct's fields).
#[derive(Debug, Clone)]
pub enum Op {
    /// Model input placeholder.
    Input,

    // --- Linear / MLP ---
    /// Fully-connected layer (see `nn::linear::Linear`).
    Linear {
        /// Size of the last input dimension.
        in_features: usize,
        /// Size of the last output dimension.
        out_features: usize,
        /// Whether a learned bias is added.
        has_bias: bool,
    },
    /// Flattens all spatial dimensions into a single feature vector (see `nn::flatten::Flatten`).
    Flatten,

    // --- Normalisation ---
    /// 1-D batch normalization (see `nn::batchnorm::BatchNorm1d`).
    BatchNorm1d {
        /// Number of channels/features.
        num_features: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Running-stats exponential moving average weight.
        momentum: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
        /// Whether to maintain running mean/variance across batches.
        track_running_stats: bool,
    },
    /// 2-D batch normalization (see `nn::batchnorm::BatchNorm2d`).
    BatchNorm2d {
        /// Number of channels/features.
        num_features: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Running-stats exponential moving average weight.
        momentum: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
        /// Whether to maintain running mean/variance across batches.
        track_running_stats: bool,
    },
    /// 3-D batch normalization (see `nn::batchnorm::BatchNorm3d`).
    BatchNorm3d {
        /// Number of channels/features.
        num_features: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Running-stats exponential moving average weight.
        momentum: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
        /// Whether to maintain running mean/variance across batches.
        track_running_stats: bool,
    },
    /// Layer normalization (see `nn::layernorm::LayerNorm`).
    LayerNorm {
        /// Shape of the trailing axes to normalize over.
        normalized_shape: alloc::vec::Vec<usize>,
        /// Numerical stability constant.
        eps: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
    },
    /// RMS normalization (see `nn::rmsnorm::RmsNorm`).
    RmsNorm {
        /// Shape of the trailing axes to normalize over.
        normalized_shape: alloc::vec::Vec<usize>,
        /// Numerical stability constant.
        eps: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
    },
    /// Group normalization (see `nn::groupnorm::GroupNorm`).
    GroupNorm {
        /// Number of groups.
        num_groups: usize,
        /// Number of channels.
        num_channels: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
    },
    /// 1-D instance normalization (see `nn::instancenorm::InstanceNorm1d`).
    InstanceNorm1d {
        /// Number of channels/features.
        num_features: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Running-stats exponential moving average weight.
        momentum: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
        /// Whether to maintain running mean/variance across batches.
        track_running_stats: bool,
    },
    /// 2-D instance normalization (see `nn::instancenorm::InstanceNorm2d`).
    InstanceNorm2d {
        /// Number of channels/features.
        num_features: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Running-stats exponential moving average weight.
        momentum: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
        /// Whether to maintain running mean/variance across batches.
        track_running_stats: bool,
    },
    /// 3-D instance normalization (see `nn::instancenorm::InstanceNorm3d`).
    InstanceNorm3d {
        /// Number of channels/features.
        num_features: usize,
        /// Numerical stability constant.
        eps: f64,
        /// Running-stats exponential moving average weight.
        momentum: f64,
        /// Whether to learn per-channel scale/shift parameters.
        affine: bool,
        /// Whether to maintain running mean/variance across batches.
        track_running_stats: bool,
    },

    // --- Convolution ---
    /// 1-D convolution (see `nn::conv1d::Conv1d`).
    Conv1d {
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution/pooling kernel length.
        kernel_l: usize,
        /// Stride between kernel applications.
        stride: usize,
        /// Zero-padding applied to the input.
        padding: usize,
        /// Whether a learned bias is added.
        has_bias: bool,
    },
    /// 2-D convolution (see `nn::conv2d::Conv2d`).
    Conv2d {
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// Vertical zero-padding.
        padding_h: usize,
        /// Horizontal zero-padding.
        padding_w: usize,
        /// Number of blocked/grouped connections (1 = standard).
        groups: usize,
        /// Whether a learned bias is added.
        has_bias: bool,
    },
    /// 3-D convolution (see `nn::conv3d::Conv3d`).
    Conv3d {
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution/pooling kernel depth.
        kernel_d: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Stride along the depth dimension.
        stride_d: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// Zero-padding along the depth dimension.
        padding_d: usize,
        /// Vertical zero-padding.
        padding_h: usize,
        /// Horizontal zero-padding.
        padding_w: usize,
        /// Whether a learned bias is added.
        has_bias: bool,
    },

    /// Fused Conv2d + BatchNorm2d (inference-only) + SiLU forward.
    ///
    /// The BN parameters (scale, shift) are stored as precomputed affine
    /// constants — not raw mean/var/gamma/beta.  Produced by the Anduin graph
    /// optimizer in `teeny-kernels` when it detects
    /// `Conv2d(no bias) → BatchNorm2d → Silu`.
    ///
    /// `bn_eps` is carried forward only for reference; the BN affine constants
    /// are passed at runtime via the `bn_scale` and `bn_shift` parameters.
    Conv2dBnSilu {
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// Vertical zero-padding.
        padding_h: usize,
        /// Horizontal zero-padding.
        padding_w: usize,
        /// Number of blocked/grouped connections (1 = standard).
        groups: usize,
        /// The fused BatchNorm's numerical stability constant (kept for reference only; the
        /// affine constants are passed at runtime via `bn_scale`/`bn_shift`).
        bn_eps: f64,
    },

    // --- Pooling ---
    /// 1-D average pooling (see `nn::pool::AvgPool1d`).
    AvgPool1d {
        /// Convolution/pooling kernel length.
        kernel_l: usize,
        /// Stride between kernel applications.
        stride: usize,
    },
    /// 2-D average pooling (see `nn::pool::AvgPool2d`).
    AvgPool2d {
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
    },
    /// 3-D average pooling (see `nn::pool::AvgPool3d`).
    AvgPool3d {
        /// Convolution/pooling kernel depth.
        kernel_d: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Stride along the depth dimension.
        stride_d: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
    },
    /// 1-D max pooling (see `nn::pool::MaxPool1d`).
    MaxPool1d {
        /// Convolution/pooling kernel length.
        kernel_l: usize,
        /// Stride between kernel applications.
        stride: usize,
    },
    /// 2-D max pooling (see `nn::pool::MaxPool2d`).
    MaxPool2d {
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// Vertical padding.
        pad_h: usize,
        /// Horizontal padding.
        pad_w: usize,
    },
    /// 3-D max pooling (see `nn::pool::MaxPool3d`).
    MaxPool3d {
        /// Convolution/pooling kernel depth.
        kernel_d: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Stride along the depth dimension.
        stride_d: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
    },
    /// 1-D power-average (Lp) pooling (see `nn::pool::LpPool1d`).
    LpPool1d {
        /// Convolution/pooling kernel length.
        kernel_l: usize,
        /// Stride between kernel applications.
        stride: usize,
        /// The `p` in the p-norm.
        p: f64,
    },
    /// 2-D power-average (Lp) pooling (see `nn::pool::LpPool2d`).
    LpPool2d {
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// The `p` in the p-norm.
        p: f64,
    },
    /// 3-D power-average (Lp) pooling (see `nn::pool::LpPool3d`).
    LpPool3d {
        /// Convolution/pooling kernel depth.
        kernel_d: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Stride along the depth dimension.
        stride_d: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// The `p` in the p-norm.
        p: f64,
    },

    // --- Upsample ---
    /// Nearest-neighbour 2-D upsampling.
    /// Output shape: `[N, C, H * scale_h, W * scale_w]`.
    UpsampleNearest2d {
        /// Vertical upsampling scale factor.
        scale_h: usize,
        /// Horizontal upsampling scale factor.
        scale_w: usize,
    },

    // --- Padding ---
    /// 1-D constant padding (see `nn::pad::ConstantPad1d`).
    ConstantPad1d {
        /// Left padding.
        pad_left: usize,
        /// Right padding.
        pad_right: usize,
        /// The constant fill value.
        value: f64,
    },
    /// 2-D constant padding (see `nn::pad::ConstantPad2d`).
    ConstantPad2d {
        /// Left padding.
        pad_l: usize,
        /// Right padding.
        pad_r: usize,
        /// Top padding.
        pad_t: usize,
        /// Bottom padding.
        pad_b: usize,
        /// The constant fill value.
        value: f64,
    },
    /// 3-D constant padding (see `nn::pad::ConstantPad3d`).
    ConstantPad3d {
        /// Padding before the depth dimension.
        pad_d1: usize,
        /// Padding after the depth dimension.
        pad_d2: usize,
        /// Padding before the height dimension.
        pad_h1: usize,
        /// Padding after the height dimension.
        pad_h2: usize,
        /// Padding before the width dimension.
        pad_w1: usize,
        /// Padding after the width dimension.
        pad_w2: usize,
        /// The constant fill value.
        value: f64,
    },
    /// 1-D reflection padding (see `nn::pad::ReflectionPad1d`).
    ReflectionPad1d {
        /// Left padding.
        pad_left: usize,
        /// Right padding.
        pad_right: usize,
    },
    /// 2-D reflection padding (see `nn::pad::ReflectionPad2d`).
    ReflectionPad2d {
        /// Left padding.
        pad_l: usize,
        /// Right padding.
        pad_r: usize,
        /// Top padding.
        pad_t: usize,
        /// Bottom padding.
        pad_b: usize,
    },
    /// 3-D reflection padding (see `nn::pad::ReflectionPad3d`).
    ReflectionPad3d {
        /// Padding before the depth dimension.
        pad_d1: usize,
        /// Padding after the depth dimension.
        pad_d2: usize,
        /// Padding before the height dimension.
        pad_h1: usize,
        /// Padding after the height dimension.
        pad_h2: usize,
        /// Padding before the width dimension.
        pad_w1: usize,
        /// Padding after the width dimension.
        pad_w2: usize,
    },
    /// 1-D replication padding (see `nn::pad::ReplicationPad1d`).
    ReplicationPad1d {
        /// Left padding.
        pad_left: usize,
        /// Right padding.
        pad_right: usize,
    },
    /// 2-D replication padding (see `nn::pad::ReplicationPad2d`).
    ReplicationPad2d {
        /// Left padding.
        pad_l: usize,
        /// Right padding.
        pad_r: usize,
        /// Top padding.
        pad_t: usize,
        /// Bottom padding.
        pad_b: usize,
    },
    /// 3-D replication padding (see `nn::pad::ReplicationPad3d`).
    ReplicationPad3d {
        /// Padding before the depth dimension.
        pad_d1: usize,
        /// Padding after the depth dimension.
        pad_d2: usize,
        /// Padding before the height dimension.
        pad_h1: usize,
        /// Padding after the height dimension.
        pad_h2: usize,
        /// Padding before the width dimension.
        pad_w1: usize,
        /// Padding after the width dimension.
        pad_w2: usize,
    },
    /// 1-D circular padding (see `nn::pad::CircularPad1d`).
    CircularPad1d {
        /// Left padding.
        pad_left: usize,
        /// Right padding.
        pad_right: usize,
    },
    /// 2-D circular padding (see `nn::pad::CircularPad2d`).
    CircularPad2d {
        /// Left padding.
        pad_l: usize,
        /// Right padding.
        pad_r: usize,
        /// Top padding.
        pad_t: usize,
        /// Bottom padding.
        pad_b: usize,
    },
    /// 3-D circular padding (see `nn::pad::CircularPad3d`).
    CircularPad3d {
        /// Padding before the depth dimension.
        pad_d1: usize,
        /// Padding after the depth dimension.
        pad_d2: usize,
        /// Padding before the height dimension.
        pad_h1: usize,
        /// Padding after the height dimension.
        pad_h2: usize,
        /// Padding before the width dimension.
        pad_w1: usize,
        /// Padding after the width dimension.
        pad_w2: usize,
    },

    // --- Activation ---
    /// ReLU activation (see `nn::activation::relu::Relu`).
    Relu,
    /// ELU activation (see `nn::activation::elu::Elu`).
    Elu {
        /// The `alpha` parameter.
        alpha: f64,
    },
    /// SELU activation (see `nn::activation::elu::Selu`).
    Selu,
    /// CELU activation (see `nn::activation::elu::Celu`).
    Celu {
        /// The `alpha` parameter.
        alpha: f64,
    },
    /// GELU activation (see `nn::activation::gelu::Gelu`).
    Gelu,
    /// Mish activation (see `nn::activation::gelu::Mish`).
    Mish,
    /// Hardtanh activation (see `nn::activation::hard::Hardtanh`).
    Hardtanh {
        /// The lower clamp bound.
        min_val: f64,
        /// The upper clamp bound.
        max_val: f64,
    },
    /// ReLU6 activation (see `nn::activation::hard::Relu6`).
    Relu6,
    /// Hard-sigmoid activation (see `nn::activation::hard::Hardsigmoid`).
    Hardsigmoid,
    /// Hard-swish activation (see `nn::activation::hard::Hardswish`).
    Hardswish,
    /// Hardshrink activation (see `nn::activation::hard::Hardshrink`).
    Hardshrink {
        /// The shrinkage threshold.
        lambda: f64,
    },
    /// Leaky ReLU activation (see `nn::activation::misc::LeakyRelu`).
    LeakyRelu {
        /// The slope applied to negative inputs.
        negative_slope: f64,
    },
    /// Threshold activation (see `nn::activation::misc::Threshold`).
    Threshold {
        /// The threshold value.
        threshold: f64,
        /// The constant fill value.
        value: f64,
    },
    /// Softsign activation (see `nn::activation::misc::Softsign`).
    Softsign,
    /// Softshrink activation (see `nn::activation::misc::Softshrink`).
    Softshrink {
        /// The shrinkage threshold.
        lambda: f64,
    },
    /// Softplus activation (see `nn::activation::misc::Softplus`).
    Softplus {
        /// The `beta` parameter.
        beta: f64,
        /// The threshold value.
        threshold: f64,
    },
    /// Sigmoid activation (see `nn::activation::sigmoid::Sigmoid`).
    Sigmoid,
    /// SiLU/Swish activation (see `nn::activation::sigmoid::Silu`).
    Silu,
    /// Log-sigmoid activation (see `nn::activation::sigmoid::Logsigmoid`).
    Logsigmoid,
    /// Tanh activation (see `nn::activation::tanh::Tanh`).
    Tanh,
    /// Tanhshrink activation (see `nn::activation::tanh::Tanhshrink`).
    Tanhshrink,
    /// Softmax activation (see `nn::activation::softmax::Softmax`).
    Softmax {
        /// The dimension to operate along.
        dim: usize,
    },

    // --- Attention ---
    /// Multi-head self-attention with Flash Attention 2 and position encoding.
    /// Represents the full `Attention.forward()` in PSABlock:
    ///   qkv conv → FA2 → pe depthwise conv → proj conv → residual add.
    /// Input/output shape: `[N, c, H, W]`.
    Attention {
        /// Number of channels.
        c: usize,
        /// Number of attention heads.
        num_heads: usize,
        /// Per-head key/query dimension.
        key_dim: usize,
    },

    // --- Tensor structural ops ---
    /// Element-wise addition of two tensors with identical shapes.
    Add,
    /// Extract one contiguous channel slice from a 4-D NCHW tensor.
    /// Output shape: `[N, chunk_c, H, W]`.
    ChannelChunk {
        /// Total number of channels across all inputs/outputs.
        c_total: usize,
        /// Number of channels in this chunk.
        chunk_c: usize,
        /// Channel offset of this chunk within the total.
        chunk_offset: usize,
    },
    /// Concatenate N 4-D NCHW tensors along the channel dimension.
    /// Output shape: `[N, c_total, H, W]`.
    ChannelCat {
        /// Total number of channels across all inputs/outputs.
        c_total: usize,
    },
    /// Adds a (C,) bias vector to a (B, C, H, W) feature map — NC layout (N=B*H*W).
    /// Output shape equals input shape.
    ChannelBiasAdd {
        /// Number of channels.
        c: usize,
    },

    /// User-defined op.  Shape and dtype must be provided via [`Graph::add_node`]
    /// or [`SymTensor::record_custom`] — the base system cannot infer them.
    Custom {
        /// The wrapped user-defined op.
        data: CustomData,
    },

    // -----------------------------------------------------------------------
    // ONNX-sourced ops — added to let the ONNX loader build a complete graph.
    // Triton/CPU lowering is not yet implemented for these variants.
    // -----------------------------------------------------------------------

    // --- Element-wise unary math ---
    /// Element-wise absolute value (ONNX `Abs`).
    Abs,
    /// Element-wise negation (ONNX `Neg`).
    Neg,
    /// Element-wise ceiling (ONNX `Ceil`).
    Ceil,
    /// Element-wise floor (ONNX `Floor`).
    Floor,
    /// Element-wise round-to-nearest-even (ONNX `Round`).
    Round,
    /// Element-wise square root (ONNX `Sqrt`).
    Sqrt,
    /// Element-wise reciprocal, `1/x` (ONNX `Reciprocal`).
    Reciprocal,
    /// Element-wise natural exponential (ONNX `Exp`).
    Exp,
    /// Element-wise natural logarithm (ONNX `Log`).
    Log,
    /// Element-wise error function (ONNX `Erf`).
    Erf,
    /// Element-wise sign (ONNX `Sign`).
    Sign,
    /// Element-wise NaN test (ONNX `IsNaN`).
    IsNaN,
    /// Element-wise infinity test (ONNX `IsInf`).
    IsInf {
        /// Whether to treat negative infinity as infinite.
        detect_negative: bool,
        /// Whether to treat positive infinity as infinite.
        detect_positive: bool,
    },
    /// Element-wise logical NOT (ONNX `Not`).
    Not,
    /// Element-wise bitwise NOT (ONNX `BitwiseNot`).
    BitwiseNot,
    /// Element-wise sine (ONNX `Sin`).
    Sin,
    /// Element-wise cosine (ONNX `Cos`).
    Cos,
    /// Element-wise tangent (ONNX `Tan`).
    Tan,
    /// Element-wise arcsine (ONNX `Asin`).
    Asin,
    /// Element-wise arccosine (ONNX `Acos`).
    Acos,
    /// Element-wise arctangent (ONNX `Atan`).
    Atan,
    /// Element-wise hyperbolic sine (ONNX `Sinh`).
    Sinh,
    /// Element-wise hyperbolic cosine (ONNX `Cosh`).
    Cosh,
    /// Element-wise inverse hyperbolic sine (ONNX `Asinh`).
    Asinh,
    /// Element-wise inverse hyperbolic cosine (ONNX `Acosh`).
    Acosh,
    /// Element-wise inverse hyperbolic tangent (ONNX `Atanh`).
    Atanh,

    // --- Element-wise binary / variadic ---
    /// Element-wise multiplication (ONNX `Mul`).
    Mul,
    /// Element-wise subtraction (ONNX `Sub`).
    Sub,
    /// Element-wise division (ONNX `Div`).
    Div,
    /// Element-wise exponentiation (ONNX `Pow`).
    Pow,
    /// Element-wise modulo (ONNX `Mod`).
    Mod {
        /// Whether to use C-style (`fmod`) semantics instead of Python-style modulo.
        fmod: bool,
    },
    /// Element-wise minimum across inputs (ONNX `Min`).
    ElemMin,
    /// Element-wise maximum across inputs (ONNX `Max`).
    ElemMax,
    /// Element-wise mean across inputs (ONNX `Mean`).
    ElemMean,
    /// Element-wise sum across inputs (ONNX `Sum`).
    ElemSum,
    /// Element-wise equality (ONNX `Equal`).
    Equal,
    /// Element-wise greater-than (ONNX `Greater`).
    Greater,
    /// Element-wise greater-than-or-equal (ONNX `GreaterOrEqual`).
    GreaterOrEqual,
    /// Element-wise less-than (ONNX `Less`).
    Less,
    /// Element-wise less-than-or-equal (ONNX `LessOrEqual`).
    LessOrEqual,
    /// Element-wise logical AND (ONNX `And`).
    And,
    /// Element-wise logical OR (ONNX `Or`).
    Or,
    /// Element-wise logical XOR (ONNX `Xor`).
    Xor,
    /// Element-wise bitwise AND (ONNX `BitwiseAnd`).
    BitwiseAnd,
    /// Element-wise bitwise OR (ONNX `BitwiseOr`).
    BitwiseOr,
    /// Element-wise bitwise XOR (ONNX `BitwiseXor`).
    BitwiseXor,
    /// Element-wise bit shift (ONNX `BitShift`).
    BitShift {
        /// Direction of iteration/shift (e.g. `"forward"`, `"reverse"`, `"bidirectional"`).
        direction: alloc::string::String,
    },

    // --- Tensor structural ---
    /// Reshapes a tensor without changing its data (ONNX `Reshape`).
    Reshape,
    /// Permutes a tensor's dimensions (ONNX `Transpose`).
    Transpose {
        /// The output permutation of input dimensions.
        perm: alloc::vec::Vec<usize>,
    },
    /// Removes size-1 dimensions (ONNX `Squeeze`).
    Squeeze {
        /// The axes to operate along.
        axes: alloc::vec::Vec<i64>,
    },
    /// Inserts size-1 dimensions (ONNX `Unsqueeze`).
    Unsqueeze {
        /// The axes to operate along.
        axes: alloc::vec::Vec<i64>,
    },
    /// Concatenates tensors along an axis (ONNX `Concat`).
    Concat {
        /// The axis to operate along.
        axis: i64,
    },
    /// Splits a tensor into multiple outputs along an axis (ONNX `Split`).
    Split {
        /// The axis to operate along.
        axis: i64,
        /// Number of outputs to split into.
        num_outputs: usize,
    },
    /// Extracts a slice of a tensor (ONNX `Slice`).
    Slice,
    /// Gathers slices along an axis using an index tensor (ONNX `Gather`).
    Gather {
        /// The axis to operate along.
        axis: i64,
    },
    /// Gathers individual elements along an axis (ONNX `GatherElements`).
    GatherElements {
        /// The axis to operate along.
        axis: i64,
    },
    /// Gathers slices using N-D indices (ONNX `GatherND`).
    GatherND {
        /// Number of leading batch dimensions.
        batch_dims: i64,
    },
    /// Scatters individual elements along an axis (ONNX `ScatterElements`).
    ScatterElements {
        /// The axis to operate along.
        axis: i64,
    },
    /// Scatters slices using N-D indices (ONNX `ScatterND`).
    ScatterND,
    /// Tiles a tensor by repeating it (ONNX `Tile`).
    Tile,
    /// Broadcasts a tensor to a larger shape (ONNX `Expand`).
    Expand,
    /// Returns a tensor's shape as a 1-D tensor (ONNX `Shape`).
    ShapeOf {
        /// Start index.
        start: i64,
        /// End index.
        end: i64,
    },
    /// Returns the total number of elements (ONNX `Size`).
    SizeOf,
    /// Passes the input through unchanged (ONNX `Identity`).
    Identity,
    /// Casts a tensor to another dtype (ONNX `Cast`).
    Cast {
        /// Target dtype.
        to: DtypeRepr,
    },
    /// Casts a tensor to match another tensor's dtype (ONNX `CastLike`).
    CastLike,
    /// Element-wise conditional selection (ONNX `Where`).
    Where,
    /// Selects slices along an axis using a boolean mask (ONNX `Compress`).
    Compress {
        /// The axis to operate along.
        axis: i64,
    },
    /// Generates a range of values (ONNX `Range`).
    Range,
    /// Constant tensor (value embedded in the ONNX model).
    Constant {
        /// The dtype to use.
        dtype: DtypeRepr,
        /// The tensor shape.
        shape: Shape,
    },
    /// Creates a constant-filled tensor of a given shape (ONNX `ConstantOfShape`).
    ConstantOfShape {
        /// The dtype to use.
        dtype: DtypeRepr,
    },
    /// Extracts the upper or lower triangular part of a matrix (ONNX `Trilu`).
    Trilu {
        /// Whether to keep the upper (vs. lower) triangular part.
        upper: bool,
    },
    /// Reinterprets a tensor's bits as another dtype without conversion.
    BitCast {
        /// Target dtype.
        to: DtypeRepr,
    },
    /// Generic padding (ONNX `Pad`).
    Pad {
        /// The mode string selecting op-specific behavior.
        mode: alloc::string::String,
    },
    /// Reverses variable-length sequences along an axis (ONNX `ReverseSequence`).
    ReverseSequence {
        /// The batch axis.
        batch_axis: i64,
        /// The time axis.
        time_axis: i64,
    },
    /// Returns the indices of non-zero elements (ONNX `NonZero`).
    NonZero,
    /// Scatters values along an axis (deprecated ONNX `Scatter`, superseded by `ScatterElements`).
    Scatter {
        /// The axis to operate along.
        axis: i64,
    },
    /// Scatters an entire tensor into another at given indices.
    TensorScatter,

    // --- Matrix ---
    /// General matrix multiply: `alpha * A @ B + beta * C` (ONNX `Gemm`).
    Gemm {
        /// The `alpha` parameter.
        alpha: f64,
        /// The `beta` parameter.
        beta: f64,
        /// Whether to transpose the first matrix operand.
        trans_a: bool,
        /// Whether to transpose the second matrix operand.
        trans_b: bool,
    },
    /// Matrix multiplication (ONNX `MatMul`).
    MatMul,
    /// Integer matrix multiplication (ONNX `MatMulInteger`).
    MatMulInteger,
    /// Einstein-summation contraction (ONNX `Einsum`).
    Einsum {
        /// The Einstein-summation equation string.
        equation: alloc::string::String,
    },
    /// Matrix determinant (ONNX `Det`).
    Det,
    /// Quantized linear matrix multiplication (ONNX `QLinearMatMul`).
    QLinearMatMul,

    // --- Convolution extras ---
    /// Transposed (deconvolution) 2-D convolution (ONNX `ConvTranspose`).
    ConvTranspose {
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
        /// Vertical zero-padding.
        padding_h: usize,
        /// Horizontal zero-padding.
        padding_w: usize,
        /// Additional vertical padding added to the output (transposed convolution).
        output_padding_h: usize,
        /// Additional horizontal padding added to the output (transposed convolution).
        output_padding_w: usize,
        /// Number of blocked/grouped connections (1 = standard).
        groups: usize,
        /// Whether a learned bias is added.
        has_bias: bool,
    },
    /// Integer convolution (ONNX `ConvInteger`).
    ConvInteger {
        /// Number of blocked/grouped connections (1 = standard).
        groups: usize,
    },
    /// Deformable convolution (ONNX `DeformConv`).
    DeformConv {
        /// Number of blocked/grouped connections (1 = standard).
        group: usize,
        /// Number of groups for deformable-convolution offset channels.
        offset_group: usize,
    },
    /// Quantized linear convolution (ONNX `QLinearConv`).
    QLinearConv {
        /// Number of blocked/grouped connections (1 = standard).
        groups: usize,
    },
    /// Combines sliding local blocks into a large tensor (ONNX `Col2Im`, inverse of im2col).
    Col2Im {
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
    },
    /// Stateful causal 1-D convolution, carrying a sliding-window state between calls (ONNX
    /// `CausalConvWithState`).
    CausalConvWithState {
        /// Name of the activation function applied after the convolution (empty = none).
        activation: alloc::string::String,
    },

    // --- Reductions ---
    /// Sum reduction along axes (ONNX `ReduceSum`).
    ReduceSum {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Mean reduction along axes (ONNX `ReduceMean`).
    ReduceMean {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Max reduction along axes (ONNX `ReduceMax`).
    ReduceMax {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Min reduction along axes (ONNX `ReduceMin`).
    ReduceMin {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Product reduction along axes (ONNX `ReduceProd`).
    ReduceProd {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// L1-norm reduction along axes (ONNX `ReduceL1`).
    ReduceL1 {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// L2-norm reduction along axes (ONNX `ReduceL2`).
    ReduceL2 {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Log-sum reduction along axes (ONNX `ReduceLogSum`).
    ReduceLogSum {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Log-sum-exp reduction along axes (ONNX `ReduceLogSumExp`).
    ReduceLogSumExp {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Sum-of-squares reduction along axes (ONNX `ReduceSumSquare`).
    ReduceSumSquare {
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether an empty `axes` list means "no-op" instead of "reduce all".
        noop_with_empty_axes: bool,
    },
    /// Cumulative sum along an axis (ONNX `CumSum`).
    CumSum {
        /// Whether to exclude the current element from the cumulative result.
        exclusive: bool,
        /// Whether to accumulate in reverse order.
        reverse: bool,
    },
    /// Cumulative product along an axis (ONNX `CumProd`).
    CumProd {
        /// Whether to exclude the current element from the cumulative result.
        exclusive: bool,
        /// Whether to accumulate in reverse order.
        reverse: bool,
    },
    /// Index of the maximum along an axis (ONNX `ArgMax`).
    ArgMax {
        /// The axis to operate along.
        axis: i64,
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether ties select the last (rather than first) matching index.
        select_last_index: bool,
    },
    /// Index of the minimum along an axis (ONNX `ArgMin`).
    ArgMin {
        /// The axis to operate along.
        axis: i64,
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
        /// Whether ties select the last (rather than first) matching index.
        select_last_index: bool,
    },
    /// Average-pools over the entire spatial extent (ONNX `GlobalAveragePool`).
    GlobalAvgPool,
    /// Max-pools over the entire spatial extent (ONNX `GlobalMaxPool`).
    GlobalMaxPool,
    /// Lp-norm normalization along an axis (ONNX `LpNormalization`).
    LpNormalization {
        /// The axis to operate along.
        axis: i64,
        /// The `p` in the p-norm.
        p: i64,
    },
    /// Mean/variance normalization along axes (ONNX `MeanVarianceNormalization`).
    MeanVarianceNormalization {
        /// The axes to operate along.
        axes: alloc::vec::Vec<i64>,
    },

    // --- Additional activations ---
    /// Log-softmax along an axis (ONNX `LogSoftmax`).
    LogSoftmax {
        /// The axis to operate along.
        axis: i64,
    },
    /// One-hot of the argmax along an axis (ONNX `Hardmax`).
    Hardmax {
        /// The axis to operate along.
        axis: i64,
    },
    /// Parametric ReLU, with a learned per-channel slope (ONNX `PRelu`).
    PRelu,
    /// ReLU that zeroes values at or below `alpha` (ONNX `ThresholdedRelu`).
    ThresholdedRelu {
        /// The `alpha` parameter.
        alpha: f64,
    },
    /// Shrinks values toward zero by `lambd`, with a `bias` offset (ONNX `Shrink`).
    Shrink {
        /// The shrinkage threshold.
        lambd: f64,
        /// A bias/offset value.
        bias: f64,
    },
    /// Clamps values to a `[min, max]` range (ONNX `Clip`).
    Clip,
    /// Swish/SiLU activation (ONNX `Swish`).
    Swish,
    /// Multi-head attention (ONNX `MultiHeadAttention`).
    MultiHeadAttention {
        /// Number of query attention heads.
        q_num_heads: usize,
        /// Number of key/value attention heads.
        kv_num_heads: usize,
    },
    /// Attention with a user-defined score-modification function (ONNX `FlexAttention`). The
    /// score-modification subgraph itself is not captured (consistent with `Loop`/`If`/`Scan`
    /// not capturing their subgraph bodies).
    FlexAttention {
        /// Attention score scale factor, if explicitly specified.
        scale: f64,
    },
    /// Linear-complexity attention (e.g. gated delta rule variants), optionally carrying state
    /// between calls (ONNX `LinearAttention`).
    LinearAttention {
        /// Number of query attention heads.
        q_num_heads: usize,
        /// Number of key/value attention heads.
        kv_num_heads: usize,
        /// Name of the state-update rule (e.g. a gated-delta variant).
        update_rule: alloc::string::String,
        /// Attention score scale factor, if explicitly specified.
        scale: f64,
    },

    // --- Normalisation (generic) ---
    /// Local response normalization (ONNX `LRN`).
    LRN {
        /// The `alpha` parameter.
        alpha: f64,
        /// The `beta` parameter.
        beta: f64,
        /// A bias/offset value.
        bias: f64,
        /// Window/kernel size.
        size: usize,
    },

    // --- Recurrent ---
    /// Long short-term memory recurrent layer (ONNX `LSTM`).
    Lstm {
        /// Size of the hidden state.
        hidden_size: usize,
        /// Direction of iteration/shift (e.g. `"forward"`, `"reverse"`, `"bidirectional"`).
        direction: alloc::string::String,
        /// Whether to run the recurrence in both directions.
        bidirectional: bool,
    },
    /// Gated recurrent unit layer (ONNX `GRU`).
    Gru {
        /// Size of the hidden state.
        hidden_size: usize,
        /// Direction of iteration/shift (e.g. `"forward"`, `"reverse"`, `"bidirectional"`).
        direction: alloc::string::String,
        /// Whether to run the recurrence in both directions.
        bidirectional: bool,
    },
    /// Simple recurrent layer (ONNX `RNN`).
    Rnn {
        /// Size of the hidden state.
        hidden_size: usize,
        /// Direction of iteration/shift (e.g. `"forward"`, `"reverse"`, `"bidirectional"`).
        direction: alloc::string::String,
        /// Whether to run the recurrence in both directions.
        bidirectional: bool,
    },

    // --- Resize / spatial ---
    /// Resizes a tensor (interpolation) (ONNX `Resize`).
    Resize {
        /// The mode string selecting op-specific behavior.
        mode: alloc::string::String,
        /// How resized coordinates map back to the input (ONNX `Resize` mode string).
        coordinate_transformation_mode: alloc::string::String,
        /// Whether to apply an anti-aliasing filter when downsampling.
        antialias: bool,
    },
    /// Samples a tensor at grid-specified locations (ONNX `GridSample`).
    GridSample {
        /// The mode string selecting op-specific behavior.
        mode: alloc::string::String,
        /// How out-of-bounds sample coordinates are handled.
        padding_mode: alloc::string::String,
        /// Whether corner pixels are aligned (vs. edge-aligned) when sampling/resizing.
        align_corners: bool,
    },
    /// Rearranges spatial blocks into depth/channels (ONNX `SpaceToDepth`).
    SpaceToDepth {
        /// Block size for the space/depth rearrangement.
        blocksize: usize,
    },
    /// Rearranges depth/channels into spatial blocks (ONNX `DepthToSpace`).
    DepthToSpace {
        /// Block size for the space/depth rearrangement.
        blocksize: usize,
        /// The mode string selecting op-specific behavior.
        mode: alloc::string::String,
    },
    /// Region-of-interest pooling with bilinear alignment (ONNX `RoiAlign`).
    RoiAlign {
        /// Output region height.
        output_h: usize,
        /// Output region width.
        output_w: usize,
        /// Number of sampling points per output bin (0 = adaptive).
        sampling_ratio: i64,
        /// Scale factor mapping ROI coordinates to the input feature map.
        spatial_scale: f64,
    },
    /// Generates a 2-D/3-D sampling grid from an affine matrix (ONNX `AffineGrid`).
    AffineGrid {
        /// Whether corner pixels are aligned (vs. edge-aligned) when sampling/resizing.
        align_corners: bool,
    },
    /// Inverse of max pooling, using stored indices (ONNX `MaxUnpool`).
    MaxUnpool {
        /// Convolution/pooling kernel height.
        kernel_h: usize,
        /// Convolution/pooling kernel width.
        kernel_w: usize,
        /// Vertical stride.
        stride_h: usize,
        /// Horizontal stride.
        stride_w: usize,
    },
    /// Crops or pads a tensor to a target shape, centered (ONNX `CenterCropPad`).
    CenterCropPad {
        /// The axes to operate along.
        axes: alloc::vec::Vec<i64>,
    },
    /// Filters overlapping boxes by score (ONNX `NonMaxSuppression`).
    NonMaxSuppression {
        /// Whether boxes are given as `(center_x, center_y, width, height)` instead of corners.
        center_point_box: bool,
    },

    // --- Misc ---
    /// Returns the top-K values/indices along an axis (ONNX `TopK`).
    TopK {
        /// The axis to operate along.
        axis: i64,
        /// Whether to return the largest (vs. smallest) K values.
        largest: bool,
        /// Whether outputs are sorted.
        sorted: bool,
    },
    /// Returns unique elements (ONNX `Unique`).
    Unique {
        /// Whether outputs are sorted.
        sorted: bool,
    },
    /// Dropout regularization (ONNX `Dropout`).
    Dropout {
        /// Whether dropout is active (vs. a no-op at inference).
        training_mode: bool,
    },
    /// Creates an identity-like 2-D tensor (ONNX `EyeLike`).
    EyeLike {
        /// The dtype to use.
        dtype: Option<DtypeRepr>,
        /// Diagonal offset.
        k: i64,
    },
    /// One-hot encodes indices along an axis (ONNX `OneHot`).
    OneHot {
        /// The axis to operate along.
        axis: i64,
    },
    /// Samples from a Bernoulli distribution using input probabilities (ONNX `Bernoulli`).
    Bernoulli {
        /// The dtype to use.
        dtype: Option<DtypeRepr>,
    },
    /// Samples uniform random values with another tensor's shape (ONNX `RandomUniformLike`).
    RandomUniformLike {
        /// The dtype to use.
        dtype: Option<DtypeRepr>,
        /// Upper bound of the sampling range.
        high: f64,
        /// Lower bound of the sampling range.
        low: f64,
    },
    /// Rotary position embedding (ONNX `RotaryEmbedding`).
    RotaryEmbedding,

    // --- Quantisation ---
    /// Linear quantization to a lower-precision dtype (ONNX `QuantizeLinear`).
    QuantizeLinear {
        /// The axis to operate along.
        axis: i64,
        /// Whether to saturate (clamp) out-of-range values instead of wrapping.
        saturate: bool,
    },
    /// Linear dequantization back to a floating-point dtype (ONNX `DequantizeLinear`).
    DequantizeLinear {
        /// The axis to operate along.
        axis: i64,
    },
    /// Dynamically computes quantization parameters and quantizes (ONNX `DynamicQuantizeLinear`).
    DynamicQuantizeLinear,

    // --- Signal ---
    /// Discrete Fourier transform (ONNX `DFT`).
    Dft {
        /// Whether to compute the inverse transform.
        inverse: bool,
        /// Whether to return only the non-redundant half of the spectrum.
        onesided: bool,
    },
    /// Short-time Fourier transform (ONNX `STFT`).
    Stft,
    /// Generates a mel-scale filterbank matrix (ONNX `MelWeightMatrix`).
    MelWeightMatrix,
    /// Generates a Hann window (ONNX `HannWindow`).
    HannWindow {
        /// Whether the window is periodic (vs. symmetric).
        periodic: bool,
    },
    /// Generates a Blackman window (ONNX `BlackmanWindow`).
    BlackmanWindow {
        /// Whether the window is periodic (vs. symmetric).
        periodic: bool,
    },
    /// Generates a Hamming window (ONNX `HammingWindow`).
    HammingWindow {
        /// Whether the window is periodic (vs. symmetric).
        periodic: bool,
    },

    // --- Loss ---
    /// Negative log-likelihood loss (ONNX `NegativeLogLikelihoodLoss`).
    NegativeLogLikelihoodLoss {
        /// The reduction mode applied to the per-element loss (e.g. `"mean"`, `"sum"`, `"none"`).
        reduction: alloc::string::String,
    },
    /// Softmax + cross-entropy loss (ONNX `SoftmaxCrossEntropyLoss`).
    SoftmaxCrossEntropyLoss {
        /// The reduction mode applied to the per-element loss (e.g. `"mean"`, `"sum"`, `"none"`).
        reduction: alloc::string::String,
    },

    // --- Sequences ---
    /// Indexes into a sequence (ONNX `SequenceAt`).
    SequenceAt,
    /// Constructs a sequence from tensors (ONNX `SequenceConstruct`).
    SequenceConstruct,
    /// Constructs an empty sequence (ONNX `SequenceEmpty`).
    SequenceEmpty,
    /// Removes an element from a sequence (ONNX `SequenceErase`).
    SequenceErase,
    /// Inserts an element into a sequence (ONNX `SequenceInsert`).
    SequenceInsert,
    /// Returns a sequence's length (ONNX `SequenceLength`).
    SequenceLength,
    /// Applies a subgraph to each element of a sequence (ONNX `SequenceMap`).
    SequenceMap,
    /// Splits a tensor into a sequence along an axis (ONNX `SplitToSequence`).
    SplitToSequence {
        /// The axis to operate along.
        axis: i64,
        /// Whether to retain reduced dimensions with length 1.
        keepdims: bool,
    },
    /// Concatenates a sequence's elements into one tensor (ONNX `ConcatFromSequence`).
    ConcatFromSequence {
        /// The axis to operate along.
        axis: i64,
        /// Whether to insert a new axis for the concatenation dimension.
        new_axis: bool,
    },
    /// Extracts the value from an optional (ONNX `OptionalGetElement`).
    OptionalGetElement,
    /// Tests whether an optional has a value (ONNX `OptionalHasElement`).
    OptionalHasElement,

    // --- Control flow ---
    /// Generic looping construct over a subgraph (ONNX `Loop`).
    Loop,
    /// Applies a subgraph iteratively over input sequences (ONNX `Scan`).
    Scan {
        /// Number of inputs treated as scanned sequences.
        num_scan_inputs: i64,
    },
    /// Conditional branch over subgraphs (ONNX `If`).
    If,

    // --- Optimiser ops ---
    /// Adagrad optimizer step (ONNX `Adagrad`).
    Adagrad,
    /// Adam optimizer step (ONNX `Adam`).
    Adam,
    /// Momentum optimizer step (ONNX `Momentum`).
    Momentum,
    /// Computes gradients of a subgraph (ONNX `Gradient`).
    Gradient,

    // --- String / NLP ---
    /// Normalizes strings (case folding, stop-word removal) (ONNX `StringNormalizer`).
    StringNormalizer,
    /// Tests strings against a regex (ONNX `RegexFullMatch`).
    RegexFullMatch {
        /// The regular expression pattern.
        pattern: alloc::string::String,
    },
    /// Concatenates strings element-wise (ONNX `StringConcat`).
    StringConcat,
    /// Splits strings on a delimiter (ONNX `StringSplit`).
    StringSplit,
    /// Computes TF-IDF n-gram features (ONNX `TfIdfVectorizer`).
    TfIdfVectorizer,
    /// Maps categorical labels to/from encoded values (ONNX `LabelEncoder`).
    LabelEncoder,

    // --- Other ML ---
    /// Selects elements from a tensor by index (ONNX-ML `ArrayFeatureExtractor`).
    ArrayFeatureExtractor,
    /// Binarizes values against a threshold (ONNX-ML `Binarizer`).
    Binarizer {
        /// The threshold value.
        threshold: f64,
    },
    /// Decision-tree ensemble inference (ONNX-ML `TreeEnsemble`).
    TreeEnsemble,
    /// Decodes an encoded image (e.g. PNG/JPEG) into a tensor (ONNX `ImageDecoder`).
    ImageDecoder,
}

/// One node in a [`Graph`]: an [`Op`] plus its producer indices, dtype, and output shape.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// This node's operation.
    pub op: Op,
    /// Indices of producer nodes in `Graph::nodes`; empty for `Input`.
    pub inputs: Vec<usize>,
    /// This node's output dtype.
    pub dtype: DtypeRepr,
    /// Output shape of this node. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
}

/// The traced computational graph: a list of [`GraphNode`]s plus optional node names.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    /// The graph's nodes, in the order they were recorded.
    pub nodes: Vec<GraphNode>,
    /// Node index → dotted name captured from [`crate::name_scope`] at recording time.
    pub names: BTreeMap<usize, String>,
}

impl Graph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a new node to the graph, returning its index.
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<usize>,
        dtype: DtypeRepr,
        shape: Shape,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            op,
            inputs,
            dtype,
            shape,
        });
        #[cfg(feature = "std")]
        if let Some(name) = crate::name_scope::current_scope() {
            self.names.insert(id, name);
        }
        id
    }

    /// Returns node indices in topological order (producers before consumers)
    /// using Kahn's algorithm. Panics if the graph contains a cycle.
    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];

        for (id, node) in self.nodes.iter().enumerate() {
            for &input in &node.inputs {
                in_degree[id] += 1;
                dependents[input].push(id);
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(id) = queue.pop() {
            order.push(id);
            for &dep in &dependents[id] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push(dep);
                }
            }
        }

        assert_eq!(order.len(), n, "graph contains a cycle");
        order
    }

}

// ---------------------------------------------------------------------------
// Shape inference — computes the output shape for each Op given an input shape
// ---------------------------------------------------------------------------

fn infer_output_shape(op: &Op, inputs: &[&Shape]) -> Shape {
    // Constant has no tensor inputs — its shape is embedded in the op itself.
    if let Op::Constant { shape, .. } = op {
        return shape.clone();
    }
    // Zero-input ops that produce no tensor output.
    if matches!(op, Op::SequenceEmpty | Op::OptionalHasElement) {
        return vec![];
    }
    let input = inputs[0];
    match op {
        Op::Input => input.clone(),

        // Element-wise / shape-preserving — output shape = input shape
        Op::Relu
        | Op::Elu { .. }
        | Op::Selu
        | Op::Celu { .. }
        | Op::Gelu
        | Op::Mish
        | Op::Hardtanh { .. }
        | Op::Relu6
        | Op::Hardsigmoid
        | Op::Hardswish
        | Op::Hardshrink { .. }
        | Op::LeakyRelu { .. }
        | Op::Threshold { .. }
        | Op::Softsign
        | Op::Softshrink { .. }
        | Op::Softplus { .. }
        | Op::Sigmoid
        | Op::Silu
        | Op::Logsigmoid
        | Op::Tanh
        | Op::Tanhshrink
        | Op::Softmax { .. }
        | Op::BatchNorm1d { .. }
        | Op::BatchNorm2d { .. }
        | Op::BatchNorm3d { .. }
        | Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::GroupNorm { .. }
        | Op::InstanceNorm1d { .. }
        | Op::InstanceNorm2d { .. }
        | Op::InstanceNorm3d { .. } => input.clone(),

        Op::Linear { out_features, .. } => {
            // [..., in_features] → [..., out_features]
            let mut out = input[..input.len() - 1].to_vec();
            out.push(Some(*out_features));
            out
        }

        Op::Flatten => {
            // [N, C, H, W, ...] → [N, C*H*W*...]
            let rest = &input[1..];
            let flat: Option<usize> = rest
                .iter()
                .try_fold(1usize, |acc, dim| dim.map(|d| acc * d));
            vec![input[0], flat]
        }

        // --- Convolution ---
        Op::Conv1d {
            out_channels,
            kernel_l,
            stride,
            padding,
            ..
        } => {
            // [N, C_in, L] → [N, C_out, L_out]
            let l_out = input[2].map(|l| (l + 2 * padding - kernel_l) / stride + 1);
            vec![input[0], Some(*out_channels), l_out]
        }

        Op::Conv2d {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ..
        }
        | Op::Conv2dBnSilu {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ..
        } => {
            // [N, C_in, H, W] → [N, C_out, H_out, W_out]
            let h_out = input[2].map(|h| (h + 2 * padding_h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w + 2 * padding_w - kernel_w) / stride_w + 1);
            vec![input[0], Some(*out_channels), h_out, w_out]
        }

        Op::Conv3d {
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            ..
        } => {
            // [N, C_in, D, H, W] → [N, C_out, D_out, H_out, W_out]
            let d_out = input[2].map(|d| (d + 2 * padding_d - kernel_d) / stride_d + 1);
            let h_out = input[3].map(|h| (h + 2 * padding_h - kernel_h) / stride_h + 1);
            let w_out = input[4].map(|w| (w + 2 * padding_w - kernel_w) / stride_w + 1);
            vec![input[0], Some(*out_channels), d_out, h_out, w_out]
        }

        // --- Pooling ---
        Op::AvgPool1d { kernel_l, stride } | Op::MaxPool1d { kernel_l, stride } => {
            let l_out = input[2].map(|l| (l - kernel_l) / stride + 1);
            vec![input[0], input[1], l_out]
        }

        Op::LpPool1d {
            kernel_l, stride, ..
        } => {
            let l_out = input[2].map(|l| (l - kernel_l) / stride + 1);
            vec![input[0], input[1], l_out]
        }

        Op::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        } => {
            let h_out = input[2].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            let h_out = input[2].map(|h| (h + 2 * pad_h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w + 2 * pad_w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::LpPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            ..
        } => {
            let h_out = input[2].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::AvgPool3d {
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
        }
        | Op::MaxPool3d {
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
        } => {
            let d_out = input[2].map(|d| (d - kernel_d) / stride_d + 1);
            let h_out = input[3].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[4].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], d_out, h_out, w_out]
        }

        Op::LpPool3d {
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            ..
        } => {
            let d_out = input[2].map(|d| (d - kernel_d) / stride_d + 1);
            let h_out = input[3].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[4].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], d_out, h_out, w_out]
        }

        // --- Upsample ---
        Op::UpsampleNearest2d { scale_h, scale_w } => {
            // [N, C, H, W] → [N, C, H * scale_h, W * scale_w]
            let h_out = input[2].map(|h| h * scale_h);
            let w_out = input[3].map(|w| w * scale_w);
            vec![input[0], input[1], h_out, w_out]
        }

        // --- Padding ---
        Op::ConstantPad1d {
            pad_left,
            pad_right,
            ..
        }
        | Op::ReflectionPad1d {
            pad_left,
            pad_right,
        }
        | Op::ReplicationPad1d {
            pad_left,
            pad_right,
        }
        | Op::CircularPad1d {
            pad_left,
            pad_right,
        } => {
            // [N, C, L] → [N, C, L + pad_left + pad_right]
            let l_out = input[2].map(|l| l + pad_left + pad_right);
            vec![input[0], input[1], l_out]
        }

        Op::ConstantPad2d {
            pad_l,
            pad_r,
            pad_t,
            pad_b,
            ..
        }
        | Op::ReflectionPad2d {
            pad_l,
            pad_r,
            pad_t,
            pad_b,
        }
        | Op::ReplicationPad2d {
            pad_l,
            pad_r,
            pad_t,
            pad_b,
        }
        | Op::CircularPad2d {
            pad_l,
            pad_r,
            pad_t,
            pad_b,
        } => {
            // [N, C, H, W] → [N, C, H + pad_t + pad_b, W + pad_l + pad_r]
            let h_out = input[2].map(|h| h + pad_t + pad_b);
            let w_out = input[3].map(|w| w + pad_l + pad_r);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::ConstantPad3d {
            pad_d1,
            pad_d2,
            pad_h1,
            pad_h2,
            pad_w1,
            pad_w2,
            ..
        }
        | Op::ReflectionPad3d {
            pad_d1,
            pad_d2,
            pad_h1,
            pad_h2,
            pad_w1,
            pad_w2,
        }
        | Op::ReplicationPad3d {
            pad_d1,
            pad_d2,
            pad_h1,
            pad_h2,
            pad_w1,
            pad_w2,
        }
        | Op::CircularPad3d {
            pad_d1,
            pad_d2,
            pad_h1,
            pad_h2,
            pad_w1,
            pad_w2,
        } => {
            // [N, C, D, H, W] → padded on each spatial dim
            let d_out = input[2].map(|d| d + pad_d1 + pad_d2);
            let h_out = input[3].map(|h| h + pad_h1 + pad_h2);
            let w_out = input[4].map(|w| w + pad_w1 + pad_w2);
            vec![input[0], input[1], d_out, h_out, w_out]
        }

        Op::Attention { .. } => input.clone(),

        Op::Add => input.clone(),

        Op::ChannelChunk { chunk_c, .. } => {
            // [N, c_total, H, W] → [N, chunk_c, H, W]
            vec![input[0], Some(*chunk_c), input[2], input[3]]
        }

        Op::ChannelCat { c_total } => {
            // multi-input; c_total encodes the output channel count
            vec![input[0], Some(*c_total), input[2], input[3]]
        }

        Op::ChannelBiasAdd { .. } => input.to_vec(),

        Op::Custom { data } => data.infer_output_shape(inputs),

        // -------------------------------------------------------------------
        // ONNX-sourced ops — shape inference below.
        // For ops whose output shape equals the primary input shape (element-
        // wise, identity-like, or shape-tracked via ONNX value_info), we just
        // clone the input shape.  Ops with genuinely different output shapes
        // have explicit arms.
        // -------------------------------------------------------------------

        // Unary element-wise — output shape = input shape
        Op::Abs
        | Op::Neg
        | Op::Ceil
        | Op::Floor
        | Op::Round
        | Op::Sqrt
        | Op::Reciprocal
        | Op::Exp
        | Op::Log
        | Op::Erf
        | Op::Sign
        | Op::IsNaN
        | Op::IsInf { .. }
        | Op::Not
        | Op::BitwiseNot
        | Op::Sin
        | Op::Cos
        | Op::Tan
        | Op::Asin
        | Op::Acos
        | Op::Atan
        | Op::Sinh
        | Op::Cosh
        | Op::Asinh
        | Op::Acosh
        | Op::Atanh
        | Op::PRelu
        | Op::ThresholdedRelu { .. }
        | Op::Shrink { .. }
        | Op::Clip
        | Op::Swish
        | Op::LogSoftmax { .. }
        | Op::Hardmax { .. }
        | Op::Dropout { .. }
        | Op::Identity
        | Op::LRN { .. }
        | Op::MeanVarianceNormalization { .. }
        | Op::LpNormalization { .. }
        | Op::Pad { .. }
        | Op::ReverseSequence { .. }
        | Op::Trilu { .. }
        | Op::CumSum { .. }
        | Op::CumProd { .. }
        | Op::QuantizeLinear { .. }
        | Op::DequantizeLinear { .. }
        | Op::DynamicQuantizeLinear
        | Op::Bernoulli { .. }
        | Op::RandomUniformLike { .. }
        | Op::EyeLike { .. }
        | Op::RotaryEmbedding
        | Op::MultiHeadAttention { .. }
        | Op::FlexAttention { .. }
        | Op::LinearAttention { .. }
        | Op::CausalConvWithState { .. } => input.clone(),

        // Binary / variadic element-wise — approximate as first-input shape
        Op::Mul
        | Op::Sub
        | Op::Div
        | Op::Pow
        | Op::Mod { .. }
        | Op::ElemMin
        | Op::ElemMax
        | Op::ElemMean
        | Op::ElemSum
        | Op::Equal
        | Op::Greater
        | Op::GreaterOrEqual
        | Op::Less
        | Op::LessOrEqual
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::BitwiseAnd
        | Op::BitwiseOr
        | Op::BitwiseXor
        | Op::BitShift { .. }
        | Op::Cast { .. }
        | Op::CastLike
        | Op::BitCast { .. }
        | Op::Where => input.clone(),

        // Structural ops where output shape = input shape or is unknown at
        // static inference time (ONNX value_info carries the true shape).
        Op::Reshape
        | Op::Squeeze { .. }
        | Op::Unsqueeze { .. }
        | Op::Slice
        | Op::Gather { .. }
        | Op::GatherElements { .. }
        | Op::GatherND { .. }
        | Op::ScatterElements { .. }
        | Op::ScatterND
        | Op::Tile
        | Op::Expand
        | Op::Compress { .. }
        | Op::Range
        | Op::ConstantOfShape { .. }
        | Op::NonZero
        | Op::Scatter { .. }
        | Op::TensorScatter
        | Op::Resize { .. }
        | Op::GridSample { .. }
        | Op::AffineGrid { .. }
        | Op::CenterCropPad { .. } => input.clone(),

        Op::Transpose { perm } => {
            if perm.is_empty() {
                input.iter().rev().cloned().collect()
            } else {
                perm.iter()
                    .map(|&i| input.get(i).copied().unwrap_or(None))
                    .collect()
            }
        }

        Op::Concat { axis } => {
            let rank = input.len();
            if rank == 0 {
                return input.clone();
            }
            let ax = axis.rem_euclid(rank as i64) as usize;
            let mut out = input.clone();
            // Sum the concatenated axis across all inputs.
            out[ax] = inputs
                .iter()
                .try_fold(0usize, |acc, s| {
                    s.get(ax).copied().unwrap_or(None).map(|d| acc + d)
                })
                .map(Some)
                .unwrap_or(None);
            out
        }

        Op::Split { axis, num_outputs } => {
            let rank = input.len();
            if rank == 0 {
                return input.clone();
            }
            let ax = axis.rem_euclid(rank as i64) as usize;
            let mut out = input.clone();
            out[ax] = input[ax].map(|d| d / num_outputs.max(&1));
            out
        }

        Op::ShapeOf { start, end } => {
            let rank = input.len() as i64;
            let s = start.rem_euclid(rank.max(1));
            let e = end.rem_euclid(rank.max(1));
            vec![Some((e - s).max(0) as usize)]
        }

        Op::SizeOf => vec![Some(1)],

        Op::Gemm {
            trans_a, trans_b, ..
        } => {
            let m = if *trans_a {
                input.get(1)
            } else {
                input.first()
            }
            .copied()
            .unwrap_or(None);
            let n = if inputs.len() >= 2 {
                let b = inputs[1];
                if *trans_b { b.first() } else { b.get(1) }
                    .copied()
                    .unwrap_or(None)
            } else {
                None
            };
            vec![m, n]
        }

        Op::MatMul | Op::MatMulInteger | Op::QLinearMatMul => {
            if inputs.len() >= 2 && !input.is_empty() {
                let other = inputs[1];
                let mut out = input[..input.len() - 1].to_vec();
                out.push(other.last().copied().unwrap_or(None));
                out
            } else {
                input.clone()
            }
        }

        Op::Einsum { .. }
        | Op::Det
        | Op::Col2Im { .. }
        | Op::ConvInteger { .. }
        | Op::DeformConv { .. }
        | Op::QLinearConv { .. } => input.clone(),

        Op::ConvTranspose {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            ..
        } => {
            let h_out =
                input[2].map(|h| (h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h);
            let w_out =
                input[3].map(|w| (w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w);
            vec![input[0], Some(*out_channels), h_out, w_out]
        }

        Op::ReduceSum { keepdims, .. }
        | Op::ReduceMean { keepdims, .. }
        | Op::ReduceMax { keepdims, .. }
        | Op::ReduceMin { keepdims, .. }
        | Op::ReduceProd { keepdims, .. }
        | Op::ReduceL1 { keepdims, .. }
        | Op::ReduceL2 { keepdims, .. }
        | Op::ReduceLogSum { keepdims, .. }
        | Op::ReduceLogSumExp { keepdims, .. }
        | Op::ReduceSumSquare { keepdims, .. } => {
            // Without axis info at static-inference time, approximate:
            // keepdims=true → same rank, keepdims=false → reduce all → scalar.
            if *keepdims {
                input.clone()
            } else {
                vec![Some(1)]
            }
        }

        Op::ArgMax { axis, keepdims, .. } | Op::ArgMin { axis, keepdims, .. } => {
            if input.is_empty() {
                return vec![];
            }
            let ax = axis.rem_euclid(input.len() as i64) as usize;
            if *keepdims {
                let mut out = input.clone();
                out[ax] = Some(1);
                out
            } else {
                let mut out = input.clone();
                out.remove(ax);
                out
            }
        }

        Op::GlobalAvgPool | Op::GlobalMaxPool => {
            let mut out = input[..2.min(input.len())].to_vec();
            for _ in 2..input.len() {
                out.push(Some(1));
            }
            out
        }

        Op::Lstm {
            hidden_size,
            bidirectional,
            ..
        }
        | Op::Gru {
            hidden_size,
            bidirectional,
            ..
        }
        | Op::Rnn {
            hidden_size,
            bidirectional,
            ..
        } => {
            let num_dirs: usize = if *bidirectional { 2 } else { 1 };
            // [seq_len, num_directions, batch, hidden_size] (approximate)
            vec![
                input.first().copied().unwrap_or(None),
                Some(num_dirs),
                input.get(1).copied().unwrap_or(None),
                Some(*hidden_size),
            ]
        }

        Op::SpaceToDepth { blocksize } => {
            let c_out = input[1].map(|c| c * blocksize * blocksize);
            let h_out = input[2].map(|h| h / blocksize);
            let w_out = input[3].map(|w| w / blocksize);
            vec![input[0], c_out, h_out, w_out]
        }

        Op::DepthToSpace { blocksize, .. } => {
            let c_out = input[1].map(|c| c / (blocksize * blocksize));
            let h_out = input[2].map(|h| h * blocksize);
            let w_out = input[3].map(|w| w * blocksize);
            vec![input[0], c_out, h_out, w_out]
        }

        Op::RoiAlign {
            output_h, output_w, ..
        } => {
            vec![input[0], input[1], Some(*output_h), Some(*output_w)]
        }

        Op::MaxUnpool {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        } => {
            let h_out = input[2].map(|h| (h - 1) * stride_h + kernel_h);
            let w_out = input[3].map(|w| (w - 1) * stride_w + kernel_w);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::NonMaxSuppression { .. } => vec![None, Some(3)],

        Op::TopK { axis, .. } => {
            // Second input is k (runtime). Return input shape as approximation.
            let _ = axis;
            input.clone()
        }

        Op::Unique { .. } => input.clone(),
        Op::OneHot { .. } => input.clone(),

        Op::NegativeLogLikelihoodLoss { .. } | Op::SoftmaxCrossEntropyLoss { .. } => {
            vec![Some(1)]
        }

        Op::Dft { onesided, .. } => {
            // DFT last dim: full=N, onesided=N/2+1. Approximate.
            if *onesided && input.len() >= 2 {
                let mut out = input.clone();
                *out.last_mut().unwrap() = None;
                out
            } else {
                input.clone()
            }
        }

        Op::Stft
        | Op::MelWeightMatrix
        | Op::HannWindow { .. }
        | Op::BlackmanWindow { .. }
        | Op::HammingWindow { .. } => input.clone(),

        Op::SequenceAt
        | Op::SequenceConstruct
        | Op::SequenceErase
        | Op::SequenceInsert
        | Op::SequenceLength
        | Op::SequenceMap
        | Op::SplitToSequence { .. }
        | Op::ConcatFromSequence { .. }
        | Op::OptionalGetElement
        | Op::Loop
        | Op::Scan { .. }
        | Op::If
        | Op::Adagrad
        | Op::Adam
        | Op::Momentum
        | Op::Gradient
        | Op::StringNormalizer
        | Op::RegexFullMatch { .. }
        | Op::StringConcat
        | Op::StringSplit
        | Op::TfIdfVectorizer
        | Op::LabelEncoder
        | Op::ArrayFeatureExtractor
        | Op::Binarizer { .. }
        | Op::TreeEnsemble
        | Op::ImageDecoder => input.clone(),

        // Handled by early returns above the match; arms required for exhaustiveness.
        Op::Constant { shape, .. } => shape.clone(),
        Op::SequenceEmpty | Op::OptionalHasElement => vec![],
    }
}

// ---------------------------------------------------------------------------
// SymTensor — a tensor that writes to the graph on every operation
// ---------------------------------------------------------------------------

/// A symbolic tensor handle. Every layer operation on a `SymTensor` records
/// itself in the shared `Graph` and returns a new `SymTensor` pointing to
/// the new node. Cloning is cheap — it shares the graph via `Rc`.
#[derive(Clone)]
pub struct SymTensor {
    /// This tensor's node index in `graph`.
    pub node_id: usize,
    /// The shared graph this tensor's operations record into.
    pub graph: Rc<RefCell<Graph>>,
    /// This tensor's dtype.
    pub dtype: DtypeRepr,
    /// Output shape of this tensor. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
}

// SymTensor satisfies Tensor<D, RANK> for any D and RANK — shape is tracked
// dynamically at runtime; the compile-time SHAPE constant is zeroed (unused).
impl<D: Dtype, const RANK: usize> RankedTensor<D, RANK> for SymTensor {
    const SHAPE: [usize; RANK] = [0; RANK];
}
impl<D: Dtype, const RANK: usize> Tensor<D, RANK> for SymTensor {}

impl SymTensor {
    /// Create an input placeholder, returning both the tensor and the shared
    /// graph handle. Keep the graph handle to inspect the result after tracing.
    ///
    /// Use `None` for dynamic dimensions (e.g. the batch axis):
    /// ```ignore
    /// SymTensor::input(DtypeRepr::F32, vec![None, Some(784)])
    /// ```
    pub fn input(dtype: DtypeRepr, shape: Shape) -> (Self, Rc<RefCell<Graph>>) {
        let graph = Rc::new(RefCell::new(Graph::new()));
        let node_id = graph
            .borrow_mut()
            .add_node(Op::Input, vec![], dtype, shape.clone());
        let tensor = Self {
            node_id,
            graph: graph.clone(),
            dtype,
            shape,
        };
        (tensor, graph)
    }

    /// Number of dimensions of this tensor.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    fn record(&self, op: Op) -> Self {
        let output_shape = infer_output_shape(&op, &[&self.shape]);
        self.record_with_shape(op, output_shape)
    }

    fn record_with_shape(&self, op: Op, shape: Shape) -> Self {
        let node_id =
            self.graph
                .borrow_mut()
                .add_node(op, vec![self.node_id], self.dtype, shape.clone());
        Self {
            node_id,
            graph: self.graph.clone(),
            dtype: self.dtype,
            shape,
        }
    }

    /// Record a custom op whose output shape is determined by [`CustomOp::infer_output_shape`].
    ///
    /// `self` is the primary (first) input.  Pass additional inputs via
    /// `other_inputs`.  Pass `dtype` to override the output element type;
    /// defaults to the primary input's dtype.
    pub fn record_custom(
        &self,
        data: CustomData,
        other_inputs: &[&SymTensor],
        dtype: Option<DtypeRepr>,
    ) -> Self {
        let mut shapes: Vec<&Shape> = vec![&self.shape];
        shapes.extend(other_inputs.iter().map(|t| &t.shape));
        let output_shape = data.infer_output_shape(&shapes);

        let mut input_ids: Vec<usize> = vec![self.node_id];
        input_ids.extend(other_inputs.iter().map(|t| t.node_id));

        let out_dtype = dtype.unwrap_or(self.dtype);
        let node_id = self.graph.borrow_mut().add_node(
            Op::Custom { data },
            input_ids,
            out_dtype,
            output_shape.clone(),
        );
        Self {
            node_id,
            graph: self.graph.clone(),
            dtype: out_dtype,
            shape: output_shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Layer<SymTensor> impls — record op instead of computing
// ---------------------------------------------------------------------------

// --- Linear / MLP ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Linear<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Linear {
            in_features: self.in_features,
            out_features: self.out_features,
            has_bias: self.has_bias,
        })
    }
}

impl<D: Dtype> Layer<SymTensor> for Flatten<D, SymTensor, SymTensor> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Flatten)
    }
}

// --- Normalisation ---

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for BatchNorm1d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::LayerNorm {
            normalized_shape: self.normalized_shape.clone(),
            eps: self.eps,
            affine: self.affine,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for RmsNorm<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::RmsNorm {
            normalized_shape: self.normalized_shape.clone(),
            eps: self.eps,
            affine: self.affine,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for GroupNorm<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::AvgPool1d {
            kernel_l: self.kernel_l,
            stride: self.stride,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for AvgPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::MaxPool1d {
            kernel_l: self.kernel_l,
            stride: self.stride,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for MaxPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::LpPool1d {
            kernel_l: self.kernel_l,
            stride: self.stride,
            p: self.p,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for LpPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
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
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Relu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Elu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Elu { alpha: self.alpha })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Selu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Selu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Celu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Celu { alpha: self.alpha })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Gelu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Gelu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Mish<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Mish)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardtanh<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Hardtanh {
            min_val: self.min_val,
            max_val: self.max_val,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Relu6<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Relu6)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardsigmoid<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Hardsigmoid)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardswish<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Hardswish)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Hardshrink<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Hardshrink {
            lambda: self.lambda,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for LeakyRelu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::LeakyRelu {
            negative_slope: self.negative_slope,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Threshold<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Threshold {
            threshold: self.threshold,
            value: self.value,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softsign<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Softsign)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softshrink<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Softshrink {
            lambda: self.lambda,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softplus<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Softplus {
            beta: self.beta,
            threshold: self.threshold,
        })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Sigmoid<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Sigmoid)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Silu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Silu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Logsigmoid<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Logsigmoid)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Tanh<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Tanh)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Tanhshrink<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Tanhshrink)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softmax<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Softmax { dim: self.dim })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{
            activation::{relu::Relu, softmax::Softmax},
            conv2d::Conv2d,
            linear::Linear,
        },
        sequential,
    };

    #[test]
    fn test_sequential_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);

        let model = sequential![
            Linear::<f32, SymTensor, SymTensor, 2>::new(784, 128, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(128, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input);

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
    fn test_topological_sort_linear_chain() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);

        let model = sequential![
            Linear::<f32, SymTensor, SymTensor, 2>::new(784, 128, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(128, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input);

        let g = graph.borrow();
        let order = g.topological_sort();
        assert_eq!(order.len(), g.nodes.len());
        for (pos, &id) in order.iter().enumerate() {
            for &input_id in &g.nodes[id].inputs {
                let input_pos = order.iter().position(|&x| x == input_id).unwrap();
                assert!(
                    input_pos < pos,
                    "producer {input_id} must come before consumer {id}"
                );
            }
        }
    }

    #[test]
    fn test_residual_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(64)]);

        let main = Linear::<f32, SymTensor, SymTensor, 2>::new(64, 64, true).call(input.clone());
        let main = Relu::<f32, SymTensor, 2>::new().call(main);
        let skip = Linear::<f32, SymTensor, SymTensor, 2>::new(64, 64, false).call(input);

        assert!(Rc::ptr_eq(&main.graph, &skip.graph));

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 4);
        assert_eq!(g.nodes[1].inputs, vec![0]);
        assert_eq!(g.nodes[3].inputs, vec![0]);
    }

    #[test]
    fn test_conv2d_graph_extraction() {
        let (input, graph) =
            SymTensor::input(DtypeRepr::F32, vec![None, Some(3), Some(32), Some(32)]);

        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(3, 64, (3, 3), (1, 1), (1, 1), true);
        let _out = Layer::call(&conv, input);

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

    #[test]
    fn test_lenet5_shapes() {
        let (input, graph) =
            SymTensor::input(DtypeRepr::F32, vec![None, Some(1), Some(28), Some(28)]);

        use crate::{
            nn::{flatten::Flatten, pool::AvgPool2d},
            sequential,
        };

        let model = sequential![
            Conv2d::<f32, SymTensor, SymTensor, 4>::new(1, 6, (5, 5), (1, 1), (2, 2), true),
            Relu::<f32, SymTensor, 4>::new(),
            AvgPool2d::<f32, SymTensor, SymTensor, 4>::new((2, 2), (2, 2)),
            Conv2d::<f32, SymTensor, SymTensor, 4>::new(6, 16, (5, 5), (1, 1), (0, 0), true),
            Relu::<f32, SymTensor, 4>::new(),
            AvgPool2d::<f32, SymTensor, SymTensor, 4>::new((2, 2), (2, 2)),
            Flatten::<f32, SymTensor, SymTensor>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(400, 120, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(120, 84, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(84, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input);

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 14);
        assert_eq!(g.nodes[0].shape, vec![None, Some(1), Some(28), Some(28)]);
        assert_eq!(g.nodes[1].shape, vec![None, Some(6), Some(28), Some(28)]);
        assert_eq!(g.nodes[2].shape, vec![None, Some(6), Some(28), Some(28)]);
        assert_eq!(g.nodes[3].shape, vec![None, Some(6), Some(14), Some(14)]);
        assert_eq!(g.nodes[4].shape, vec![None, Some(16), Some(10), Some(10)]);
        assert_eq!(g.nodes[5].shape, vec![None, Some(16), Some(10), Some(10)]);
        assert_eq!(g.nodes[6].shape, vec![None, Some(16), Some(5), Some(5)]);
        assert_eq!(g.nodes[7].shape, vec![None, Some(400)]);
        assert_eq!(g.nodes[8].shape, vec![None, Some(120)]);
        assert_eq!(g.nodes[9].shape, vec![None, Some(120)]);
        assert_eq!(g.nodes[10].shape, vec![None, Some(84)]);
        assert_eq!(g.nodes[11].shape, vec![None, Some(84)]);
        assert_eq!(g.nodes[12].shape, vec![None, Some(10)]);
        assert_eq!(g.nodes[13].shape, vec![None, Some(10)]);
    }

}
