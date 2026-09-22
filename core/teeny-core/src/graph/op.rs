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

//! The computational-graph operation set.
//!
//! [`Op`] is the graph IR's instruction set: one variant per `nn` layer or
//! primitive, carrying that op's configuration, together with the forward
//! shape inference over those variants ([`Op::infer_output_shape`]).
//!
//! [`CustomOp`] is the extension point for ops defined outside this crate;
//! `Op::Custom` wraps one and delegates shape inference and lowering to it.

use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::any::Any;

use super::{DtypeRepr, Shape};
use crate::errors::{Error, Result};

/// Trait implemented by user-defined ops.
pub trait CustomOp: Any + Send + Sync + core::fmt::Debug {
    /// Identifier used in error messages and debug output.
    fn name(&self) -> &str;

    /// Compute the output shape given the shapes of all input tensors in order.
    ///
    /// # Errors
    ///
    /// Return an error when `input_shapes` are ones this op cannot accept —
    /// a rank it does not handle, or an extent that leaves no valid output.
    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape>;

    /// Compute the input shapes given the shape of the output tensor — the
    /// reverse of [`CustomOp::infer_output_shape`].
    ///
    /// Return `Ok(Some(shapes))` with one shape per input, in the order
    /// `infer_output_shape` receives them, or `Ok(None)` when this op is not
    /// invertible because its forward pass discards what the reverse would
    /// need. Return an error when `output_shape` could not have been produced
    /// by this op.
    ///
    /// There is no default: an op that cannot invert must say so explicitly
    /// with `Ok(None)`, so that a missing reverse is a decision rather than an
    /// oversight.
    fn infer_input_shape(&self, output_shape: &Shape) -> Result<Option<Vec<Shape>>>;

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

    /// Opt into Anduin pointwise fusion: return the CTA `BLOCK_SIZE` when this
    /// custom op is unary elementwise with the standard `n_elements` grid.
    ///
    /// Default `None` (not pointwise-fusable). Graph optimizers live in
    /// `teeny-kernels` and combine this with kernel probes; keep the hook here
    /// so customs can participate without an op-name allowlist.
    fn pointwise_fuse_block_size(&self) -> Option<i32> {
        None
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

    /// User-defined op.  Shape and dtype must be provided via
    /// [`Graph::add_node`](super::Graph::add_node) or
    /// [`SymTensor::record_custom`](super::SymTensor::record_custom) — the base
    /// system cannot infer them.
    Custom {
        /// The user-defined op.
        data: Arc<dyn CustomOp>,
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

// ---------------------------------------------------------------------------
// Shape inference — computes the output shape for each Op given an input shape
// ---------------------------------------------------------------------------

/// Output extent of a sliding-window op (convolution or pooling) along one
/// axis: `floor((extent + 2 * padding - kernel) / stride) + 1`.
///
/// `op` and `axis` name the caller in the errors below, e.g.
/// `("Conv2d", "height")`.
///
/// # Errors
///
/// Rejects a window that cannot fit, or a zero stride, rather than leaving
/// it to the arithmetic. These are all `usize`, so `extent + 2 * padding -
/// kernel` otherwise underflows: in debug builds that is an "attempt to
/// subtract with overflow" naming only this line, and in release it wraps to
/// an enormous extent that propagates silently through the rest of the graph.
fn window_out_dim(
    op: &str,
    axis: &str,
    extent: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize> {
    if stride == 0 {
        return Err(Error::ZeroWindowStride {
            op: op.into(),
            axis: axis.into(),
        }
        .into());
    }
    let padded = extent + 2 * padding;
    if kernel > padded {
        return Err(Error::WindowDoesNotFit {
            op: op.into(),
            axis: axis.into(),
            extent,
            kernel,
            padding,
        }
        .into());
    }
    Ok((padded - kernel) / stride + 1)
}

/// Recovers a sliding window's input extent from its output extent — the
/// reverse of [`window_out_dim`], sharing its `op`/`axis` naming.
///
/// Floor division makes the forward map many-to-one: exactly `stride` inputs
/// produce any given output. This returns the smallest of them,
/// `stride * (out - 1) + kernel - 2 * padding`, which is the only one when
/// `stride == 1`. Re-running [`window_out_dim`] on the result always gives
/// `out` back, and never trips either of its guards.
///
/// # Errors
///
/// Returns [`Error::ZeroWindowStride`] for a zero stride, and
/// [`Error::WindowOutputUnreachable`] when no input produces `out` — which is
/// every `out` below `(2p - k) / s + 1`, including 0.
fn window_in_dim(
    op: &str,
    axis: &str,
    out: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize> {
    if stride == 0 {
        return Err(Error::ZeroWindowStride {
            op: op.into(),
            axis: axis.into(),
        }
        .into());
    }
    out.checked_sub(1)
        .and_then(|steps| steps.checked_mul(stride))
        .and_then(|span| span.checked_add(kernel))
        .and_then(|padded| padded.checked_sub(2 * padding))
        .ok_or_else(|| {
            Error::WindowOutputUnreachable {
                op: op.into(),
                axis: axis.into(),
                out,
                kernel,
                stride,
                padding,
            }
            .into()
        })
}

/// One spatial axis's window parameters — kernel, stride and padding, in the
/// order [`window_in_dim`] and [`window_out_dim`] take them.
type Window = (usize, usize, usize);

/// One direction's per-axis arithmetic: [`window_out_dim`] going forward,
/// [`window_in_dim`] going back.
type WindowDim = fn(&str, &str, usize, usize, usize, usize) -> Result<usize>;

/// Names the spatial axes of an `N`-dimensional window, matching the wording
/// both directions use in their errors.
fn spatial_axes(n: usize) -> &'static [&'static str] {
    match n {
        1 => &["length"],
        2 => &["height", "width"],
        _ => &["depth", "height", "width"],
    }
}

/// Maps one `[N, C, spatial..]` shape of a window op onto the other, in
/// whichever direction `axis_dim` runs.
///
/// The batch dim carries over untouched. `channels` replaces the channel dim
/// for an op that changes it — a convolution's `out_channels` going forward,
/// its `in_channels` coming back — and `None` keeps the one already there,
/// which is what every pool wants. Each spatial dim goes through `axis_dim`,
/// and a dim that is already unknown stays unknown.
///
/// `N` is the number of spatial dims, so `windows` holds one entry per axis in
/// axis order and cannot disagree with the rank the op works at.
///
/// # Panics
///
/// Indexes `shape` up to `N + 2`, so a shorter one panics — see
/// `teenygrad-3kt`. [`window_in_shape`] rank-checks before calling this;
/// [`window_out_shape`] inherits `infer_output_shape`'s existing behaviour.
///
/// # Errors
///
/// Whatever `axis_dim` reports for a spatial extent it cannot map.
fn window_shape<const N: usize>(
    op: &str,
    shape: &Shape,
    channels: Option<usize>,
    windows: [Window; N],
    axis_dim: WindowDim,
) -> Result<Shape> {
    let mut mapped = Shape::with_capacity(N + 2);
    mapped.push(shape[0]);
    mapped.push(match channels {
        Some(channels) => Some(channels),
        None => shape[1],
    });

    let axes = spatial_axes(N);
    for (i, (kernel, stride, padding)) in windows.into_iter().enumerate() {
        // `axes` covers the 1-D to 3-D ops that exist; the fallback keeps a
        // wider one from indexing off the end.
        let axis = axes.get(i).copied().unwrap_or("spatial");
        let extent = shape[i + 2]
            .map(|extent| axis_dim(op, axis, extent, kernel, stride, padding))
            .transpose()?;
        mapped.push(extent);
    }

    Ok(mapped)
}

/// Computes a window op's output shape from its input shape.
///
/// # Panics
///
/// Inherits [`window_shape`]'s indexing.
///
/// # Errors
///
/// Whatever [`window_out_dim`] reports — a zero stride, or a kernel that does
/// not fit the padded input.
fn window_out_shape<const N: usize>(
    op: &str,
    input: &Shape,
    out_channels: Option<usize>,
    windows: [Window; N],
) -> Result<Shape> {
    window_shape(op, input, out_channels, windows, window_out_dim)
}

/// Recovers a window op's input shape from its output shape, in the
/// `Ok(Some(one_shape))` form [`Op::infer_input_shape`] returns: these ops
/// record a single data input, their kernel coming from the variant.
///
/// # Errors
///
/// Returns [`Error::ShapeRankMismatch`] when `output` is not rank `N + 2`, and
/// whatever [`window_in_dim`] reports for a spatial extent it cannot reverse.
fn window_in_shape<const N: usize>(
    op: &str,
    output: &Shape,
    in_channels: Option<usize>,
    windows: [Window; N],
) -> Result<Option<Vec<Shape>>> {
    expect_rank(op, output, N + 2)?;
    let input = window_shape(op, output, in_channels, windows, window_in_dim)?;
    Ok(Some(vec![input]))
}

/// Checks a shape's rank before the fixed-rank arms of
/// [`Op::infer_input_shape`] index into it.
///
/// # Errors
///
/// Returns [`Error::ShapeRankMismatch`] when `shape` is not rank `expected`.
fn expect_rank(op: &str, shape: &Shape, expected: usize) -> Result<()> {
    if shape.len() != expected {
        return Err(Error::ShapeRankMismatch {
            op: op.into(),
            expected,
            actual: shape.len(),
        }
        .into());
    }
    Ok(())
}

impl Op {
    /// Computes this op's output shape from the shapes of its inputs, in order.
    ///
    /// `Op::Constant`, `Op::SequenceEmpty` and `Op::OptionalHasElement` carry
    /// their own shape and ignore `inputs`; every other variant reads
    /// `inputs[0]`, so `inputs` must be non-empty.
    ///
    /// # Errors
    ///
    /// Sliding-window ops (convolutions and pooling) fail when the window
    /// cannot fit the padded input, or when a stride is zero — see
    /// [`Error::WindowDoesNotFit`] and [`Error::ZeroWindowStride`].
    pub fn infer_output_shape(&self, inputs: &[&Shape]) -> Result<Shape> {
        // Constant has no tensor inputs — its shape is embedded in the op itself.
        if let Op::Constant { shape, .. } = self {
            return Ok(shape.clone());
        }
        // Zero-input ops that produce no tensor output.
        if matches!(self, Op::SequenceEmpty | Op::OptionalHasElement) {
            return Ok(vec![]);
        }
        let input = inputs[0];
        Ok(match self {
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
            //
            // Each arm names its axes in order and hands them to
            // `window_out_shape`, which walks them through the window
            // arithmetic. A convolution passes `out_channels`, which replaces
            // the input's channel dim; a pool passes `None`, leaving it alone.
            Op::Conv1d {
                out_channels,
                kernel_l,
                stride,
                padding,
                ..
            } => {
                // [N, C_in, L] → [N, C_out, L_out]
                window_out_shape(
                    "Conv1d",
                    input,
                    Some(*out_channels),
                    [(*kernel_l, *stride, *padding)],
                )?
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
            } => {
                // [N, C_in, H, W] → [N, C_out, H_out, W_out]
                window_out_shape(
                    "Conv2d",
                    input,
                    Some(*out_channels),
                    [
                        (*kernel_h, *stride_h, *padding_h),
                        (*kernel_w, *stride_w, *padding_w),
                    ],
                )?
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
                window_out_shape(
                    "Conv3d",
                    input,
                    Some(*out_channels),
                    [
                        (*kernel_d, *stride_d, *padding_d),
                        (*kernel_h, *stride_h, *padding_h),
                        (*kernel_w, *stride_w, *padding_w),
                    ],
                )?
            }

            // --- Pooling ---
            Op::AvgPool1d { kernel_l, stride } | Op::MaxPool1d { kernel_l, stride } => {
                let name = if matches!(self, Op::AvgPool1d { .. }) {
                    "AvgPool1d"
                } else {
                    "MaxPool1d"
                };
                window_out_shape(name, input, None, [(*kernel_l, *stride, 0)])?
            }

            Op::LpPool1d {
                kernel_l, stride, ..
            } => window_out_shape("LpPool1d", input, None, [(*kernel_l, *stride, 0)])?,

            Op::AvgPool2d {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
            } => window_out_shape(
                "AvgPool2d",
                input,
                None,
                [(*kernel_h, *stride_h, 0), (*kernel_w, *stride_w, 0)],
            )?,

            Op::MaxPool2d {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
            } => window_out_shape(
                "MaxPool2d",
                input,
                None,
                [
                    (*kernel_h, *stride_h, *pad_h),
                    (*kernel_w, *stride_w, *pad_w),
                ],
            )?,

            Op::LpPool2d {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                ..
            } => window_out_shape(
                "LpPool2d",
                input,
                None,
                [(*kernel_h, *stride_h, 0), (*kernel_w, *stride_w, 0)],
            )?,

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
                let name = if matches!(self, Op::AvgPool3d { .. }) {
                    "AvgPool3d"
                } else {
                    "MaxPool3d"
                };
                window_out_shape(
                    name,
                    input,
                    None,
                    [
                        (*kernel_d, *stride_d, 0),
                        (*kernel_h, *stride_h, 0),
                        (*kernel_w, *stride_w, 0),
                    ],
                )?
            }

            Op::LpPool3d {
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                ..
            } => window_out_shape(
                "LpPool3d",
                input,
                None,
                [
                    (*kernel_d, *stride_d, 0),
                    (*kernel_h, *stride_h, 0),
                    (*kernel_w, *stride_w, 0),
                ],
            )?,

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

            Op::Custom { data } => data.infer_output_shape(inputs)?,

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
                    return Ok(input.clone());
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
                    return Ok(input.clone());
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
                let h_out = input[2]
                    .map(|h| (h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h);
                let w_out = input[3]
                    .map(|w| (w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w);
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
                    return Ok(vec![]);
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
        })
    }

    /// Recovers this op's input shapes from its output shape — the reverse of
    /// [`Op::infer_output_shape`].
    ///
    /// `Ok(Some(shapes))` holds one shape per input, in the order
    /// `infer_output_shape` expects them, so a unary op yields a single shape
    /// and `Op::Input` yields none. `Ok(None)` means this op is not
    /// invertible — its forward pass discarded what the reverse would need.
    ///
    /// Implemented for the single-input shape-preserving ops — the
    /// activations, norms and unary element-wise math — plus `Op::Input`,
    /// `Op::Custom`, which delegates to [`CustomOp::infer_input_shape`], and
    /// the sliding-window ops: the three convolutions, the nine pooling
    /// variants and `Op::Split`. Every other variant panics with
    /// `unimplemented!`.
    ///
    /// # Ambiguity
    ///
    /// Floor division makes a window's forward pass many-to-one: `stride`
    /// different input extents produce the same output extent, and `Split`
    /// divides its axis the same way. Where the true reverse is a set, these
    /// arms return its smallest member — the input that fits exactly, with no
    /// remainder the forward pass would have dropped. For `stride == 1` that
    /// set is a single value, so the answer is the exact reverse.
    ///
    /// Multi-input ops are deliberately excluded: this returns one shape per
    /// input, but an `Op` variant does not record how many inputs its node
    /// has, so `Add`, `Concat` and the like cannot report a correct-length
    /// result from the output shape alone. The convolutions are covered
    /// despite their weights because a node records only the data input; the
    /// kernel comes from the variant. See `teenygrad-3fy` for the remaining
    /// cases — ops that lose exactly one dimension, and the total-loss set.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` could not have been produced by this op,
    /// which is why the reverse pass is fallible where the forward one is not.
    /// No pointwise op can fail this way: the reverse is the identity and
    /// every shape is reachable. A window op rejects an output whose rank is
    /// not the one it produces ([`Error::ShapeRankMismatch`]) and an extent
    /// below the smallest it can emit ([`Error::WindowOutputUnreachable`]).
    pub fn infer_input_shape(&self, output: &Shape) -> Result<Option<Vec<Shape>>> {
        match self {
            // A placeholder has no producers, so there is nothing to infer.
            Op::Input => Ok(Some(vec![])),

            // Element-wise / shape-preserving — input shape = output shape.
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
            | Op::InstanceNorm3d { .. } => Ok(Some(vec![output.clone()])),

            // Unary element-wise — same reverse, one input, shape unchanged.
            // The forward pass groups more ops than these under "unary", but
            // the rest take several inputs (`PRelu`, `Clip`, `CumSum`, the
            // attentions) or change shape (`Pad`), and a single returned shape
            // would misreport their arity. See teenygrad-3fy.5.
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
            | Op::ThresholdedRelu { .. }
            | Op::Shrink { .. }
            | Op::Swish
            | Op::LogSoftmax { .. }
            | Op::Hardmax { .. }
            | Op::Identity
            | Op::LRN { .. }
            | Op::MeanVarianceNormalization { .. }
            | Op::LpNormalization { .. }
            | Op::Bernoulli { .. }
            | Op::RandomUniformLike { .. }
            | Op::EyeLike { .. } => Ok(Some(vec![output.clone()])),

            // A custom op knows its own reverse, or reports that it has none.
            Op::Custom { data } => data.infer_input_shape(output),

            // --- Floor-ambiguous windows: convolution and pooling ---
            //
            // Each arm names its axes in order and hands them to
            // `window_in_shape`, which enforces the rank and reverses every
            // spatial dim. Convolutions pass `in_channels`, which the forward
            // pass overwrote with `out_channels`; pools pass `None`, having
            // left the channel dim alone.
            Op::Conv1d {
                in_channels,
                kernel_l,
                stride,
                padding,
                ..
            } => window_in_shape(
                "Conv1d",
                output,
                Some(*in_channels),
                [(*kernel_l, *stride, *padding)],
            ),

            Op::Conv2d {
                in_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                ..
            } => window_in_shape(
                "Conv2d",
                output,
                Some(*in_channels),
                [
                    (*kernel_h, *stride_h, *padding_h),
                    (*kernel_w, *stride_w, *padding_w),
                ],
            ),

            Op::Conv3d {
                in_channels,
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
            } => window_in_shape(
                "Conv3d",
                output,
                Some(*in_channels),
                [
                    (*kernel_d, *stride_d, *padding_d),
                    (*kernel_h, *stride_h, *padding_h),
                    (*kernel_w, *stride_w, *padding_w),
                ],
            ),

            Op::AvgPool1d { kernel_l, stride } | Op::MaxPool1d { kernel_l, stride } => {
                let name = if matches!(self, Op::AvgPool1d { .. }) {
                    "AvgPool1d"
                } else {
                    "MaxPool1d"
                };
                window_in_shape(name, output, None, [(*kernel_l, *stride, 0)])
            }

            Op::LpPool1d {
                kernel_l, stride, ..
            } => window_in_shape("LpPool1d", output, None, [(*kernel_l, *stride, 0)]),

            Op::AvgPool2d {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
            } => window_in_shape(
                "AvgPool2d",
                output,
                None,
                [(*kernel_h, *stride_h, 0), (*kernel_w, *stride_w, 0)],
            ),

            Op::MaxPool2d {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
            } => window_in_shape(
                "MaxPool2d",
                output,
                None,
                [
                    (*kernel_h, *stride_h, *pad_h),
                    (*kernel_w, *stride_w, *pad_w),
                ],
            ),

            Op::LpPool2d {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                ..
            } => window_in_shape(
                "LpPool2d",
                output,
                None,
                [(*kernel_h, *stride_h, 0), (*kernel_w, *stride_w, 0)],
            ),

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
                let name = if matches!(self, Op::AvgPool3d { .. }) {
                    "AvgPool3d"
                } else {
                    "MaxPool3d"
                };
                window_in_shape(
                    name,
                    output,
                    None,
                    [
                        (*kernel_d, *stride_d, 0),
                        (*kernel_h, *stride_h, 0),
                        (*kernel_w, *stride_w, 0),
                    ],
                )
            }

            Op::LpPool3d {
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                ..
            } => window_in_shape(
                "LpPool3d",
                output,
                None,
                [
                    (*kernel_d, *stride_d, 0),
                    (*kernel_h, *stride_h, 0),
                    (*kernel_w, *stride_w, 0),
                ],
            ),

            // `Split` divides one axis by `num_outputs`, so it has the same
            // floor fan-in as a window of kernel and stride `num_outputs`:
            // `num_outputs` inputs share each output. Same convention — the
            // smallest, which is the one that divided evenly. A rank-0 shape
            // has no axis to split, and the forward pass passes it through.
            Op::Split { axis, num_outputs } => {
                let rank = output.len();
                if rank == 0 {
                    return Ok(Some(vec![output.clone()]));
                }
                let ax = axis.rem_euclid(rank as i64) as usize;
                let parts = *num_outputs.max(&1);
                let mut input = output.clone();
                input[ax] = output[ax]
                    .map(|d| -> Result<usize> {
                        d.checked_mul(parts).ok_or_else(|| {
                            Error::ShapeExtentOverflow {
                                op: "Split".into(),
                                axis: ax,
                                extent: d,
                                factor: parts,
                            }
                            .into()
                        })
                    })
                    .transpose()?;
                Ok(Some(vec![input]))
            }

            other => unimplemented!("Op::infer_input_shape is not implemented for {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    use crate::graph::SymTensor;
    use crate::nn::{
        Layer,
        activation::{relu::Relu, softmax::Softmax},
        conv2d::Conv2d,
        linear::Linear,
    };

    // Unit tests call `infer_output_shape` directly, which is the only way to
    // reach the multi-input arms: `SymTensor::record` always passes a single
    // input shape, so `Concat`, `MatMul`, `Gemm` and `ChannelCat` are
    // unreachable through the layer API.

    #[test]
    fn elementwise_ops_preserve_the_input_shape() {
        let input = vec![None, Some(16), Some(8), Some(8)];
        assert_eq!(Op::Relu.infer_output_shape(&[&input]).unwrap(), input);
        assert_eq!(Op::Sigmoid.infer_output_shape(&[&input]).unwrap(), input);
    }

    #[test]
    fn linear_replaces_only_the_last_dim() {
        let input = vec![None, Some(784)];
        let op = Op::Linear {
            in_features: 784,
            out_features: 128,
            has_bias: true,
        };
        assert_eq!(
            op.infer_output_shape(&[&input]).unwrap(),
            vec![None, Some(128)]
        );
    }

    #[test]
    fn flatten_folds_the_trailing_dims_and_keeps_the_batch_axis() {
        let input = vec![None, Some(16), Some(5), Some(5)];
        assert_eq!(
            Op::Flatten.infer_output_shape(&[&input]).unwrap(),
            vec![None, Some(400)]
        );
    }

    #[test]
    fn flatten_is_unknown_when_any_folded_dim_is_dynamic() {
        let input = vec![Some(2), Some(16), None, Some(5)];
        assert_eq!(
            Op::Flatten.infer_output_shape(&[&input]).unwrap(),
            vec![Some(2), None]
        );
    }

    #[test]
    fn conv2d_applies_the_window_arithmetic_per_spatial_axis() {
        let input = vec![None, Some(3), Some(32), Some(32)];
        let op = Op::Conv2d {
            in_channels: 3,
            out_channels: 64,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 1,
            padding_h: 1,
            padding_w: 1,
            groups: 1,
            has_bias: true,
        };
        assert_eq!(
            op.infer_output_shape(&[&input]).unwrap(),
            vec![None, Some(64), Some(16), Some(32)]
        );
    }

    #[test]
    fn global_avg_pool_collapses_the_spatial_dims_to_one() {
        let input = vec![None, Some(16), Some(7), Some(7)];
        assert_eq!(
            Op::GlobalAvgPool.infer_output_shape(&[&input]).unwrap(),
            vec![None, Some(16), Some(1), Some(1)]
        );
    }

    #[test]
    fn concat_sums_the_axis_extent_across_every_input() {
        let a = vec![Some(2), Some(3), Some(8)];
        let b = vec![Some(2), Some(5), Some(8)];
        let c = vec![Some(2), Some(7), Some(8)];
        let op = Op::Concat { axis: 1 };
        assert_eq!(
            op.infer_output_shape(&[&a, &b, &c]).unwrap(),
            vec![Some(2), Some(15), Some(8)]
        );
    }

    #[test]
    fn concat_axis_is_unknown_when_any_input_is_dynamic() {
        let a = vec![Some(2), Some(3)];
        let b = vec![Some(2), None];
        let op = Op::Concat { axis: -1 };
        assert_eq!(
            op.infer_output_shape(&[&a, &b]).unwrap(),
            vec![Some(2), None]
        );
    }

    #[test]
    fn matmul_takes_the_trailing_dim_from_the_second_input() {
        let a = vec![None, Some(2), Some(3)];
        let b = vec![Some(3), Some(4)];
        assert_eq!(
            Op::MatMul.infer_output_shape(&[&a, &b]).unwrap(),
            vec![None, Some(2), Some(4)]
        );
    }

    #[test]
    fn gemm_reads_m_and_n_transpose_aware() {
        let a = vec![Some(3), Some(2)];
        let b = vec![Some(5), Some(3)];
        let op = Op::Gemm {
            alpha: 1.0,
            beta: 1.0,
            trans_a: true,
            trans_b: true,
        };
        // trans_a takes M from a[1], trans_b takes N from b[0].
        assert_eq!(
            op.infer_output_shape(&[&a, &b]).unwrap(),
            vec![Some(2), Some(5)]
        );
    }

    #[test]
    fn channel_cat_takes_the_channel_count_from_its_config() {
        let a = vec![Some(1), Some(4), Some(8), Some(8)];
        let b = vec![Some(1), Some(6), Some(8), Some(8)];
        let op = Op::ChannelCat { c_total: 10 };
        assert_eq!(
            op.infer_output_shape(&[&a, &b]).unwrap(),
            vec![Some(1), Some(10), Some(8), Some(8)]
        );
    }

    #[test]
    fn split_divides_the_axis_by_the_output_count() {
        let input = vec![Some(2), Some(12), Some(8)];
        let op = Op::Split {
            axis: 1,
            num_outputs: 4,
        };
        assert_eq!(
            op.infer_output_shape(&[&input]).unwrap(),
            vec![Some(2), Some(3), Some(8)]
        );
    }

    #[test]
    fn constant_carries_its_own_shape_and_ignores_its_inputs() {
        let op = Op::Constant {
            shape: vec![Some(4), Some(4)],
            dtype: DtypeRepr::F32,
        };
        assert_eq!(op.infer_output_shape(&[]).unwrap(), vec![Some(4), Some(4)]);
    }

    #[test]
    fn unary_elementwise_math_reverses_to_the_identity() {
        let output = vec![None, Some(8), Some(4)];
        for op in [
            Op::Abs,
            Op::Neg,
            Op::Sqrt,
            Op::Exp,
            Op::Log,
            Op::Erf,
            Op::Sin,
            Op::Not,
            Op::Identity,
            Op::Hardmax { axis: 1 },
        ] {
            assert_eq!(
                op.infer_input_shape(&output).unwrap(),
                Some(vec![output.clone()]),
                "{op:?} should reverse to the identity"
            );
        }
    }

    /// Multi-input ops stay unimplemented: one returned shape would misreport
    /// how many inputs the node has.
    #[test]
    #[should_panic(expected = "not implemented for Add")]
    fn multi_input_elementwise_is_still_unimplemented() {
        let _ = Op::Add.infer_input_shape(&vec![Some(2), Some(2)]);
    }

    // --- window errors ---------------------------------------------------

    fn conv2d(kernel: usize, stride: usize, padding: usize) -> Op {
        Op::Conv2d {
            in_channels: 3,
            out_channels: 8,
            kernel_h: kernel,
            kernel_w: kernel,
            stride_h: stride,
            stride_w: stride,
            padding_h: padding,
            padding_w: padding,
            groups: 1,
            has_bias: false,
        }
    }

    #[test]
    fn a_window_that_cannot_fit_is_an_error_not_a_panic() {
        let input = vec![Some(1), Some(3), Some(4), Some(4)];
        let err = conv2d(7, 1, 1).infer_output_shape(&[&input]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Conv2d: height kernel 7 does not fit its input"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("raise the height padding to at least 2"),
            "error should say how to fix it: {msg}"
        );
    }

    #[test]
    fn a_zero_stride_is_an_error_not_a_panic() {
        let input = vec![Some(1), Some(3), Some(8), Some(8)];
        let err = conv2d(3, 0, 0).infer_output_shape(&[&input]).unwrap_err();
        assert!(
            err.to_string().contains("Conv2d: height stride is 0"),
            "unexpected error: {err}"
        );
    }

    /// The context travels as typed fields, so a caller can inspect the
    /// failure rather than parse the message.
    #[test]
    fn window_errors_carry_their_context_as_fields() {
        let input = vec![Some(1), Some(3), Some(4), Some(4)];
        let err = conv2d(7, 1, 1).infer_output_shape(&[&input]).unwrap_err();
        match err.downcast_ref::<Error>() {
            Some(Error::WindowDoesNotFit {
                op,
                axis,
                extent,
                kernel,
                padding,
            }) => {
                assert_eq!(op, "Conv2d");
                assert_eq!(axis, "height");
                assert_eq!(*extent, 4);
                assert_eq!(*kernel, 7);
                assert_eq!(*padding, 1);
            }
            other => panic!("expected WindowDoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn a_window_that_exactly_fills_its_padded_input_is_not_an_error() {
        let input = vec![Some(1), Some(3), Some(4), Some(4)];
        assert_eq!(
            conv2d(6, 1, 1).infer_output_shape(&[&input]).unwrap(),
            vec![Some(1), Some(8), Some(1), Some(1)]
        );
    }

    // --- infer_input_shape (reverse) -------------------------------------

    #[test]
    fn pointwise_reverse_is_the_identity() {
        let output = vec![Some(2), Some(16), Some(8), Some(8)];
        for op in [Op::Relu, Op::Sigmoid, Op::Silu, Op::Tanh, Op::Gelu] {
            assert_eq!(
                op.infer_input_shape(&output).unwrap(),
                Some(vec![output.clone()])
            );
        }
    }

    #[test]
    fn pointwise_reverse_keeps_dynamic_dims_dynamic() {
        let output = vec![None, Some(128)];
        assert_eq!(
            Op::Relu.infer_input_shape(&output).unwrap(),
            Some(vec![vec![None, Some(128)]])
        );
    }

    #[test]
    fn shape_preserving_ops_with_config_also_reverse_to_identity() {
        let output = vec![None, Some(16), Some(8), Some(8)];
        let batchnorm = Op::BatchNorm2d {
            num_features: 16,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        };
        let softmax = Op::Softmax { dim: 1 };
        assert_eq!(
            batchnorm.infer_input_shape(&output).unwrap(),
            Some(vec![output.clone()])
        );
        assert_eq!(
            softmax.infer_input_shape(&output).unwrap(),
            Some(vec![output])
        );
    }

    #[test]
    fn input_placeholder_has_no_producers_to_infer() {
        let out = Op::Input.infer_input_shape(&vec![None, Some(3)]).unwrap();
        assert_eq!(out, Some(vec![]));
    }

    /// The reverse of the forward pass round-trips for every pointwise op.
    #[test]
    fn pointwise_round_trips_through_both_directions() {
        let input = vec![None, Some(16), Some(8), Some(8)];
        let op = Op::Relu;
        let output = op.infer_output_shape(&[&input]).unwrap();
        assert_eq!(op.infer_input_shape(&output).unwrap(), Some(vec![input]));
    }

    /// The closed form, swept directly over the helper pair: for every
    /// `(kernel, stride, padding)` and every reachable output, the reverse is
    /// a genuine pre-image, and it is the smallest one.
    #[test]
    fn window_in_dim_is_the_smallest_pre_image_of_window_out_dim() {
        for kernel in 1..8 {
            for stride in 1..6 {
                for padding in 0..4 {
                    for out in 1..60 {
                        let Ok(extent) = window_in_dim("op", "axis", out, kernel, stride, padding)
                        else {
                            continue; // `out` below the smallest this config emits.
                        };
                        assert_eq!(
                            window_out_dim("op", "axis", extent, kernel, stride, padding).unwrap(),
                            out,
                            "k={kernel} s={stride} p={padding} out={out}: {extent} is not a pre-image"
                        );
                        // Smallest: one less either underflows the window or
                        // lands on a smaller output.
                        if extent > 0 {
                            let below =
                                window_out_dim("op", "axis", extent - 1, kernel, stride, padding);
                            assert!(
                                below.is_err() || below.unwrap() < out,
                                "k={kernel} s={stride} p={padding} out={out}: {extent} is not the smallest"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every sliding-window op covered by the reverse pass, built from one
    /// `(kernel, stride, padding)` triple, paired with the input shape it was
    /// sized for. Only the convolutions and `MaxPool2d` carry padding; the
    /// other pools always window their input unpadded.
    fn window_ops(kernel: usize, stride: usize, padding: usize, extent: usize) -> Vec<(Op, Shape)> {
        let (n, c, e) = (Some(2), Some(3), Some(extent));
        let (k, s, p) = (kernel, stride, padding);
        vec![
            (
                Op::Conv1d {
                    in_channels: 3,
                    out_channels: 8,
                    kernel_l: k,
                    stride: s,
                    padding: p,
                    has_bias: true,
                },
                vec![n, c, e],
            ),
            (
                Op::Conv2d {
                    in_channels: 3,
                    out_channels: 8,
                    kernel_h: k,
                    kernel_w: k,
                    stride_h: s,
                    stride_w: s,
                    padding_h: p,
                    padding_w: p,
                    groups: 1,
                    has_bias: true,
                },
                vec![n, c, e, e],
            ),
            (
                Op::Conv3d {
                    in_channels: 3,
                    out_channels: 8,
                    kernel_d: k,
                    kernel_h: k,
                    kernel_w: k,
                    stride_d: s,
                    stride_h: s,
                    stride_w: s,
                    padding_d: p,
                    padding_h: p,
                    padding_w: p,
                    has_bias: true,
                },
                vec![n, c, e, e, e],
            ),
            (
                Op::AvgPool1d {
                    kernel_l: k,
                    stride: s,
                },
                vec![n, c, e],
            ),
            (
                Op::MaxPool1d {
                    kernel_l: k,
                    stride: s,
                },
                vec![n, c, e],
            ),
            (
                Op::LpPool1d {
                    kernel_l: k,
                    stride: s,
                    p: 2.0,
                },
                vec![n, c, e],
            ),
            (
                Op::AvgPool2d {
                    kernel_h: k,
                    kernel_w: k,
                    stride_h: s,
                    stride_w: s,
                },
                vec![n, c, e, e],
            ),
            (
                Op::MaxPool2d {
                    kernel_h: k,
                    kernel_w: k,
                    stride_h: s,
                    stride_w: s,
                    pad_h: p,
                    pad_w: p,
                },
                vec![n, c, e, e],
            ),
            (
                Op::LpPool2d {
                    kernel_h: k,
                    kernel_w: k,
                    stride_h: s,
                    stride_w: s,
                    p: 2.0,
                },
                vec![n, c, e, e],
            ),
            (
                Op::AvgPool3d {
                    kernel_d: k,
                    kernel_h: k,
                    kernel_w: k,
                    stride_d: s,
                    stride_h: s,
                    stride_w: s,
                },
                vec![n, c, e, e, e],
            ),
            (
                Op::MaxPool3d {
                    kernel_d: k,
                    kernel_h: k,
                    kernel_w: k,
                    stride_d: s,
                    stride_h: s,
                    stride_w: s,
                },
                vec![n, c, e, e, e],
            ),
            (
                Op::LpPool3d {
                    kernel_d: k,
                    kernel_h: k,
                    kernel_w: k,
                    stride_d: s,
                    stride_h: s,
                    stride_w: s,
                    p: 2.0,
                },
                vec![n, c, e, e, e],
            ),
        ]
    }

    /// Unit stride is the unambiguous case: one input produces each output, so
    /// the reverse recovers the original shape exactly, for every window op.
    #[test]
    fn unit_stride_window_reverse_round_trips_exactly() {
        for (op, input) in window_ops(3, 1, 1, 32) {
            let output = op.infer_output_shape(&[&input]).unwrap();
            assert_eq!(
                op.infer_input_shape(&output).unwrap(),
                Some(vec![input.clone()]),
                "unit-stride reverse lost information for {op:?}"
            );
        }
    }

    /// With a stride above 1 the fan-in is real, so the reverse is only
    /// required to land on an input the forward pass maps back to `output`.
    #[test]
    fn strided_window_reverse_re_runs_forward_to_the_same_output() {
        for stride in [2, 3, 4] {
            for (op, input) in window_ops(3, stride, 1, 32) {
                let output = op.infer_output_shape(&[&input]).unwrap();
                let reversed = op.infer_input_shape(&output).unwrap().unwrap();
                assert_eq!(
                    op.infer_output_shape(&[&reversed[0]]).unwrap(),
                    output,
                    "reversed shape does not re-run forward for {op:?}"
                );
            }
        }
    }

    /// The convention: of the `stride` inputs that produce a given output, the
    /// reverse returns the smallest. The bead's worked example is `k = 3,
    /// s = 2, p = 1`, where both 15 and 16 produce an output of 8.
    #[test]
    fn strided_reverse_returns_the_smallest_input_of_the_fan_in() {
        let op = Op::MaxPool2d {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
        };
        let output = vec![Some(2), Some(3), Some(8), Some(8)];
        assert_eq!(
            op.infer_input_shape(&output).unwrap(),
            Some(vec![vec![Some(2), Some(3), Some(15), Some(15)]])
        );
        // The larger member of the fan-in is a real input, just not the one
        // the convention picks.
        let larger = vec![Some(2), Some(3), Some(16), Some(16)];
        assert_eq!(op.infer_output_shape(&[&larger]).unwrap(), output);
    }

    /// A convolution's channel count comes from the op, not the output: the
    /// forward pass replaced `in_channels` with `out_channels`.
    #[test]
    fn conv_reverse_restores_the_declared_input_channels() {
        let op = Op::Conv2d {
            in_channels: 3,
            out_channels: 64,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 1,
            padding_w: 1,
            groups: 1,
            has_bias: true,
        };
        let output = vec![None, Some(64), Some(32), Some(32)];
        assert_eq!(
            op.infer_input_shape(&output).unwrap(),
            Some(vec![vec![None, Some(3), Some(32), Some(32)]])
        );
    }

    /// An unknown spatial extent stays unknown; there is nothing to reverse.
    #[test]
    fn window_reverse_keeps_dynamic_dims_dynamic() {
        for (op, input) in window_ops(3, 2, 1, 32) {
            let mut output = op.infer_output_shape(&[&input]).unwrap();
            let last = output.len() - 1;
            output[last] = None;
            let reversed = op.infer_input_shape(&output).unwrap().unwrap();
            assert_eq!(reversed[0][last], None, "dynamic dim resolved for {op:?}");
        }
    }

    /// The window guards are unreachable from the reverse — every shape it
    /// returns fits its own kernel — but a caller can still hand it a shape
    /// the forward pass never produced. That is an error, not a panic.
    #[test]
    fn window_reverse_rejects_an_output_extent_it_could_not_have_produced() {
        // Padding 5 around a kernel of 3 means the smallest output is 8.
        let op = Op::Conv1d {
            in_channels: 3,
            out_channels: 8,
            kernel_l: 3,
            stride: 1,
            padding: 5,
            has_bias: true,
        };
        let err = op
            .infer_input_shape(&vec![Some(2), Some(8), Some(1)])
            .expect_err("output extent 1 is below the smallest this op emits");
        let msg = err.to_string();
        assert!(msg.contains("length"), "{msg}");
        assert!(
            msg.contains("smallest output this op can produce is 8"),
            "{msg}"
        );
    }

    /// A zero-extent output is unreachable for the same reason: a window op
    /// always emits at least one position.
    #[test]
    fn window_reverse_rejects_a_zero_output_extent() {
        let op = Op::AvgPool1d {
            kernel_l: 2,
            stride: 2,
        };
        assert!(
            op.infer_input_shape(&vec![Some(2), Some(3), Some(0)])
                .is_err()
        );
    }

    /// A zero stride is rejected rather than dividing or multiplying by zero.
    #[test]
    fn window_reverse_rejects_a_zero_stride() {
        let op = Op::MaxPool1d {
            kernel_l: 2,
            stride: 0,
        };
        let err = op
            .infer_input_shape(&vec![Some(2), Some(3), Some(4)])
            .expect_err("a zero stride has no reverse");
        assert!(err.to_string().contains("stride is 0"), "{err}");
    }

    /// Indexing the spatial dims is guarded by a rank check.
    #[test]
    fn window_reverse_rejects_a_wrong_rank_output() {
        for (op, input) in window_ops(3, 1, 1, 32) {
            let rank = input.len();
            let short = vec![Some(2), Some(3)];
            let err = op
                .infer_input_shape(&short)
                .expect_err("rank 2 is not a shape this op produces");
            let msg = err.to_string();
            assert!(msg.contains(&alloc::format!("rank-{rank} shape")), "{msg}");
        }
    }

    /// `Split` divides one axis, so its reverse multiplies it back — landing on
    /// the input that divided evenly, the smallest of the `num_outputs` inputs
    /// that share this output.
    #[test]
    fn split_reverse_multiplies_the_axis_back_out() {
        let op = Op::Split {
            axis: 1,
            num_outputs: 4,
        };
        let input = vec![Some(2), Some(32), Some(8)];
        let output = op.infer_output_shape(&[&input]).unwrap();
        assert_eq!(output, vec![Some(2), Some(8), Some(8)]);
        assert_eq!(op.infer_input_shape(&output).unwrap(), Some(vec![input]));
    }

    /// A negative axis counts from the end, in both directions.
    #[test]
    fn split_reverse_resolves_a_negative_axis() {
        let op = Op::Split {
            axis: -1,
            num_outputs: 2,
        };
        let output = vec![Some(2), Some(6)];
        assert_eq!(
            op.infer_input_shape(&output).unwrap(),
            Some(vec![vec![Some(2), Some(12)]])
        );
    }

    /// An uneven split loses the remainder, so the reverse lands on the
    /// even input rather than the original — the documented convention.
    #[test]
    fn split_reverse_returns_the_even_input_of_an_uneven_split() {
        let op = Op::Split {
            axis: 0,
            num_outputs: 3,
        };
        let input = vec![Some(10)];
        let output = op.infer_output_shape(&[&input]).unwrap();
        assert_eq!(output, vec![Some(3)]);
        let reversed = op.infer_input_shape(&output).unwrap().unwrap();
        assert_eq!(reversed[0], vec![Some(9)]);
        // Still a valid answer: it re-runs forward to the same output.
        assert_eq!(op.infer_output_shape(&[&reversed[0]]).unwrap(), output);
    }

    /// A rank-0 shape has no axis to split; the forward pass passes it through
    /// untouched, and so does the reverse.
    #[test]
    fn split_reverse_passes_a_rank_zero_shape_through() {
        let op = Op::Split {
            axis: 0,
            num_outputs: 2,
        };
        assert_eq!(op.infer_input_shape(&vec![]).unwrap(), Some(vec![vec![]]));
    }

    #[test]
    #[should_panic(expected = "Op::infer_input_shape is not implemented for Flatten")]
    fn unimplemented_ops_panic_naming_the_variant() {
        let _ = Op::Flatten.infer_input_shape(&vec![Some(2), Some(400)]);
    }

    /// A custom op that can invert itself: shape-preserving, like a fused
    /// activation.
    #[derive(Debug)]
    struct PassThroughOp;

    impl CustomOp for PassThroughOp {
        fn name(&self) -> &str {
            "test.pass_through"
        }

        fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape> {
            Ok(input_shapes[0].clone())
        }

        fn infer_input_shape(&self, output_shape: &Shape) -> Result<Option<Vec<Shape>>> {
            Ok(Some(vec![output_shape.clone()]))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A custom op that rejects output shapes it could not have produced.
    #[derive(Debug)]
    struct EvenRankOnlyOp;

    impl CustomOp for EvenRankOnlyOp {
        fn name(&self) -> &str {
            "test.even_rank_only"
        }

        fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape> {
            if !input_shapes[0].len().is_multiple_of(2) {
                return Err(anyhow::anyhow!(
                    "test.even_rank_only: rank {} is odd",
                    input_shapes[0].len()
                ));
            }
            Ok(input_shapes[0].clone())
        }

        fn infer_input_shape(&self, output_shape: &Shape) -> Result<Option<Vec<Shape>>> {
            if !output_shape.len().is_multiple_of(2) {
                return Err(anyhow::anyhow!(
                    "test.even_rank_only: rank {} is odd",
                    output_shape.len()
                ));
            }
            Ok(Some(vec![output_shape.clone()]))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A custom op that declines to invert, which it must now say explicitly.
    #[derive(Debug)]
    struct OpaqueOp;

    impl CustomOp for OpaqueOp {
        fn name(&self) -> &str {
            "test.opaque"
        }

        fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape> {
            Ok(input_shapes[0].clone())
        }

        fn infer_input_shape(&self, _output_shape: &Shape) -> Result<Option<Vec<Shape>>> {
            Ok(None)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn custom_op_forward_errors_propagate_through_op() {
        let op = Op::Custom {
            data: Arc::new(EvenRankOnlyOp),
        };
        let odd = vec![Some(2), Some(3), Some(4)];
        let err = op.infer_output_shape(&[&odd]).unwrap_err().to_string();
        assert!(err.contains("rank 3 is odd"), "unexpected error: {err}");

        let even = vec![Some(2), Some(4)];
        assert_eq!(op.infer_output_shape(&[&even]).unwrap(), even);
    }

    #[test]
    fn custom_op_reverse_is_delegated_to_the_trait() {
        let output = vec![None, Some(16), Some(8), Some(8)];
        let op = Op::Custom {
            data: Arc::new(PassThroughOp),
        };
        assert_eq!(
            op.infer_input_shape(&output).unwrap(),
            Some(vec![output.clone()])
        );
    }

    #[test]
    fn custom_op_errors_propagate_unchanged() {
        let op = Op::Custom {
            data: Arc::new(EvenRankOnlyOp),
        };
        let err = op
            .infer_input_shape(&vec![Some(2), Some(3), Some(4)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank 3 is odd"), "unexpected error: {err}");
    }

    #[test]
    fn custom_op_declining_to_invert_reports_none_not_an_error() {
        let op = Op::Custom {
            data: Arc::new(OpaqueOp),
        };
        assert_eq!(op.infer_input_shape(&vec![Some(2), Some(4)]).unwrap(), None);
    }

    /// End-to-end cover for the inference arms an actual model walks through.
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
