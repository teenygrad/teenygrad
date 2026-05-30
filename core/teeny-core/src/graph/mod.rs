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
        groupnorm::GroupNorm,
        instancenorm::{InstanceNorm1d, InstanceNorm2d, InstanceNorm3d},
        layernorm::LayerNorm,
        rmsnorm::RmsNorm,
        conv2d::Conv2d,
        conv3d::Conv3d,
        flatten::Flatten,
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
    },
};

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DtypeRepr {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
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
    pub fn new<T: CustomOp>(op: T) -> Self {
        Self(Arc::new(op))
    }

    pub fn name(&self) -> &str {
        self.0.name()
    }

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

#[derive(Debug, Clone)]
pub enum Op {
    /// Model input placeholder.
    Input,

    // --- Linear / MLP ---
    Linear {
        in_features: usize,
        out_features: usize,
        has_bias: bool,
    },
    Flatten,

    // --- Normalisation ---
    BatchNorm1d {
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    },
    BatchNorm2d {
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    },
    BatchNorm3d {
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    },
    LayerNorm {
        normalized_shape: alloc::vec::Vec<usize>,
        eps: f64,
        affine: bool,
    },
    RmsNorm {
        normalized_shape: alloc::vec::Vec<usize>,
        eps: f64,
        affine: bool,
    },
    GroupNorm {
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        affine: bool,
    },
    InstanceNorm1d {
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    },
    InstanceNorm2d {
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    },
    InstanceNorm3d {
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    },

    // --- Convolution ---
    Conv1d {
        in_channels: usize,
        out_channels: usize,
        kernel_l: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
    },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        groups: usize,
        has_bias: bool,
    },
    Conv3d {
        in_channels: usize,
        out_channels: usize,
        kernel_d: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_d: usize,
        stride_h: usize,
        stride_w: usize,
        padding_d: usize,
        padding_h: usize,
        padding_w: usize,
        has_bias: bool,
    },

    /// Fused Conv2d + BatchNorm2d (inference-only) + SiLU forward.
    ///
    /// The BN parameters (scale, shift) are stored as precomputed affine
    /// constants — not raw mean/var/gamma/beta.  Produced by `Graph::optimise()`
    /// when it detects the pattern `Conv2d(no bias) → BatchNorm2d → Silu`.
    ///
    /// `bn_eps` is carried forward only for reference; the BN affine constants
    /// are passed at runtime via the `bn_scale` and `bn_shift` parameters.
    Conv2dBnSilu {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        groups: usize,
        bn_eps: f64,
    },

    // --- Pooling ---
    AvgPool1d { kernel_l: usize, stride: usize },
    AvgPool2d { kernel_h: usize, kernel_w: usize, stride_h: usize, stride_w: usize },
    AvgPool3d { kernel_d: usize, kernel_h: usize, kernel_w: usize, stride_d: usize, stride_h: usize, stride_w: usize },
    MaxPool1d { kernel_l: usize, stride: usize },
    MaxPool2d { kernel_h: usize, kernel_w: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize },
    MaxPool3d { kernel_d: usize, kernel_h: usize, kernel_w: usize, stride_d: usize, stride_h: usize, stride_w: usize },
    LpPool1d { kernel_l: usize, stride: usize, p: f64 },
    LpPool2d { kernel_h: usize, kernel_w: usize, stride_h: usize, stride_w: usize, p: f64 },
    LpPool3d { kernel_d: usize, kernel_h: usize, kernel_w: usize, stride_d: usize, stride_h: usize, stride_w: usize, p: f64 },

    // --- Upsample ---
    /// Nearest-neighbour 2-D upsampling.
    /// Output shape: `[N, C, H * scale_h, W * scale_w]`.
    UpsampleNearest2d {
        scale_h: usize,
        scale_w: usize,
    },

    // --- Padding ---
    ConstantPad1d { pad_left: usize, pad_right: usize, value: f64 },
    ConstantPad2d { pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize, value: f64 },
    ConstantPad3d { pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize, value: f64 },
    ReflectionPad1d { pad_left: usize, pad_right: usize },
    ReflectionPad2d { pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize },
    ReflectionPad3d { pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize },
    ReplicationPad1d { pad_left: usize, pad_right: usize },
    ReplicationPad2d { pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize },
    ReplicationPad3d { pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize },
    CircularPad1d { pad_left: usize, pad_right: usize },
    CircularPad2d { pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize },
    CircularPad3d { pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize },

    // --- Activation ---
    Relu,
    Elu { alpha: f64 },
    Selu,
    Celu { alpha: f64 },
    Gelu,
    Mish,
    Hardtanh { min_val: f64, max_val: f64 },
    Relu6,
    Hardsigmoid,
    Hardswish,
    Hardshrink { lambda: f64 },
    LeakyRelu { negative_slope: f64 },
    Threshold { threshold: f64, value: f64 },
    Softsign,
    Softshrink { lambda: f64 },
    Softplus { beta: f64, threshold: f64 },
    Sigmoid,
    Silu,
    Logsigmoid,
    Tanh,
    Tanhshrink,
    Softmax { dim: usize },

    // --- Attention ---
    /// Multi-head self-attention with Flash Attention 2 and position encoding.
    /// Represents the full `Attention.forward()` in PSABlock:
    ///   qkv conv → FA2 → pe depthwise conv → proj conv → residual add.
    /// Input/output shape: `[N, c, H, W]`.
    Attention {
        c:         usize,
        num_heads: usize,
        key_dim:   usize,
    },

    // --- Tensor structural ops ---
    /// Element-wise addition of two tensors with identical shapes.
    Add,
    /// Extract one contiguous channel slice from a 4-D NCHW tensor.
    /// Output shape: `[N, chunk_c, H, W]`.
    ChannelChunk {
        c_total: usize,
        chunk_c: usize,
        chunk_offset: usize,
    },
    /// Concatenate N 4-D NCHW tensors along the channel dimension.
    /// Output shape: `[N, c_total, H, W]`.
    ChannelCat {
        c_total: usize,
    },
    /// Adds a (C,) bias vector to a (B, C, H, W) feature map — NC layout (N=B*H*W).
    /// Output shape equals input shape.
    ChannelBiasAdd { c: usize },

    /// User-defined op.  Shape and dtype must be provided via [`Graph::add_node`]
    /// or [`SymTensor::record_custom`] — the base system cannot infer them.
    Custom {
        data: CustomData,
    },

    // -----------------------------------------------------------------------
    // ONNX-sourced ops — added to let the ONNX loader build a complete graph.
    // Triton/CPU lowering is not yet implemented for these variants.
    // -----------------------------------------------------------------------

    // --- Element-wise unary math ---
    Abs,
    Neg,
    Ceil,
    Floor,
    Round,
    Sqrt,
    Reciprocal,
    Exp,
    Log,
    Erf,
    Sign,
    IsNaN,
    IsInf { detect_negative: bool, detect_positive: bool },
    Not,
    BitwiseNot,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,

    // --- Element-wise binary / variadic ---
    Mul,
    Sub,
    Div,
    Pow,
    Mod { fmod: bool },
    ElemMin,
    ElemMax,
    ElemMean,
    ElemSum,
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    And,
    Or,
    Xor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitShift { direction: alloc::string::String },

    // --- Tensor structural ---
    Reshape,
    Transpose { perm: alloc::vec::Vec<usize> },
    Squeeze { axes: alloc::vec::Vec<i64> },
    Unsqueeze { axes: alloc::vec::Vec<i64> },
    Concat { axis: i64 },
    Split { axis: i64, num_outputs: usize },
    Slice,
    Gather { axis: i64 },
    GatherElements { axis: i64 },
    GatherND { batch_dims: i64 },
    ScatterElements { axis: i64 },
    ScatterND,
    Tile,
    Expand,
    ShapeOf { start: i64, end: i64 },
    SizeOf,
    Identity,
    Cast { to: DtypeRepr },
    CastLike,
    Where,
    Compress { axis: i64 },
    Range,
    /// Constant tensor (value embedded in the ONNX model).
    Constant { dtype: DtypeRepr, shape: Shape },
    ConstantOfShape { dtype: DtypeRepr },
    Trilu { upper: bool },
    BitCast { to: DtypeRepr },
    Pad { mode: alloc::string::String },
    ReverseSequence { batch_axis: i64, time_axis: i64 },
    NonZero,
    Scatter { axis: i64 },
    TensorScatter,

    // --- Matrix ---
    Gemm { alpha: f64, beta: f64, trans_a: bool, trans_b: bool },
    MatMul,
    MatMulInteger,
    Einsum { equation: alloc::string::String },
    Det,
    QLinearMatMul,

    // --- Convolution extras ---
    ConvTranspose {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        output_padding_h: usize,
        output_padding_w: usize,
        groups: usize,
        has_bias: bool,
    },
    ConvInteger { groups: usize },
    DeformConv { group: usize, offset_group: usize },
    QLinearConv { groups: usize },
    Col2Im { kernel_h: usize, kernel_w: usize },

    // --- Reductions ---
    ReduceSum { keepdims: bool, noop_with_empty_axes: bool },
    ReduceMean { keepdims: bool, noop_with_empty_axes: bool },
    ReduceMax { keepdims: bool, noop_with_empty_axes: bool },
    ReduceMin { keepdims: bool, noop_with_empty_axes: bool },
    ReduceProd { keepdims: bool, noop_with_empty_axes: bool },
    ReduceL1 { keepdims: bool, noop_with_empty_axes: bool },
    ReduceL2 { keepdims: bool, noop_with_empty_axes: bool },
    ReduceLogSum { keepdims: bool, noop_with_empty_axes: bool },
    ReduceLogSumExp { keepdims: bool, noop_with_empty_axes: bool },
    ReduceSumSquare { keepdims: bool, noop_with_empty_axes: bool },
    CumSum { exclusive: bool, reverse: bool },
    CumProd { exclusive: bool, reverse: bool },
    ArgMax { axis: i64, keepdims: bool, select_last_index: bool },
    ArgMin { axis: i64, keepdims: bool, select_last_index: bool },
    GlobalAvgPool,
    GlobalMaxPool,
    LpNormalization { axis: i64, p: i64 },
    MeanVarianceNormalization { axes: alloc::vec::Vec<i64> },

    // --- Additional activations ---
    LogSoftmax { axis: i64 },
    Hardmax { axis: i64 },
    PRelu,
    ThresholdedRelu { alpha: f64 },
    Shrink { lambd: f64, bias: f64 },
    Clip,
    Swish,
    MultiHeadAttention { q_num_heads: usize, kv_num_heads: usize },

    // --- Normalisation (generic) ---
    LRN { alpha: f64, beta: f64, bias: f64, size: usize },

    // --- Recurrent ---
    Lstm { hidden_size: usize, direction: alloc::string::String, bidirectional: bool },
    Gru { hidden_size: usize, direction: alloc::string::String, bidirectional: bool },
    Rnn { hidden_size: usize, direction: alloc::string::String, bidirectional: bool },

    // --- Resize / spatial ---
    Resize {
        mode: alloc::string::String,
        coordinate_transformation_mode: alloc::string::String,
        antialias: bool,
    },
    GridSample { mode: alloc::string::String, padding_mode: alloc::string::String, align_corners: bool },
    SpaceToDepth { blocksize: usize },
    DepthToSpace { blocksize: usize, mode: alloc::string::String },
    RoiAlign { output_h: usize, output_w: usize, sampling_ratio: i64, spatial_scale: f64 },
    AffineGrid { align_corners: bool },
    MaxUnpool { kernel_h: usize, kernel_w: usize, stride_h: usize, stride_w: usize },
    CenterCropPad { axes: alloc::vec::Vec<i64> },
    NonMaxSuppression { center_point_box: bool },

    // --- Misc ---
    TopK { axis: i64, largest: bool, sorted: bool },
    Unique { sorted: bool },
    Dropout { training_mode: bool },
    EyeLike { dtype: Option<DtypeRepr>, k: i64 },
    OneHot { axis: i64 },
    Bernoulli { dtype: Option<DtypeRepr> },
    RandomUniformLike { dtype: Option<DtypeRepr>, high: f64, low: f64 },
    RotaryEmbedding,

    // --- Quantisation ---
    QuantizeLinear { axis: i64, saturate: bool },
    DequantizeLinear { axis: i64 },
    DynamicQuantizeLinear,

    // --- Signal ---
    Dft { inverse: bool, onesided: bool },
    Stft,
    MelWeightMatrix,
    HannWindow { periodic: bool },
    BlackmanWindow { periodic: bool },
    HammingWindow { periodic: bool },

    // --- Loss ---
    NegativeLogLikelihoodLoss { reduction: alloc::string::String },
    SoftmaxCrossEntropyLoss { reduction: alloc::string::String },

    // --- Sequences ---
    SequenceAt,
    SequenceConstruct,
    SequenceEmpty,
    SequenceErase,
    SequenceInsert,
    SequenceLength,
    SequenceMap,
    SplitToSequence { axis: i64, keepdims: bool },
    ConcatFromSequence { axis: i64, new_axis: bool },
    OptionalGetElement,
    OptionalHasElement,

    // --- Control flow ---
    Loop,
    Scan { num_scan_inputs: i64 },
    If,

    // --- Optimiser ops ---
    Adagrad,
    Adam,
    Momentum,
    Gradient,

    // --- String / NLP ---
    StringNormalizer,
    RegexFullMatch { pattern: alloc::string::String },
    StringConcat,
    StringSplit,
    TfIdfVectorizer,
    LabelEncoder,

    // --- Other ML ---
    ArrayFeatureExtractor,
    Binarizer { threshold: f64 },
    TreeEnsemble,
    ImageDecoder,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub op: Op,
    /// Indices of producer nodes in `Graph::nodes`; empty for `Input`.
    pub inputs: Vec<usize>,
    pub dtype: DtypeRepr,
    /// Output shape of this node. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    /// Node index → dotted name captured from [`crate::name_scope`] at recording time.
    pub names: BTreeMap<usize, String>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<usize>,
        dtype: DtypeRepr,
        shape: Shape,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GraphNode { op, inputs, dtype, shape });
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

    /// Rewrite the graph by fusing compatible op sequences into single fused ops.
    ///
    /// Currently recognises: `Conv2d(no bias) → BatchNorm2d → Silu`
    /// and replaces the triple with a single `Conv2dBnSilu` node.
    ///
    /// Nodes that are absorbed into a fused node are removed from the output
    /// graph; remaining node indices are renumbered contiguously.
    pub fn optimise(&self) -> Graph {
        let n = self.nodes.len();

        // Build per-node consumer counts so we can detect single-use nodes.
        let mut n_consumers = vec![0usize; n];
        for node in &self.nodes {
            for &inp in &node.inputs {
                n_consumers[inp] += 1;
            }
        }

        // dead[i] — node i is absorbed into a downstream fused node.
        let mut dead = vec![false; n];
        // node_override[i] — replacement (op, inputs) for node i.
        let mut node_override: Vec<Option<(Op, Vec<usize>)>> = vec![None; n];

        for silu_idx in 0..n {
            if !matches!(self.nodes[silu_idx].op, Op::Silu) { continue; }
            if self.nodes[silu_idx].inputs.len() != 1 { continue; }

            let bn_idx = self.nodes[silu_idx].inputs[0];
            if !matches!(self.nodes[bn_idx].op, Op::BatchNorm2d { .. }) { continue; }
            if n_consumers[bn_idx] != 1 { continue; }
            if self.nodes[bn_idx].inputs.len() != 1 { continue; }

            let conv_idx = self.nodes[bn_idx].inputs[0];
            if !matches!(self.nodes[conv_idx].op, Op::Conv2d { has_bias: false, .. }) { continue; }
            if n_consumers[conv_idx] != 1 { continue; }

            let (in_channels, out_channels, kernel_h, kernel_w,
                 stride_h, stride_w, padding_h, padding_w, groups) =
                if let Op::Conv2d { in_channels, out_channels, kernel_h, kernel_w,
                                    stride_h, stride_w, padding_h, padding_w, groups, .. } =
                    self.nodes[conv_idx].op
                {
                    (in_channels, out_channels, kernel_h, kernel_w,
                     stride_h, stride_w, padding_h, padding_w, groups)
                } else { unreachable!() };

            let bn_eps = if let Op::BatchNorm2d { eps, .. } = self.nodes[bn_idx].op {
                eps
            } else { unreachable!() };

            dead[conv_idx] = true;
            dead[bn_idx] = true;
            node_override[silu_idx] = Some((
                Op::Conv2dBnSilu {
                    in_channels, out_channels,
                    kernel_h, kernel_w,
                    stride_h, stride_w,
                    padding_h, padding_w,
                    groups, bn_eps,
                },
                self.nodes[conv_idx].inputs.clone(),
            ));
        }

        // Build old → new index mapping (dead nodes are skipped).
        let mut old_to_new = vec![0usize; n];
        let mut new_count = 0usize;
        for i in 0..n {
            if !dead[i] {
                old_to_new[i] = new_count;
                new_count += 1;
            }
        }

        // Emit new graph in topological order (original ordering is already valid).
        let mut new_graph = Graph::new();
        for old_idx in 0..n {
            if dead[old_idx] { continue; }
            let node = &self.nodes[old_idx];
            let (op, inputs) = if let Some((fused_op, fused_inputs)) =
                node_override[old_idx].clone()
            {
                let mapped = fused_inputs.iter().map(|&i| old_to_new[i]).collect();
                (fused_op, mapped)
            } else {
                let mapped = node.inputs.iter().map(|&i| old_to_new[i]).collect();
                (node.op.clone(), mapped)
            };
            let new_idx = new_graph.nodes.len();
            new_graph.nodes.push(GraphNode {
                op,
                inputs,
                dtype: node.dtype,
                shape: node.shape.clone(),
            });
            if let Some(name) = self.names.get(&old_idx) {
                new_graph.names.insert(new_idx, name.clone());
            }
        }

        new_graph
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
        Op::Conv1d { out_channels, kernel_l, stride, padding, .. } => {
            // [N, C_in, L] → [N, C_out, L_out]
            let l_out = input[2].map(|l| (l + 2 * padding - kernel_l) / stride + 1);
            vec![input[0], Some(*out_channels), l_out]
        }

        Op::Conv2d { out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, .. }
        | Op::Conv2dBnSilu { out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, .. } => {
            // [N, C_in, H, W] → [N, C_out, H_out, W_out]
            let h_out = input[2].map(|h| (h + 2 * padding_h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w + 2 * padding_w - kernel_w) / stride_w + 1);
            vec![input[0], Some(*out_channels), h_out, w_out]
        }

        Op::Conv3d { out_channels, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, .. } => {
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

        Op::LpPool1d { kernel_l, stride, .. } => {
            let l_out = input[2].map(|l| (l - kernel_l) / stride + 1);
            vec![input[0], input[1], l_out]
        }

        Op::AvgPool2d { kernel_h, kernel_w, stride_h, stride_w } => {
            let h_out = input[2].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::MaxPool2d { kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w } => {
            let h_out = input[2].map(|h| (h + 2 * pad_h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w + 2 * pad_w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::LpPool2d { kernel_h, kernel_w, stride_h, stride_w, .. } => {
            let h_out = input[2].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::AvgPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w }
        | Op::MaxPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w } => {
            let d_out = input[2].map(|d| (d - kernel_d) / stride_d + 1);
            let h_out = input[3].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[4].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], d_out, h_out, w_out]
        }

        Op::LpPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, .. } => {
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
        Op::ConstantPad1d { pad_left, pad_right, .. }
        | Op::ReflectionPad1d { pad_left, pad_right }
        | Op::ReplicationPad1d { pad_left, pad_right }
        | Op::CircularPad1d { pad_left, pad_right } => {
            // [N, C, L] → [N, C, L + pad_left + pad_right]
            let l_out = input[2].map(|l| l + pad_left + pad_right);
            vec![input[0], input[1], l_out]
        }

        Op::ConstantPad2d { pad_l, pad_r, pad_t, pad_b, .. }
        | Op::ReflectionPad2d { pad_l, pad_r, pad_t, pad_b }
        | Op::ReplicationPad2d { pad_l, pad_r, pad_t, pad_b }
        | Op::CircularPad2d { pad_l, pad_r, pad_t, pad_b } => {
            // [N, C, H, W] → [N, C, H + pad_t + pad_b, W + pad_l + pad_r]
            let h_out = input[2].map(|h| h + pad_t + pad_b);
            let w_out = input[3].map(|w| w + pad_l + pad_r);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::ConstantPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2, .. }
        | Op::ReflectionPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 }
        | Op::ReplicationPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 }
        | Op::CircularPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
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
        Op::Abs | Op::Neg | Op::Ceil | Op::Floor | Op::Round | Op::Sqrt
        | Op::Reciprocal | Op::Exp | Op::Log | Op::Erf | Op::Sign | Op::IsNaN
        | Op::IsInf { .. } | Op::Not | Op::BitwiseNot
        | Op::Sin | Op::Cos | Op::Tan | Op::Asin | Op::Acos | Op::Atan
        | Op::Sinh | Op::Cosh | Op::Asinh | Op::Acosh | Op::Atanh
        | Op::PRelu | Op::ThresholdedRelu { .. } | Op::Shrink { .. } | Op::Clip
        | Op::Swish | Op::LogSoftmax { .. } | Op::Hardmax { .. }
        | Op::Dropout { .. } | Op::Identity
        | Op::LRN { .. } | Op::MeanVarianceNormalization { .. }
        | Op::LpNormalization { .. }
        | Op::Pad { .. } | Op::ReverseSequence { .. } | Op::Trilu { .. }
        | Op::CumSum { .. } | Op::CumProd { .. }
        | Op::QuantizeLinear { .. } | Op::DequantizeLinear { .. } | Op::DynamicQuantizeLinear
        | Op::Bernoulli { .. } | Op::RandomUniformLike { .. } | Op::EyeLike { .. }
        | Op::RotaryEmbedding | Op::MultiHeadAttention { .. }
        => input.clone(),

        // Binary / variadic element-wise — approximate as first-input shape
        Op::Mul | Op::Sub | Op::Div | Op::Pow | Op::Mod { .. }
        | Op::ElemMin | Op::ElemMax | Op::ElemMean | Op::ElemSum
        | Op::Equal | Op::Greater | Op::GreaterOrEqual | Op::Less | Op::LessOrEqual
        | Op::And | Op::Or | Op::Xor
        | Op::BitwiseAnd | Op::BitwiseOr | Op::BitwiseXor | Op::BitShift { .. }
        | Op::Cast { .. } | Op::CastLike | Op::BitCast { .. } | Op::Where
        => input.clone(),

        // Structural ops where output shape = input shape or is unknown at
        // static inference time (ONNX value_info carries the true shape).
        Op::Reshape | Op::Squeeze { .. } | Op::Unsqueeze { .. }
        | Op::Slice | Op::Gather { .. } | Op::GatherElements { .. }
        | Op::GatherND { .. } | Op::ScatterElements { .. } | Op::ScatterND
        | Op::Tile | Op::Expand | Op::Compress { .. } | Op::Range
        | Op::ConstantOfShape { .. } | Op::NonZero
        | Op::Scatter { .. } | Op::TensorScatter
        | Op::Resize { .. } | Op::GridSample { .. }
        | Op::AffineGrid { .. } | Op::CenterCropPad { .. }
        => input.clone(),

        Op::Transpose { perm } => {
            if perm.is_empty() {
                input.iter().rev().cloned().collect()
            } else {
                perm.iter().map(|&i| input.get(i).copied().unwrap_or(None)).collect()
            }
        }

        Op::Concat { axis } => {
            let rank = input.len();
            if rank == 0 { return input.clone(); }
            let ax = axis.rem_euclid(rank as i64) as usize;
            let mut out = input.clone();
            // Sum the concatenated axis across all inputs.
            out[ax] = inputs.iter().try_fold(0usize, |acc, s| {
                s.get(ax).copied().unwrap_or(None).map(|d| acc + d)
            }).map(Some).unwrap_or(None);
            out
        }

        Op::Split { axis, num_outputs } => {
            let rank = input.len();
            if rank == 0 { return input.clone(); }
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

        Op::Gemm { trans_a, trans_b, .. } => {
            let m = if *trans_a { input.get(1) } else { input.first() }
                .copied().unwrap_or(None);
            let n = if inputs.len() >= 2 {
                let b = inputs[1];
                if *trans_b { b.first() } else { b.get(1) }.copied().unwrap_or(None)
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

        Op::Einsum { .. } | Op::Det | Op::Col2Im { .. }
        | Op::ConvInteger { .. } | Op::DeformConv { .. } | Op::QLinearConv { .. }
        => input.clone(),

        Op::ConvTranspose {
            out_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w, ..
        } => {
            let h_out = input[2].map(|h| {
                (h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
            });
            let w_out = input[3].map(|w| {
                (w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
            });
            vec![input[0], Some(*out_channels), h_out, w_out]
        }

        Op::ReduceSum { keepdims, .. } | Op::ReduceMean { keepdims, .. }
        | Op::ReduceMax { keepdims, .. } | Op::ReduceMin { keepdims, .. }
        | Op::ReduceProd { keepdims, .. } | Op::ReduceL1 { keepdims, .. }
        | Op::ReduceL2 { keepdims, .. } | Op::ReduceLogSum { keepdims, .. }
        | Op::ReduceLogSumExp { keepdims, .. } | Op::ReduceSumSquare { keepdims, .. } => {
            // Without axis info at static-inference time, approximate:
            // keepdims=true → same rank, keepdims=false → reduce all → scalar.
            if *keepdims { input.clone() } else { vec![Some(1)] }
        }

        Op::ArgMax { axis, keepdims, .. } | Op::ArgMin { axis, keepdims, .. } => {
            if input.is_empty() { return vec![]; }
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
            for _ in 2..input.len() { out.push(Some(1)); }
            out
        }

        Op::Lstm { hidden_size, bidirectional, .. }
        | Op::Gru { hidden_size, bidirectional, .. }
        | Op::Rnn { hidden_size, bidirectional, .. } => {
            let num_dirs: usize = if *bidirectional { 2 } else { 1 };
            // [seq_len, num_directions, batch, hidden_size] (approximate)
            vec![input.first().copied().unwrap_or(None),
                 Some(num_dirs),
                 input.get(1).copied().unwrap_or(None),
                 Some(*hidden_size)]
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

        Op::RoiAlign { output_h, output_w, .. } => {
            vec![input[0], input[1], Some(*output_h), Some(*output_w)]
        }

        Op::MaxUnpool { kernel_h, kernel_w, stride_h, stride_w } => {
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

        Op::Stft | Op::MelWeightMatrix
        | Op::HannWindow { .. } | Op::BlackmanWindow { .. } | Op::HammingWindow { .. }
        => input.clone(),

        Op::SequenceAt | Op::SequenceConstruct | Op::SequenceErase | Op::SequenceInsert
        | Op::SequenceLength | Op::SequenceMap
        | Op::SplitToSequence { .. } | Op::ConcatFromSequence { .. }
        | Op::OptionalGetElement
        | Op::Loop | Op::Scan { .. } | Op::If
        | Op::Adagrad | Op::Adam | Op::Momentum | Op::Gradient
        | Op::StringNormalizer | Op::RegexFullMatch { .. } | Op::StringConcat | Op::StringSplit
        | Op::TfIdfVectorizer | Op::LabelEncoder
        | Op::ArrayFeatureExtractor | Op::Binarizer { .. } | Op::TreeEnsemble | Op::ImageDecoder
        => input.clone(),

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
    pub node_id: usize,
    pub graph: Rc<RefCell<Graph>>,
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
        let tensor = Self { node_id, graph: graph.clone(), dtype, shape };
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
        Self { node_id, graph: self.graph.clone(), dtype: self.dtype, shape }
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
        let node_id = self
            .graph
            .borrow_mut()
            .add_node(Op::Custom { data }, input_ids, out_dtype, output_shape.clone());
        Self { node_id, graph: self.graph.clone(), dtype: out_dtype, shape: output_shape }
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
        input.record(Op::AvgPool1d { kernel_l: self.kernel_l, stride: self.stride })
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
        input.record(Op::MaxPool1d { kernel_l: self.kernel_l, stride: self.stride })
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
        input.record(Op::LpPool1d { kernel_l: self.kernel_l, stride: self.stride, p: self.p })
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
        input.record(Op::ReflectionPad1d { pad_left: self.pad_left, pad_right: self.pad_right })
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
        input.record(Op::ReplicationPad1d { pad_left: self.pad_left, pad_right: self.pad_right })
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
        input.record(Op::CircularPad1d { pad_left: self.pad_left, pad_right: self.pad_right })
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
        input.record(Op::Hardtanh { min_val: self.min_val, max_val: self.max_val })
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
        input.record(Op::Hardshrink { lambda: self.lambda })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for LeakyRelu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::LeakyRelu { negative_slope: self.negative_slope })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Threshold<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Threshold { threshold: self.threshold, value: self.value })
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
        input.record(Op::Softshrink { lambda: self.lambda })
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softplus<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Softplus { beta: self.beta, threshold: self.threshold })
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
            Op::Linear { in_features: 784, out_features: 128, .. }
        ));
        assert_eq!(g.nodes[1].shape, vec![None, Some(128)]);

        assert!(matches!(g.nodes[2].op, Op::Relu));
        assert_eq!(g.nodes[2].shape, vec![None, Some(128)]);

        assert!(matches!(
            g.nodes[3].op,
            Op::Linear { in_features: 128, out_features: 10, .. }
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

        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(
            3, 64, (3, 3), (1, 1), (1, 1), true,
        );
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
