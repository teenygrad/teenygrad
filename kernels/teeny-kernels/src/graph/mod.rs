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

use std::sync::Arc;
use teeny_core::{
    graph::{DtypeRepr, Graph, Op, Shape},
    model::{
        ExecutableOp, KernelTileSpec, Lowering, LoweringMode, RuntimeOp, TensorTileSpec,
        TileAxisBinding, TileCarryBinding, TileLoopSpec,
    },
    utils::dag::Dag,
};

pub mod optimizer;

pub use optimizer::{
    Anduin, DagCodegen, EdgeId, ExecuteDevice, GraphOptimizer, NodeId, Profiler, SimpleProfiler,
    SubGraphTilingResult, TileConfig, TileDim, TileEdge, TileEdgeShape, TileGraph, TileOp, Trace,
    TraceEvent, codegen, common_thread_block_size, schedule_graph,
};

use crate::nn::{
    activation::extra::{
        LogSoftmaxBackward, LogSoftmaxForward, PreluForward, ShrinkRuntimeOp, SwishBackward,
        SwishForward, ThresholdedReluRuntimeOp,
    },
    activation::{
        elu::{
            CeluForward, CeluForwardDispatch, EluForward, EluForwardDispatch, SeluForward,
            SeluForwardDispatch,
        },
        gelu::{GeluForwardDispatch, MishForward, MishForwardDispatch},
        hard::{
            HardshrinkForward, HardshrinkForwardDispatch, HardsigmoidForward,
            HardsigmoidForwardDispatch, HardswishForward, HardswishForwardDispatch,
            HardtanhForward, HardtanhForwardDispatch, Relu6Forward, Relu6ForwardDispatch,
        },
        misc::{
            LeakyReluForward, LeakyReluForwardDispatch, SoftplusForward, SoftplusForwardDispatch,
            SoftshrinkForward, SoftshrinkForwardDispatch, SoftsignForward, SoftsignForwardDispatch,
            ThresholdForward, ThresholdForwardDispatch,
        },
        relu::{ReluBackward, ReluForward},
        sigmoid::{
            LogsigmoidForward, LogsigmoidForwardDispatch, SigmoidForwardDispatch, SiluForward,
            SiluForwardDispatch,
        },
        softmax::SoftmaxForward,
        tanh::{TanhForwardDispatch, TanhshrinkForward, TanhshrinkForwardDispatch},
    },
    conv::{
        conv1d::Conv1dForward,
        conv2d::{Conv2dBackward, Conv2dBiasForward, Conv2dForward},
        conv3d::Conv3dForward,
    },
    mlp::{
        flatten::FlattenForward,
        linear::{LinearBackward, LinearForward},
    },
    norm::{
        batchnorm::{BatchNorm2dNchwInferenceRuntimeOp, BatchNormForwardInference},
        groupnorm::GroupNormForwardInference,
        instancenorm::InstanceNormForwardInference,
        layernorm::{LayerNormForwardInference, LayerNormForwardInferenceRuntimeOp},
        rmsnorm::RmsNormForward,
    },
    pad::{
        circular_pad1d::CircularPad1dForward, circular_pad2d::CircularPad2dForward,
        circular_pad3d::CircularPad3dForward, constant_pad1d::ConstantPad1dForward,
        constant_pad2d::ConstantPad2dForward, constant_pad3d::ConstantPad3dForward,
        reflection_pad1d::ReflectionPad1dForward, reflection_pad2d::ReflectionPad2dForward,
        reflection_pad3d::ReflectionPad3dForward, replication_pad1d::ReplicationPad1dForward,
        replication_pad2d::ReplicationPad2dForward, replication_pad3d::ReplicationPad3dForward,
    },
    pool::{
        avgpool1d::Avgpool1dForward,
        avgpool2d::Avgpool2dForward,
        avgpool3d::Avgpool3dForward,
        lppool1d::Lppool1dForward,
        lppool2d::Lppool2dForward,
        lppool3d::Lppool3dForward,
        maxpool1d::Maxpool1dForward,
        maxpool2d::{Maxpool2dBackward, Maxpool2dForward},
        maxpool3d::Maxpool3dForward,
    },
    tensor::{
        channel_bias_add::{ChannelBiasAddRuntimeOp, NchwBiasAddRuntimeOp},
        channel_cat::ChannelCatRuntimeOp,
        channel_chunk::ChannelChunkRuntimeOp,
        elemwise_add::{ElemwiseAddBackward, ElemwiseAddForward},
        elemwise_binary::{
            ClipRuntimeOp, ElemwiseDivBackward, ElemwiseDivForward, ElemwiseEqualForward,
            ElemwiseFmodForward, ElemwiseGreaterEqualForward, ElemwiseGreaterForward,
            ElemwiseLessEqualForward, ElemwiseLessForward, ElemwiseMaxBackward, ElemwiseMaxForward,
            ElemwiseMeanBackward, ElemwiseMeanForward, ElemwiseMinBackward, ElemwiseMinForward,
            ElemwiseMulBackward, ElemwiseMulForward, ElemwisePowBackward, ElemwisePowForward,
            ElemwiseSubBackward, ElemwiseSubForward, ElemwiseSumBackward, ElemwiseSumForward,
            ElemwiseWhereBackward, ElemwiseWhereForward,
        },
        elemwise_unary::{
            ElemwiseAbsBackward, ElemwiseAbsForward, ElemwiseAcosBackward, ElemwiseAcosForward,
            ElemwiseAcoshBackward, ElemwiseAcoshForward, ElemwiseAsinBackward, ElemwiseAsinForward,
            ElemwiseAsinhBackward, ElemwiseAsinhForward, ElemwiseAtanBackward, ElemwiseAtanForward,
            ElemwiseAtanhBackward, ElemwiseAtanhForward, ElemwiseCeilForward, ElemwiseCosBackward,
            ElemwiseCosForward, ElemwiseCoshBackward, ElemwiseCoshForward, ElemwiseErfBackward,
            ElemwiseErfForward, ElemwiseExpBackward, ElemwiseExpForward, ElemwiseFloorForward,
            ElemwiseIsnanForward, ElemwiseLogBackward, ElemwiseLogForward, ElemwiseNegBackward,
            ElemwiseNegForward, ElemwiseReciprocalBackward, ElemwiseReciprocalForward,
            ElemwiseSignForward, ElemwiseSinBackward, ElemwiseSinForward, ElemwiseSinhBackward,
            ElemwiseSinhForward, ElemwiseSqrtBackward, ElemwiseSqrtForward, ElemwiseTanBackward,
            ElemwiseTanForward,
        },
        reduction::{
            CumProdForward, CumSumForward, GlobalAvgPoolForward, GlobalMaxPoolForward,
            ReduceL1Forward, ReduceL2Forward, ReduceLogSumExpForward, ReduceLogSumForward,
            ReduceMaxForward, ReduceMeanForward, ReduceMinForward, ReduceProdForward,
            ReduceSumForward, ReduceSumSquareForward,
        },
        transpose::TransposeRuntimeOp,
        upsample_nearest2d::{UpsampleNearest2dBackward, UpsampleNearest2dForward},
    },
};

use crate::math::gemm::MatMulRuntimeOp;

use crate::errors::Result;

#[cfg(feature = "training")]
use crate::nn::norm::batchnorm::{
    BatchNorm2dNchwBackward, BatchNormNormalizeForward, BatchNormNormalizeRuntimeOp,
    BatchNormStatsForward, BatchNormStatsRuntimeOp,
};

// ---------------------------------------------------------------------------
// Tile-shape metadata (teenygrad-1nr.2) — declarative KernelTileSpecs for a
// first proof-of-concept slice of ops, consumed by TileGraph::propagate.
// See teeny_core::model::KernelTileSpec's doc comment for the design.
// ---------------------------------------------------------------------------

/// Builds a [`KernelTileSpec`] for a flat, single-`BLOCK_SIZE` elementwise
/// kernel -- every kernel [`exec_from`] assembles (`sigmoid_forward`,
/// `silu_forward`, ... -- see that function's own doc comment for the
/// full list), plus `Op::Relu` below: the whole tensor is read/written as
/// one flattened `n_elements` range regardless of its real declared rank.
/// `x_ptr`/`y_ptr` share one axis spanning *every* real dim (`dims:
/// &[0..rank]`), matching [`TileAxisBinding::dims`]'s flattened-axis
/// convention (teenygrad-1nr.8/.9) -- input and output share the same
/// `extent_param` name, so propagating an output tile resolves the
/// input's tile with no arithmetic at all.
///
/// Unlike every other spec in this file, this can't be a single `const`:
/// the same kernel gets applied to tensors of any real rank (a `Sigmoid`
/// node might be 2-D, 3-D, or 4-D depending on the graph), and
/// `TensorTileSpec::rank`/`TileAxisBinding::dims` must match that real
/// rank exactly for `TileGraph::propagate` to do anything with it (a
/// fixed `rank: 1` `const`, which is what this spec used to be, only
/// ever matched an already-flattened 1-D node -- never a realistic ND
/// tensor). Built fresh per call, `Box::leak`ing the rank-sized `dims`
/// slice: a small, permanent, bounded allocation (one call per node
/// `TritonLowering` ever lowers), not a per-iteration leak.
fn flat_elementwise_tile_spec(rank: usize) -> KernelTileSpec {
    let dims: &'static [usize] = Box::leak((0..rank).collect::<Vec<usize>>().into_boxed_slice());
    let axis = TileAxisBinding {
        dims,
        block_const: "BLOCK_SIZE",
        extent_param: "n_elements",
        window: None,
        divide_by: None,
    };
    let axes: &'static [TileAxisBinding] = Box::leak(Box::new([axis]));
    let x = TensorTileSpec {
        param: "x_ptr",
        rank,
        axes,
        reduction_axis: None,
        untiled_dims: &[],
    };
    let y = TensorTileSpec {
        param: "y_ptr",
        ..x
    };
    KernelTileSpec {
        inputs: Box::leak(Box::new([x])),
        outputs: Box::leak(Box::new([y])),
        loop_spec: None,
    }
}

/// GEMM-shaped: `a_ptr: [M, K]`, `b_ptr: [K, N]`, `c_ptr: [M, N]`. `M`/`N`
/// are shared with `c_ptr`'s own axes, so propagating `c_ptr`'s chosen
/// output tile resolves them on `a_ptr`/`b_ptr` too; `K` has no output-side
/// counterpart and is correctly left unresolved (its tile size is a search
/// decision, not something `Propagate` derives — see the module doc
/// comment on `teeny_core::model::tile_spec`).
const MATMUL_TILE_SPEC: KernelTileSpec = {
    const A: TensorTileSpec = TensorTileSpec {
        param: "a_ptr",
        rank: 2,
        axes: &[
            TileAxisBinding {
                dims: &[0],
                block_const: "BLOCK_M",
                extent_param: "M",
                window: None,
                divide_by: None,
            },
            TileAxisBinding {
                dims: &[1],
                block_const: "BLOCK_K",
                extent_param: "K",
                window: None,
                divide_by: None,
            },
        ],
        reduction_axis: Some(1),
        untiled_dims: &[],
    };
    const B: TensorTileSpec = TensorTileSpec {
        param: "b_ptr",
        rank: 2,
        axes: &[
            TileAxisBinding {
                dims: &[0],
                block_const: "BLOCK_K",
                extent_param: "K",
                window: None,
                divide_by: None,
            },
            TileAxisBinding {
                dims: &[1],
                block_const: "BLOCK_N",
                extent_param: "N",
                window: None,
                divide_by: None,
            },
        ],
        reduction_axis: Some(0),
        untiled_dims: &[],
    };
    const C: TensorTileSpec = TensorTileSpec {
        param: "c_ptr",
        rank: 2,
        axes: &[
            TileAxisBinding {
                dims: &[0],
                block_const: "BLOCK_M",
                extent_param: "M",
                window: None,
                divide_by: None,
            },
            TileAxisBinding {
                dims: &[1],
                block_const: "BLOCK_N",
                extent_param: "N",
                window: None,
                divide_by: None,
            },
        ],
        reduction_axis: None,
        untiled_dims: &[],
    };
    KernelTileSpec {
        inputs: &[A, B],
        outputs: &[C],
        loop_spec: None,
    }
};

/// NCHW batchnorm2d inference (`batch_norm_2d_nchw_forward_inference`,
/// `nn::norm::batchnorm`): grid `[C, B]` (one CTA per channel×batch), each
/// CTA looping the *flattened* `H*W` range in `BLOCK_HW`-wide tiles -- no
/// single real axis (H alone, or W alone) corresponds to `BLOCK_HW`, so
/// this uses `TileAxisBinding::dims` spanning both (`&[2, 3]`, W
/// innermost, matching NCHW's row-major layout) instead of one dim per
/// binding like `RELU_TILE_SPEC`/`MATMUL_TILE_SPEC` above. Batch/channels
/// (dims 0/1) are real but grid-driven, left out of `axes` (untiled, kept
/// at full extent by `Propagate`). Shape-preserving elementwise (per
/// channel) like `RELU_TILE_SPEC`, so `x_ptr`/`y_ptr` share `"HW"`.
const BATCHNORM2D_TILE_SPEC: KernelTileSpec = {
    const HW: TileAxisBinding = TileAxisBinding {
        dims: &[2, 3],
        block_const: "BLOCK_HW",
        extent_param: "HW",
        window: None,
        divide_by: None,
    };
    const X: TensorTileSpec = TensorTileSpec {
        param: "x_ptr",
        rank: 4,
        axes: &[HW],
        reduction_axis: None,
        untiled_dims: &["B", "C"],
    };
    const Y: TensorTileSpec = TensorTileSpec {
        param: "y_ptr",
        ..X
    };
    KernelTileSpec {
        inputs: &[X],
        outputs: &[Y],
        loop_spec: None,
    }
};

/// `conv1d_forward`/`conv2d_forward`/`conv3d_forward`
/// (`nn::conv::{conv1d,conv2d,conv3d}`), and every `avg`/`max`/`lp`-pool
/// kernel across the same 1-D/2-D/3-D ranks (`nn::pool::*`) share the same
/// shape: grid decodes to `(b, c[, d[, h]], ow-tile)` (2-D adds `h`, 3-D
/// adds `d`), and only the *innermost* spatial axis (`L`/`W`) is
/// genuinely block-tiled, via `BLOCK_OL`/`BLOCK_OW` -- every other real
/// dim (batch, channels, and any outer spatial axes) is grid-driven, with
/// no block-size generic of its own. Input and output have no shared
/// axis (unlike `RELU_TILE_SPEC`/`BATCHNORM2D_TILE_SPEC`'s shape-
/// preserving case): input keeps every dim at full extent (no axes at
/// all -- conv's own windowed read, and pooling's own kernel/stride
/// read, both fall back to full extent this way, same as leaving a dim
/// out of `axes` always does; see `TileWindow`'s own doc comment on why
/// this codebase doesn't yet model the windowed extent itself), output
/// gets one axis for its own `BLOCK_OL`/`BLOCK_OW`-tiled dim.
///
/// One shared `const` per rank, not one per real kernel: `avgpool2d`/
/// `maxpool2d`/`lppool2d` (etc.) are structurally identical down to their
/// real `input_ptr`/`output_ptr` param names, and `param` isn't consumed
/// by `TileGraph::propagate` at all (only `rank`/`axes`/`divide_by`
/// are) -- see `teeny_core::model::tile_spec`'s module doc comment.
fn windowed_last_axis_tile_spec(
    rank: usize,
    block_const: &'static str,
    extent_param: &'static str,
    input_param: &'static str,
    output_param: &'static str,
) -> KernelTileSpec {
    let last_dim = rank.saturating_sub(1);
    let axis: &'static [TileAxisBinding] = Box::leak(Box::new([TileAxisBinding {
        dims: Box::leak(Box::new([last_dim])),
        block_const,
        extent_param,
        window: None,
        divide_by: None,
    }]));
    let input = TensorTileSpec {
        param: input_param,
        rank,
        axes: &[],
        reduction_axis: None,
        untiled_dims: &[],
    };
    let output = TensorTileSpec {
        param: output_param,
        rank,
        axes: axis,
        reduction_axis: None,
        untiled_dims: &[],
    };
    KernelTileSpec {
        inputs: Box::leak(Box::new([input])),
        outputs: Box::leak(Box::new([output])),
        loop_spec: None,
    }
}

/// `conv1d_forward`: `x_ptr`/`y_ptr`, `[B, C, L]` -> `[B, C_OUT, OL]`,
/// `BLOCK_OL` tiling `y_ptr`'s `L` axis (dim 2). A fixed rank (unlike
/// [`flat_elementwise_tile_spec`]'s per-instance case): every `Conv1d`
/// node is rank 3, so this can be a plain `const`.
const CONV1D_TILE_SPEC: KernelTileSpec = {
    const AXIS: TileAxisBinding = TileAxisBinding {
        dims: &[2],
        block_const: "BLOCK_OL",
        extent_param: "OL",
        window: None,
        divide_by: None,
    };
    const X: TensorTileSpec = TensorTileSpec {
        param: "x_ptr",
        rank: 3,
        axes: &[],
        reduction_axis: None,
        untiled_dims: &[],
    };
    const Y: TensorTileSpec = TensorTileSpec {
        param: "y_ptr",
        rank: 3,
        axes: &[AXIS],
        reduction_axis: None,
        untiled_dims: &[],
    };
    KernelTileSpec {
        inputs: &[X],
        outputs: &[Y],
        loop_spec: None,
    }
};

/// `conv2d_forward`'s real accumulation loop, layered onto the
/// macro-derived `Conv2dForward::tile_spec()` (teenygrad-1nr.19) at its
/// call site below rather than declared via `#[tile(...)]`: the kernel
/// body (`kernels/teeny-kernels/src/nn/conv/conv2d.rs`) accumulates into
/// `acc: [BLOCK_OW]` over a flat `for idx in 0..loop_bound` loop,
/// `loop_bound = (C_IN/G)*KH*KW` -- loop-carry metadata
/// ([`TileLoopSpec`]) isn't representable per-axis the way tile/grid
/// shape is, so `#[tile(...)]` doesn't attempt to derive it
/// (teenygrad-1nr.12).
const CONV2D_LOOP_SPEC: TileLoopSpec = TileLoopSpec {
    carries: &[TileCarryBinding {
        name: "acc",
        shape_consts: &["BLOCK_OW"],
    }],
    trip_count_factors: &["C_IN", "G", "KH", "KW"],
};

/// `conv3d_forward`: `x_ptr`/`y_ptr`, `[B, C, D, H, W]` ->
/// `[B, C_OUT, OD, OH, OW]`, `BLOCK_OW` tiling `y_ptr`'s `W` axis (dim 4).
/// Fixed rank 5, always NCDHW.
const CONV3D_TILE_SPEC: KernelTileSpec = {
    const AXIS: TileAxisBinding = TileAxisBinding {
        dims: &[4],
        block_const: "BLOCK_OW",
        extent_param: "OW",
        window: None,
        divide_by: None,
    };
    const X: TensorTileSpec = TensorTileSpec {
        param: "x_ptr",
        rank: 5,
        axes: &[],
        reduction_axis: None,
        untiled_dims: &[],
    };
    const Y: TensorTileSpec = TensorTileSpec {
        param: "y_ptr",
        rank: 5,
        axes: &[AXIS],
        reduction_axis: None,
        untiled_dims: &[],
    };
    KernelTileSpec {
        inputs: &[X],
        outputs: &[Y],
        loop_spec: None,
    }
};

// ---------------------------------------------------------------------------
// Dtype dispatch macros
//
// Each macro matches a DtypeRepr at runtime, instantiates the kernel struct
// with the corresponding concrete Rust type, and builds a KernelExecutable.
//
// make_num_kernel!  — for kernels with D: Num (int + float)
// make_float_kernel! — for kernels with D: Float (float only)
// ---------------------------------------------------------------------------

/// Dispatch to a D: Num kernel based on `$node.dtype`.
/// Usage: `make_num_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_num_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let (name, ks, body, probe_bs, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I8  => { let k = $K::<i8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I16 => { let k = $K::<i16>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I32 => { let k = $K::<i32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I64 => { let k = $K::<i64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U8  => { let k = $K::<u8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U16 => { let k = $K::<u16>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U32 => { let k = $K::<u32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U64 => { let k = $K::<u64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($K))),
        };
        Box::new(KernelExecutable {
            entry_point: format!("{}_entry_point", name),
            name,
            kernel_source: ks,
            kernel_body: body,
            pointwise_fuse_block_size: probe_bs,
            tile_spec: None,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: String::new(),
            #[cfg(feature = "training")]
            backward_entry_point: String::new(),
            runtime_op: rop,
        })
    }};
    // Variant with explicit backward kernel type (for ops that have backward support)
    ($K:ident ($($arg:expr),*), $Bwd:ident ($($barg:expr),*), $node:expr) => {{
        let (name, ks, body, probe_bs, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I8  => { let k = $K::<i8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I16 => { let k = $K::<i16>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I32 => { let k = $K::<i32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I64 => { let k = $K::<i64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U8  => { let k = $K::<u8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U16 => { let k = $K::<u16>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U32 => { let k = $K::<u32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U64 => { let k = $K::<u64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($K))),
        };
        #[cfg(feature = "training")]
        let (bwd_name, bwd_ks) = match $node.dtype {
            DtypeRepr::F32 => { let k = $Bwd::<f32>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::F64 => { let k = $Bwd::<f64>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::I8  => { let k = $Bwd::<i8>::new($($barg),*);  (k.name.to_string(), k.source.clone()) }
            DtypeRepr::I16 => { let k = $Bwd::<i16>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::I32 => { let k = $Bwd::<i32>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::I64 => { let k = $Bwd::<i64>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::U8  => { let k = $Bwd::<u8>::new($($barg),*);  (k.name.to_string(), k.source.clone()) }
            DtypeRepr::U16 => { let k = $Bwd::<u16>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::U32 => { let k = $Bwd::<u32>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::U64 => { let k = $Bwd::<u64>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($Bwd))),
        };
        Box::new(KernelExecutable {
            entry_point: format!("{}_entry_point", name),
            name,
            kernel_source: ks,
            kernel_body: body,
            pointwise_fuse_block_size: probe_bs,
            tile_spec: None,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: bwd_ks,
            #[cfg(feature = "training")]
            backward_entry_point: format!("{}_entry_point", bwd_name),
            runtime_op: rop,
        })
    }};
}

/// Dispatch to a D: Float kernel based on `$node.dtype`.
/// Usage: `make_float_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_float_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let (name, ks, body, probe_bs, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a Float dtype for {}", other, stringify!($K))),
        };
        Box::new(KernelExecutable {
            entry_point: format!("{}_entry_point", name),
            name,
            kernel_source: ks,
            kernel_body: body,
            pointwise_fuse_block_size: probe_bs,
            tile_spec: None,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: String::new(),
            #[cfg(feature = "training")]
            backward_entry_point: String::new(),
            runtime_op: rop,
        })
    }};
    // Variant with explicit float backward kernel
    ($K:ident ($($arg:expr),*), $Bwd:ident ($($barg:expr),*), $node:expr) => {{
        let (name, ks, body, probe_bs, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a Float dtype for {}", other, stringify!($K))),
        };
        #[cfg(feature = "training")]
        let (bwd_name, bwd_ks) = match $node.dtype {
            DtypeRepr::F32 => { let k = $Bwd::<f32>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            DtypeRepr::F64 => { let k = $Bwd::<f64>::new($($barg),*); (k.name.to_string(), k.source.clone()) }
            other => return Err(anyhow::anyhow!("{:?} is not a Float dtype for {}", other, stringify!($Bwd))),
        };
        Box::new(KernelExecutable {
            entry_point: format!("{}_entry_point", name),
            name,
            kernel_source: ks,
            kernel_body: body,
            pointwise_fuse_block_size: probe_bs,
            tile_spec: None,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: bwd_ks,
            #[cfg(feature = "training")]
            backward_entry_point: format!("{}_entry_point", bwd_name),
            runtime_op: rop,
        })
    }};
}

/// Assemble a [`KernelExecutable`] from a dtype-resolved [`KernelInstance`]
/// produced by a `#[kernel(dtypes = [..])]` dispatcher. Every current
/// caller is one of the flat, single-`BLOCK_SIZE` elementwise activations
/// (`Elu`/`Selu`/`Celu`/`Gelu`/`Mish`/`Hardtanh`/`Relu6`/`Hardsigmoid`/
/// `Hardswish`/`Hardshrink`/`LeakyRelu`/`Threshold`/`Softsign`/
/// `Softshrink`/`Softplus`/`Sigmoid`/`Silu`/`Logsigmoid`/`Tanh`/
/// `Tanhshrink`), so `tile_spec` is set unconditionally here via
/// [`flat_elementwise_tile_spec`] rather than per call site -- if a
/// future caller of this function isn't shaped like that, give it its
/// own construction instead of adding a case here (mirrors `Op::Relu`'s
/// own arm below, which doesn't call `exec_from` but sets the same spec).
fn exec_from(
    shape: Shape,
    dtype: DtypeRepr,
    inst: teeny_core::model::KernelInstance,
) -> Box<KernelExecutable> {
    let tile_spec = Some(flat_elementwise_tile_spec(shape.len()));
    Box::new(KernelExecutable {
        entry_point: format!("{}_entry_point", inst.name),
        name: inst.name,
        kernel_source: inst.source,
        kernel_body: inst.kernel_body,
        pointwise_fuse_block_size: inst.pointwise_fuse_block_size,
        tile_spec,
        shape,
        dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: inst
            .backward
            .as_ref()
            .map(|b| b.source.clone())
            .unwrap_or_default(),
        #[cfg(feature = "training")]
        backward_entry_point: inst
            .backward
            .as_ref()
            .map(|b| format!("{}_entry_point", b.name))
            .unwrap_or_default(),
        runtime_op: inst.runtime_op,
    })
}

// ---------------------------------------------------------------------------
// KernelExecutable — compilable unit produced by TritonLowering
// ---------------------------------------------------------------------------

/// A lowered op that carries the kernel source needed for compilation.
///
/// Callers that have `teeny-compiler` as a dependency can pass `kernel_source`
/// and `kernel_entry_point` to `compile_kernel` along with a chosen `Target`.
#[derive(Clone)]
pub struct KernelExecutable {
    pub name: String,
    /// Combined forward source (`kernel_body` + entry wrapper) used for compilation.
    pub kernel_source: String,
    /// Forward kernel body only (no C-ABI entry). Used when composing fused entries.
    pub kernel_body: String,
    pub entry_point: String,
    pub shape: Shape,
    pub dtype: DtypeRepr,
    /// Runtime dispatch object: how to pack args and compute the launch grid.
    /// `Input` nodes carry a no-op implementation.
    pub runtime_op: Arc<dyn RuntimeOp>,
    /// `Some(BLOCK_SIZE)` when this kernel passes the pointwise-fuse probe.
    pub pointwise_fuse_block_size: Option<i32>,
    /// Declarative tile-shape metadata (see [`teeny_core::model::KernelTileSpec`]),
    /// if this op has been annotated. `None` for the vast majority of ops —
    /// coverage is opt-in, hand-authored per op at the `TritonLowering`
    /// construction site, not derived.
    pub tile_spec: Option<KernelTileSpec>,
    /// Backward kernel source. Empty if this op has no backward.
    #[cfg(feature = "training")]
    pub backward_kernel_source: String,
    /// Backward kernel entry point name.
    #[cfg(feature = "training")]
    pub backward_entry_point: String,
}

impl ExecutableOp for KernelExecutable {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_input(&self) -> bool {
        self.name == "input"
    }

    fn forward_kernel_source(&self) -> &str {
        &self.kernel_source
    }

    fn forward_kernel_entry_point(&self) -> &str {
        &self.entry_point
    }

    fn output_shape(&self) -> &Shape {
        &self.shape
    }

    fn output_dtype(&self) -> DtypeRepr {
        self.dtype
    }

    fn runtime_op(&self) -> Option<Arc<dyn RuntimeOp>> {
        if self.is_input() {
            None
        } else {
            Some(Arc::clone(&self.runtime_op))
        }
    }

    fn tile_spec(&self) -> Option<KernelTileSpec> {
        self.tile_spec
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[cfg(feature = "training")]
    fn backward_kernel_source(&self) -> &str {
        &self.backward_kernel_source
    }

    #[cfg(feature = "training")]
    fn backward_kernel_entry_point(&self) -> &str {
        &self.backward_entry_point
    }
}

// ---------------------------------------------------------------------------
// Stub RuntimeOp impls for kernels not yet fully supported at runtime.
// These satisfy the Arc<dyn RuntimeOp> bound in the dispatch macros but
// panic if ever called through a LoadedModel.
// ---------------------------------------------------------------------------

macro_rules! impl_stub_runtime_op_num {
    ($T:ident) => {
        impl<D: teeny_core::dtype::Num + Send + Sync + 'static> RuntimeOp for $T<D> {
            fn n_activation_inputs(&self) -> usize {
                unimplemented!(concat!(stringify!($T), " has no runtime support"))
            }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
                unimplemented!()
            }
            fn pack_args(
                &self,
                _: &[(teeny_core::model::RawPtr, &[usize])],
                _: &[teeny_core::model::RawPtr],
                _: teeny_core::model::RawPtr,
                _: &[usize],
                _: i32,
                _: &mut dyn teeny_core::device::program::ArgVisitor,
            ) {
                unimplemented!()
            }
            fn grid(&self, _: &[usize]) -> [u32; 3] {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_stub_runtime_op_float {
    ($T:ident) => {
        impl<D: teeny_core::dtype::Float + Send + Sync + 'static> RuntimeOp for $T<D> {
            fn n_activation_inputs(&self) -> usize {
                unimplemented!(concat!(stringify!($T), " has no runtime support"))
            }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
                unimplemented!()
            }
            fn pack_args(
                &self,
                _: &[(teeny_core::model::RawPtr, &[usize])],
                _: &[teeny_core::model::RawPtr],
                _: teeny_core::model::RawPtr,
                _: &[usize],
                _: i32,
                _: &mut dyn teeny_core::device::program::ArgVisitor,
            ) {
                unimplemented!()
            }
            fn grid(&self, _: &[usize]) -> [u32; 3] {
                unimplemented!()
            }
        }
    };
}

// Normalisation
impl_stub_runtime_op_float!(BatchNormForwardInference);
impl_stub_runtime_op_float!(LayerNormForwardInference);
impl_stub_runtime_op_float!(RmsNormForward);
impl_stub_runtime_op_float!(GroupNormForwardInference);
impl_stub_runtime_op_float!(InstanceNormForwardInference);

// Convolution
impl_stub_runtime_op_num!(Conv3dForward);

// Pooling
impl_stub_runtime_op_num!(Avgpool1dForward);
impl_stub_runtime_op_num!(Avgpool3dForward);
impl_stub_runtime_op_num!(Maxpool1dForward);
impl_stub_runtime_op_num!(Maxpool3dForward);
impl_stub_runtime_op_float!(Lppool1dForward);
impl_stub_runtime_op_float!(Lppool2dForward);
impl_stub_runtime_op_float!(Lppool3dForward);

// Padding
impl_stub_runtime_op_num!(ConstantPad1dForward);
impl_stub_runtime_op_num!(ConstantPad2dForward);
impl_stub_runtime_op_num!(ConstantPad3dForward);
impl_stub_runtime_op_num!(ReflectionPad1dForward);
impl_stub_runtime_op_num!(ReflectionPad2dForward);
impl_stub_runtime_op_num!(ReflectionPad3dForward);
impl_stub_runtime_op_num!(ReplicationPad1dForward);
impl_stub_runtime_op_num!(ReplicationPad2dForward);
impl_stub_runtime_op_num!(ReplicationPad3dForward);
impl_stub_runtime_op_num!(CircularPad1dForward);
impl_stub_runtime_op_num!(CircularPad2dForward);
impl_stub_runtime_op_num!(CircularPad3dForward);

// Activation — dtype-generic (D: Float) kernels without runtime support yet.
// GeluForward, SiluForward, and TanhForward have real RuntimeOp impls in their
// kernel modules.
impl_stub_runtime_op_float!(EluForward);
impl_stub_runtime_op_float!(SeluForward);
impl_stub_runtime_op_float!(CeluForward);
impl_stub_runtime_op_float!(MishForward);
impl_stub_runtime_op_float!(HardtanhForward);
impl_stub_runtime_op_float!(Relu6Forward);
impl_stub_runtime_op_float!(HardsigmoidForward);
impl_stub_runtime_op_float!(HardswishForward);
impl_stub_runtime_op_float!(HardshrinkForward);
impl_stub_runtime_op_float!(LeakyReluForward);
impl_stub_runtime_op_float!(ThresholdForward);
impl_stub_runtime_op_float!(SoftsignForward);
impl_stub_runtime_op_float!(SoftshrinkForward);
impl_stub_runtime_op_float!(SoftplusForward);
impl_stub_runtime_op_float!(LogsigmoidForward);
impl_stub_runtime_op_float!(TanhshrinkForward);

// ---------------------------------------------------------------------------
// No-op RuntimeOp for Input placeholder nodes
// ---------------------------------------------------------------------------

struct InputRuntimeOp;

impl RuntimeOp for InputRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        0
    }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }
    fn pack_args(
        &self,
        _: &[(teeny_core::model::RawPtr, &[usize])],
        _: &[teeny_core::model::RawPtr],
        _: teeny_core::model::RawPtr,
        _: &[usize],
        _: i32,
        _: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
    }
    fn grid(&self, _: &[usize]) -> [u32; 3] {
        [0, 0, 0]
    }
}

// ---------------------------------------------------------------------------
// TritonLowering
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct TritonLowering {}

impl TritonLowering {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Picks (BLOCK_M, BLOCK_N, BLOCK_K) for a GEMM of shape `[M, K] @ [K, N] -> [M, N]`
/// based on its size, instead of one fixed tile size for every shape.
///
/// Per spinorml-4gx's ONNX Runtime profile, cuDNN/CUTLASS auto-tunes tile size per
/// layer rather than using one fixed configuration — seven distinct configs were
/// observed in active use for one model (64x64_16, 128x64_16, 256x64_16, 128x128_16,
/// 256x128_16, 64x128_32, in `BLOCK_M x BLOCK_N _ BLOCK_K` terms). This mirrors that
/// spirit with a small fixed table rather than the exact tuned values, which are
/// specific to CUTLASS's own kernel templates:
///
/// - BLOCK_K grows with `k` (more reduction work amortizes each K-tile's load cost;
///   a small `k` gets a small BLOCK_K instead of wasting shared memory tiling past
///   the reduction dimension's actual extent). Never below 8: `T::dot`'s TF32 tensor
///   core path needs at least that much K per MMA tile.
/// - BLOCK_M/BLOCK_N grow with `m`/`n` for the same reason (bigger output tiles only
///   pay off once there's enough output to fill many CTAs at that size; small outputs
///   get finer tiles instead, for occupancy).
///
/// `m` is `None` when the batch dimension is dynamic (unknown at lowering time) — it's
/// treated as small (64) rather than guessed large, so an unexpectedly small runtime
/// batch doesn't end up under-occupied at a tile size chosen for a batch that never
/// materializes. This under-estimates for large dynamic batches rather than
/// over-tiling for a batch size that never appears.
fn pick_gemm_tile_sizes(m: Option<usize>, n: usize, k: usize) -> (i32, i32, i32) {
    let m = m.unwrap_or(64);
    let block_k = if k >= 128 {
        32
    } else if k >= 32 {
        16
    } else {
        8
    };
    // Capped at 128×128: a 256×128 tile needs ~128 KiB dynamic shared memory,
    // above the opt-in ceiling on some sm_120 devices (~99 KiB on RTX 5070).
    // 128×128 (~64 KiB) fits under the per-function opt-in raised via
    // `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` in teeny-cuda's launch path.
    let (block_m, block_n) = match (m, n) {
        (m, n) if m >= 128 && n >= 128 => (128, 128),
        (m, _) if m >= 128 => (128, 64),
        (_, n) if n >= 128 => (64, 128),
        _ => (64, 64),
    };
    (block_m, block_n, block_k)
}

#[cfg(test)]
mod pick_gemm_tile_sizes_tests {
    use super::pick_gemm_tile_sizes;

    #[test]
    fn small_shape_gets_smallest_tiles() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 16), (64, 64, 8));
    }

    #[test]
    fn large_m_and_n_get_the_largest_tile() {
        // Largest safe tile: 256×128 would exceed the sm_120 opt-in shared-memory ceiling.
        assert_eq!(pick_gemm_tile_sizes(Some(512), 256, 256), (128, 128, 32));
    }

    #[test]
    fn large_m_only_widens_block_m_not_block_n() {
        assert_eq!(pick_gemm_tile_sizes(Some(512), 32, 64), (128, 64, 16));
    }

    #[test]
    fn large_n_only_widens_block_n_not_block_m() {
        assert_eq!(pick_gemm_tile_sizes(Some(32), 512, 64), (64, 128, 16));
    }

    #[test]
    fn unknown_dynamic_batch_treated_as_small() {
        // m=None should behave the same as an explicit small m, not a large one.
        assert_eq!(
            pick_gemm_tile_sizes(None, 64, 16),
            pick_gemm_tile_sizes(Some(64), 64, 16)
        );
    }

    #[test]
    fn block_k_never_drops_below_the_tensor_core_minimum() {
        let (_, _, block_k) = pick_gemm_tile_sizes(Some(64), 64, 1);
        assert!(block_k >= 8);
    }

    #[test]
    fn block_k_grows_with_k() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 8).2, 8);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 32).2, 16);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 128).2, 32);
    }
}

impl TritonLowering {
    /// Lower a single unary `op` through the same Op→kernel table as
    /// [`Self::lower_with_mapping`], without running a graph optimizer.
    ///
    /// Builds a tiny `Input → op` graph and returns the lowered
    /// [`KernelExecutable`] for `op`. Used by pointwise fusion to obtain
    /// member kernels (and their fuse probes) without a parallel Dispatch map.
    pub fn lower_unary_op(&self, op: &Op, dtype: DtypeRepr) -> Result<KernelExecutable> {
        let shape: Shape = vec![Some(16)];
        let mut graph = Graph::new();
        let input = graph.add_node(Op::Input, vec![], dtype, shape.clone());
        let out = graph.add_node(op.clone(), vec![input], dtype, shape);

        // Never run Anduin (or any optimizer) while resolving a fuse member —
        // that would recurse into pointwise fusion.
        let lowering = TritonLowering::default();
        let (dag, map, _) = lowering.lower_with_mapping(&graph, LoweringMode::Inference)?;
        let dag_idx = map[out];
        let exec = dag.node(dag_idx).value.as_ref();
        if exec.is_input() {
            return Err(anyhow::anyhow!(
                "lower_unary_op: op {op:?} lowered to an Input placeholder"
            ));
        }
        exec.as_any()
            .downcast_ref::<KernelExecutable>()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("lower_unary_op: expected KernelExecutable for {op:?}"))
    }

    /// Like `lower` but also returns the graph-node-index → DAG-node-index
    /// mapping, plus `graph` itself (one DAG node per graph node — the
    /// mapping is the identity permutation of `graph`'s topological order).
    /// Useful for middleware lowerings that need to patch specific DAG nodes
    /// after the base lowering runs, and for placing pretrained weights
    /// (keyed by the graph's node names) onto the right DAG node.
    ///
    /// Does not run a [`GraphOptimizer`](crate::graph::optimizer::GraphOptimizer)
    /// (e.g. [`Anduin`]) — callers that want fusion run one over this
    /// function's `(Dag, Vec<usize>)` output themselves, via
    /// [`GraphOptimizer::optimize`](crate::graph::optimizer::GraphOptimizer::optimize).
    pub fn lower_with_mapping(
        &self,
        graph: &Graph,
        mode: LoweringMode,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>, Graph)> {
        let _ = mode; // used by #[cfg(feature = "training")] branch below
        let node_indexes = graph.topological_sort();
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        // Maps graph node index → DAG node index (one-to-one since we add every node)
        let mut graph_to_dag = vec![0usize; graph.nodes.len()];

        for node_index in node_indexes {
            let node = &graph.nodes[node_index];

            // Training BatchNorm needs two sequential DAG nodes: stats then normalize.
            // BatchNorm2d uses NCHW-native kernels (falls through to the inference path below)
            // which already supports training via has_backward=true.
            #[cfg(feature = "training")]
            if mode == LoweringMode::Training
                && let Op::BatchNorm1d {
                    num_features,
                    eps,
                    momentum,
                    ..
                }
                | Op::BatchNorm3d {
                    num_features,
                    eps,
                    momentum,
                    ..
                } = &node.op
            {
                let c = *num_features;
                let eps_f32 = *eps as f32;
                let momentum_f32 = *momentum as f32;
                const BLOCK_N: i32 = 64;

                let (stats_name, stats_src, stats_rop): (String, String, Arc<dyn RuntimeOp>) =
                    match node.dtype {
                        DtypeRepr::F32 => {
                            let k = BatchNormStatsForward::<f32>::new(BLOCK_N);
                            let src = k.source.clone();
                            let rop: Arc<dyn RuntimeOp> = Arc::new(
                                BatchNormStatsRuntimeOp::<f32>::new(BLOCK_N, eps_f32, momentum_f32),
                            );
                            (k.name.to_string(), src, rop)
                        }
                        DtypeRepr::F64 => {
                            let k = BatchNormStatsForward::<f64>::new(BLOCK_N);
                            let src = k.source.clone();
                            let rop: Arc<dyn RuntimeOp> = Arc::new(
                                BatchNormStatsRuntimeOp::<f64>::new(BLOCK_N, eps_f32, momentum_f32),
                            );
                            (k.name.to_string(), src, rop)
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not a Float dtype for BatchNormStatsForward",
                                other
                            ));
                        }
                    };

                let stats_node = Box::new(KernelExecutable {
                    entry_point: format!("{}_entry_point", stats_name),
                    name: stats_name,
                    kernel_source: stats_src,
                    kernel_body: String::new(),
                    pointwise_fuse_block_size: None,
                    tile_spec: None,
                    shape: vec![Some(2 * c)],
                    dtype: node.dtype,
                    backward_kernel_source: String::new(),
                    backward_entry_point: String::new(),
                    runtime_op: stats_rop,
                }) as Box<dyn ExecutableOp>;

                let stats_dag_idx = dag.add_node(stats_node);
                for &input_graph_idx in &node.inputs {
                    dag.add_edge(graph_to_dag[input_graph_idx], stats_dag_idx);
                }

                let (norm_name, norm_src, norm_bwd_src, norm_rop): (
                    String,
                    String,
                    String,
                    Arc<dyn RuntimeOp>,
                ) = match node.dtype {
                    DtypeRepr::F32 => {
                        let k = BatchNormNormalizeForward::<f32>::new(BLOCK_N);
                        let src = k.source.clone();
                        let rop = BatchNormNormalizeRuntimeOp::<f32>::new(BLOCK_N);
                        let bwd_src = rop.backward_source().to_string();
                        (
                            k.name.to_string(),
                            src,
                            bwd_src,
                            Arc::new(rop) as Arc<dyn RuntimeOp>,
                        )
                    }
                    DtypeRepr::F64 => {
                        let k = BatchNormNormalizeForward::<f64>::new(BLOCK_N);
                        let src = k.source.clone();
                        let rop = BatchNormNormalizeRuntimeOp::<f64>::new(BLOCK_N);
                        let bwd_src = rop.backward_source().to_string();
                        (
                            k.name.to_string(),
                            src,
                            bwd_src,
                            Arc::new(rop) as Arc<dyn RuntimeOp>,
                        )
                    }
                    other => {
                        return Err(anyhow::anyhow!(
                            "{:?} is not a Float dtype for BatchNormNormalizeForward",
                            other
                        ));
                    }
                };

                let norm_node = Box::new(KernelExecutable {
                    entry_point: format!("{}_entry_point", norm_name),
                    name: norm_name,
                    kernel_source: norm_src,
                    kernel_body: String::new(),
                    pointwise_fuse_block_size: None,
                    tile_spec: None,
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    backward_kernel_source: norm_bwd_src,
                    backward_entry_point: String::new(),
                    runtime_op: norm_rop,
                }) as Box<dyn ExecutableOp>;

                let norm_dag_idx = dag.add_node(norm_node);
                // normalize depends on x (same inputs as the BatchNorm graph node)
                for &input_graph_idx in &node.inputs {
                    dag.add_edge(graph_to_dag[input_graph_idx], norm_dag_idx);
                }
                // normalize also depends on the stats node output
                dag.add_edge(stats_dag_idx, norm_dag_idx);

                graph_to_dag[node_index] = norm_dag_idx;
                continue;
            }

            // Conv2d with bias, inference mode → one fused conv2d_bias_forward kernel
            // instead of two separate launches (spinorml-ia5). conv2d_bias_forward has
            // no backward pass, so training still takes the split path below.
            if mode == LoweringMode::Inference
                && let Op::Conv2d {
                    has_bias: true,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    groups,
                    ..
                } = &node.op
            {
                let (name, ks, rop): (String, String, Arc<dyn RuntimeOp>) = match node.dtype {
                    DtypeRepr::F32 => {
                        let k = Conv2dBiasForward::<f32>::new(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            *groups as i32,
                            16,
                        );
                        let nm = k.name.to_string();
                        let src = k.source.clone();
                        let rop: Arc<dyn RuntimeOp> = Arc::new(k);
                        (nm, src, rop)
                    }
                    DtypeRepr::F64 => {
                        let k = Conv2dBiasForward::<f64>::new(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            *groups as i32,
                            16,
                        );
                        let nm = k.name.to_string();
                        let src = k.source.clone();
                        let rop: Arc<dyn RuntimeOp> = Arc::new(k);
                        (nm, src, rop)
                    }
                    other => {
                        return Err(anyhow::anyhow!(
                            "{:?} is not supported for Conv2dBiasForward",
                            other
                        ));
                    }
                };
                let dag_idx = dag.add_node(Box::new(KernelExecutable {
                    entry_point: format!("{}_entry_point", name),
                    name,
                    kernel_source: ks,
                    kernel_body: String::new(),
                    pointwise_fuse_block_size: None,
                    tile_spec: None,
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    #[cfg(feature = "training")]
                    backward_kernel_source: String::new(),
                    #[cfg(feature = "training")]
                    backward_entry_point: String::new(),
                    runtime_op: rop,
                }) as Box<dyn ExecutableOp>);
                for &input_graph_idx in &node.inputs {
                    dag.add_edge(graph_to_dag[input_graph_idx], dag_idx);
                }
                graph_to_dag[node_index] = dag_idx;
                continue;
            }

            // Conv2d with bias (training mode) → split into Conv2d (weight only) +
            // NchwBiasAdd (bias only), since each needs its own backward kernel.
            if let Op::Conv2d {
                has_bias: true,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                groups,
                out_channels,
                ..
            } = &node.op
            {
                const BIAS_BLOCK_HW: i32 = 128;
                let (conv_name, conv_ks, conv_rop): (String, String, Arc<dyn RuntimeOp>) =
                    match node.dtype {
                        DtypeRepr::F32 => {
                            let k = Conv2dForward::<f32>::new(
                                *kernel_h as i32,
                                *kernel_w as i32,
                                *stride_h as i32,
                                *stride_w as i32,
                                *padding_h as i32,
                                *padding_w as i32,
                                *groups as i32,
                                16,
                            );
                            let src = k.source.clone();
                            let rop: Arc<dyn RuntimeOp> = Arc::new(Conv2dForward::<f32>::new(
                                *kernel_h as i32,
                                *kernel_w as i32,
                                *stride_h as i32,
                                *stride_w as i32,
                                *padding_h as i32,
                                *padding_w as i32,
                                *groups as i32,
                                16,
                            ));
                            (k.name.to_string(), src, rop)
                        }
                        DtypeRepr::F64 => {
                            let k = Conv2dForward::<f64>::new(
                                *kernel_h as i32,
                                *kernel_w as i32,
                                *stride_h as i32,
                                *stride_w as i32,
                                *padding_h as i32,
                                *padding_w as i32,
                                *groups as i32,
                                16,
                            );
                            let src = k.source.clone();
                            let rop: Arc<dyn RuntimeOp> = Arc::new(Conv2dForward::<f64>::new(
                                *kernel_h as i32,
                                *kernel_w as i32,
                                *stride_h as i32,
                                *stride_w as i32,
                                *padding_h as i32,
                                *padding_w as i32,
                                *groups as i32,
                                16,
                            ));
                            (k.name.to_string(), src, rop)
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not supported for Conv2dForward",
                                other
                            ));
                        }
                    };

                #[cfg(feature = "training")]
                let conv_bwd_ks = match node.dtype {
                    DtypeRepr::F32 => {
                        Conv2dBackward::<f32>::new(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            *groups as i32,
                            16,
                        )
                        .source
                    }
                    DtypeRepr::F64 => {
                        Conv2dBackward::<f64>::new(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            *groups as i32,
                            16,
                        )
                        .source
                    }
                    _ => String::new(),
                };

                let conv_dag_idx = dag.add_node(Box::new(KernelExecutable {
                    entry_point: format!("{}_entry_point", conv_name),
                    name: conv_name,
                    kernel_source: conv_ks,
                    kernel_body: String::new(),
                    pointwise_fuse_block_size: None,
                    tile_spec: None,
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    #[cfg(feature = "training")]
                    backward_kernel_source: conv_bwd_ks,
                    #[cfg(feature = "training")]
                    backward_entry_point: String::new(),
                    runtime_op: conv_rop,
                }) as Box<dyn ExecutableOp>);
                for &input_graph_idx in &node.inputs {
                    dag.add_edge(graph_to_dag[input_graph_idx], conv_dag_idx);
                }

                let (bias_name, bias_ks, bias_rop): (String, String, Arc<dyn RuntimeOp>) =
                    match node.dtype {
                        DtypeRepr::F32 => {
                            let r = NchwBiasAddRuntimeOp::<f32>::new(BIAS_BLOCK_HW);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = NchwBiasAddRuntimeOp::<f64>::new(BIAS_BLOCK_HW);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not supported for NchwBiasAdd",
                                other
                            ));
                        }
                    };

                #[cfg(feature = "training")]
                let bias_bwd_ks = match node.dtype {
                    DtypeRepr::F32 => NchwBiasAddRuntimeOp::<f32>::new(BIAS_BLOCK_HW)
                        .backward_source()
                        .to_string(),
                    DtypeRepr::F64 => NchwBiasAddRuntimeOp::<f64>::new(BIAS_BLOCK_HW)
                        .backward_source()
                        .to_string(),
                    _ => String::new(),
                };

                let biasadd_dag_idx = dag.add_node(Box::new(KernelExecutable {
                    entry_point: format!("{}_entry_point", bias_name),
                    name: bias_name,
                    kernel_source: bias_ks,
                    kernel_body: String::new(),
                    pointwise_fuse_block_size: None,
                    tile_spec: None,
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    #[cfg(feature = "training")]
                    backward_kernel_source: bias_bwd_ks,
                    #[cfg(feature = "training")]
                    backward_entry_point: String::new(),
                    runtime_op: bias_rop,
                }) as Box<dyn ExecutableOp>);
                dag.add_edge(conv_dag_idx, biasadd_dag_idx);
                graph_to_dag[node_index] = biasadd_dag_idx;
                // conv_dag_idx == biasadd_dag_idx - 1 (added consecutively).
                // extra_dag_names() uses this invariant to propagate the name to conv_dag_idx.
                let _ = (conv_dag_idx, out_channels);
                continue;
            }

            let executable: Box<dyn ExecutableOp> = match &node.op {
                Op::Input => Box::new(KernelExecutable {
                    name: "input".to_string(),
                    kernel_source: String::new(),
                    kernel_body: String::new(),
                    pointwise_fuse_block_size: None,
                    tile_spec: None,
                    entry_point: String::new(),
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    #[cfg(feature = "training")]
                    backward_kernel_source: String::new(),
                    #[cfg(feature = "training")]
                    backward_entry_point: String::new(),
                    runtime_op: Arc::new(InputRuntimeOp),
                }),

                // --- Linear / MLP ---
                Op::Linear { has_bias, .. } => {
                    make_num_kernel!(
                        LinearForward(*has_bias, 32, 64, 32, 8),
                        LinearBackward(*has_bias, 32, 64, 32, 8),
                        node
                    )
                }
                Op::Flatten => make_num_kernel!(FlattenForward(32, 256), node),

                // --- Normalisation ---
                Op::BatchNorm1d { .. } | Op::BatchNorm3d { .. } => {
                    make_float_kernel!(BatchNormForwardInference(64), node)
                }
                Op::BatchNorm2d { eps, .. } => {
                    let eps_f32 = *eps as f32;
                    const BN2D_BLOCK_HW: i32 = 128;
                    let (name, ks, rop): (String, String, Arc<dyn RuntimeOp>) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = BatchNorm2dNchwInferenceRuntimeOp::<f32>::new(
                                BN2D_BLOCK_HW,
                                eps_f32,
                            );
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = BatchNorm2dNchwInferenceRuntimeOp::<f64>::new(
                                BN2D_BLOCK_HW,
                                eps_f32,
                            );
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not a Float dtype for BatchNorm2d",
                                other
                            ));
                        }
                    };
                    #[cfg(feature = "training")]
                    let bwd_ks = match node.dtype {
                        DtypeRepr::F32 => BatchNorm2dNchwBackward::<f32>::new(BN2D_BLOCK_HW).source,
                        DtypeRepr::F64 => BatchNorm2dNchwBackward::<f64>::new(BN2D_BLOCK_HW).source,
                        _ => String::new(),
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: Some(BATCHNORM2D_TILE_SPEC),
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_ks,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::LayerNorm { eps, .. } => {
                    let eps_f32 = *eps as f32;
                    const LN_BLOCK_N: i32 = 1024;
                    let (name, ks, rop): (String, String, Arc<dyn RuntimeOp>) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r =
                                LayerNormForwardInferenceRuntimeOp::<f32>::new(LN_BLOCK_N, eps_f32);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r =
                                LayerNormForwardInferenceRuntimeOp::<f64>::new(LN_BLOCK_N, eps_f32);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not a Float dtype for LayerNorm",
                                other
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::RmsNorm { .. } => {
                    make_float_kernel!(RmsNormForward(1024), node)
                }
                Op::GroupNorm { .. } => {
                    make_float_kernel!(GroupNormForwardInference(256), node)
                }
                Op::InstanceNorm1d { .. }
                | Op::InstanceNorm2d { .. }
                | Op::InstanceNorm3d { .. } => {
                    make_float_kernel!(InstanceNormForwardInference(256), node)
                }

                // --- Convolution ---
                Op::Conv1d {
                    kernel_l,
                    stride,
                    padding,
                    ..
                } => {
                    let mut exec = make_num_kernel!(
                        Conv1dForward(*kernel_l as i32, *stride as i32, *padding as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(CONV1D_TILE_SPEC);
                    exec
                }
                Op::Conv2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    groups,
                    ..
                } => {
                    let mut exec = make_num_kernel!(
                        Conv2dForward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            *groups as i32,
                            16
                        ),
                        Conv2dBackward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            *groups as i32,
                            16
                        ),
                        node
                    );
                    // Conv2dForward::tile_spec (teenygrad-1nr.19, derived
                    // by `#[tiled_kernel]` from conv2d_forward's own
                    // `#[tile(...)]`-tagged x_ptr/y_ptr) is dtype-
                    // independent, like ReluForward::tile_spec above.
                    // loop_spec isn't attribute-derived (see
                    // CONV2D_LOOP_SPEC's own doc comment) -- layered on
                    // top here.
                    exec.tile_spec = Some(KernelTileSpec {
                        loop_spec: Some(CONV2D_LOOP_SPEC),
                        ..Conv2dForward::<f32>::tile_spec()
                    });
                    exec
                }
                Op::Conv3d {
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
                    let mut exec = make_num_kernel!(
                        Conv3dForward(
                            *kernel_d as i32,
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_d as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *padding_d as i32,
                            *padding_h as i32,
                            *padding_w as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(CONV3D_TILE_SPEC);
                    exec
                }

                // --- Pooling ---
                Op::AvgPool1d { kernel_l, stride } => {
                    let mut exec = make_num_kernel!(
                        Avgpool1dForward(*kernel_l as i32, *stride as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::AvgPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                } => {
                    let mut exec = make_num_kernel!(
                        Avgpool2dForward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::AvgPool3d {
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    stride_d,
                    stride_h,
                    stride_w,
                } => {
                    let mut exec = make_num_kernel!(
                        Avgpool3dForward(
                            *kernel_d as i32,
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_d as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::MaxPool1d { kernel_l, stride } => {
                    let mut exec = make_num_kernel!(
                        Maxpool1dForward(*kernel_l as i32, *stride as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::MaxPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                } => {
                    let mut exec = make_num_kernel!(
                        Maxpool2dForward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *pad_h as i32,
                            *pad_w as i32,
                            16
                        ),
                        Maxpool2dBackward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *pad_h as i32,
                            *pad_w as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::MaxPool3d {
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    stride_d,
                    stride_h,
                    stride_w,
                } => {
                    let mut exec = make_num_kernel!(
                        Maxpool3dForward(
                            *kernel_d as i32,
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_d as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::LpPool1d {
                    kernel_l, stride, ..
                } => {
                    let mut exec = make_float_kernel!(
                        Lppool1dForward(*kernel_l as i32, *stride as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::LpPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    ..
                } => {
                    let mut exec = make_float_kernel!(
                        Lppool2dForward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
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
                    let mut exec = make_float_kernel!(
                        Lppool3dForward(
                            *kernel_d as i32,
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_d as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }

                // --- Padding ---
                Op::ConstantPad1d {
                    pad_left,
                    pad_right,
                    ..
                } => {
                    let mut exec = make_num_kernel!(
                        ConstantPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ConstantPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                    ..
                } => {
                    let mut exec = make_num_kernel!(
                        ConstantPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ConstantPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                    ..
                } => {
                    let mut exec = make_num_kernel!(
                        ConstantPad3dForward(
                            *pad_d1 as i32,
                            *pad_d2 as i32,
                            *pad_h1 as i32,
                            *pad_h2 as i32,
                            *pad_w1 as i32,
                            *pad_w2 as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ReflectionPad1d {
                    pad_left,
                    pad_right,
                } => {
                    let mut exec = make_num_kernel!(
                        ReflectionPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ReflectionPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                } => {
                    let mut exec = make_num_kernel!(
                        ReflectionPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ReflectionPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                } => {
                    let mut exec = make_num_kernel!(
                        ReflectionPad3dForward(
                            *pad_d1 as i32,
                            *pad_d2 as i32,
                            *pad_h1 as i32,
                            *pad_h2 as i32,
                            *pad_w1 as i32,
                            *pad_w2 as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ReplicationPad1d {
                    pad_left,
                    pad_right,
                } => {
                    let mut exec = make_num_kernel!(
                        ReplicationPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ReplicationPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                } => {
                    let mut exec = make_num_kernel!(
                        ReplicationPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::ReplicationPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                } => {
                    let mut exec = make_num_kernel!(
                        ReplicationPad3dForward(
                            *pad_d1 as i32,
                            *pad_d2 as i32,
                            *pad_h1 as i32,
                            *pad_h2 as i32,
                            *pad_w1 as i32,
                            *pad_w2 as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::CircularPad1d {
                    pad_left,
                    pad_right,
                } => {
                    let mut exec = make_num_kernel!(
                        CircularPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        3,
                        "BLOCK_OL",
                        "OL",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::CircularPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                } => {
                    let mut exec = make_num_kernel!(
                        CircularPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        4,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }
                Op::CircularPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                } => {
                    let mut exec = make_num_kernel!(
                        CircularPad3dForward(
                            *pad_d1 as i32,
                            *pad_d2 as i32,
                            *pad_h1 as i32,
                            *pad_h2 as i32,
                            *pad_w1 as i32,
                            *pad_w2 as i32,
                            8
                        ),
                        node
                    );
                    exec.tile_spec = Some(windowed_last_axis_tile_spec(
                        5,
                        "BLOCK_OW",
                        "OW",
                        "input_ptr",
                        "output_ptr",
                    ));
                    exec
                }

                // --- Activation (D: Num) ---
                Op::Relu => {
                    let mut exec = make_num_kernel!(ReluForward(1024), ReluBackward(1024), node);
                    // `ReluForward::tile_spec` (teenygrad-1nr.18, derived by
                    // `#[tiled_kernel]` from `relu_forward`'s own
                    // `#[tile(...)]`-tagged params) is dtype-independent --
                    // block/extent names and dims don't vary with `D` -- so
                    // any dtype monomorphization gives the same result.
                    exec.tile_spec = Some(ReluForward::<f32>::tile_spec(node.shape.len()));
                    exec
                }

                // --- Activation (D: Float — dtype-dispatched) ---
                Op::Elu { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    EluForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Selu => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    SeluForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Celu { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    CeluForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Gelu => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    GeluForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Mish => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    MishForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Hardtanh { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    HardtanhForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Relu6 => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    Relu6ForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Hardsigmoid => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    HardsigmoidForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Hardswish => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    HardswishForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Hardshrink { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    HardshrinkForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::LeakyRelu { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    LeakyReluForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Threshold { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    ThresholdForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Softsign => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    SoftsignForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Softshrink { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    SoftshrinkForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Softplus { .. } => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    SoftplusForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Sigmoid => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    SigmoidForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Silu => {
                    let mut exec = exec_from(
                        node.shape.clone(),
                        node.dtype,
                        SiluForwardDispatch::dispatch(node.dtype, 1024)?,
                    );
                    // See the `Op::Relu` arm above: `SiluForward::tile_spec`
                    // is dtype-independent, so `exec_from`'s hand-authored
                    // `flat_elementwise_tile_spec` fallback is overridden
                    // here with the macro-derived one (teenygrad-1nr.18).
                    exec.tile_spec = Some(SiluForward::<f32>::tile_spec(node.shape.len()));
                    exec
                }
                Op::Logsigmoid => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    LogsigmoidForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Tanh => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    TanhForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
                Op::Tanhshrink => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    TanhshrinkForwardDispatch::dispatch(node.dtype, 1024)?,
                ),

                // --- Activation (D: Float) ---
                Op::Softmax { .. } => {
                    // BLOCK_SIZE must be >= n_cols (the last dim), rounded up to next power of 2.
                    let n_cols = node.shape.last().and_then(|d| *d).unwrap_or(1024);
                    let block_size = n_cols.next_power_of_two() as i32;
                    make_float_kernel!(SoftmaxForward(block_size), node)
                }

                // --- Upsample ---
                Op::UpsampleNearest2d { scale_h, scale_w } => {
                    make_num_kernel!(
                        UpsampleNearest2dForward(*scale_h as i32, *scale_w as i32, 16),
                        UpsampleNearest2dBackward(*scale_h as i32, *scale_w as i32, 16),
                        node
                    )
                }

                Op::ChannelCat { .. } => {
                    let n_inputs = node.inputs.len();
                    let (name, fwd_src, bwd_src, rop): (
                        String,
                        String,
                        String,
                        Arc<dyn RuntimeOp>,
                    ) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = ChannelCatRuntimeOp::<f32>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = ChannelCatRuntimeOp::<f64>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I8 => {
                            let r = ChannelCatRuntimeOp::<i8>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I16 => {
                            let r = ChannelCatRuntimeOp::<i16>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I32 => {
                            let r = ChannelCatRuntimeOp::<i32>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I64 => {
                            let r = ChannelCatRuntimeOp::<i64>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U8 => {
                            let r = ChannelCatRuntimeOp::<u8>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U16 => {
                            let r = ChannelCatRuntimeOp::<u16>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U32 => {
                            let r = ChannelCatRuntimeOp::<u32>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U64 => {
                            let r = ChannelCatRuntimeOp::<u64>::new(128, n_inputs);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not supported for ChannelCat",
                                other
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: fwd_src,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                Op::ChannelChunk {
                    chunk_c,
                    chunk_offset,
                    ..
                } => {
                    let chunk_c = *chunk_c;
                    let chunk_offset = *chunk_offset;
                    let (name, fwd_src, bwd_src, rop): (
                        String,
                        String,
                        String,
                        Arc<dyn RuntimeOp>,
                    ) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = ChannelChunkRuntimeOp::<f32>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = ChannelChunkRuntimeOp::<f64>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I8 => {
                            let r = ChannelChunkRuntimeOp::<i8>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I16 => {
                            let r = ChannelChunkRuntimeOp::<i16>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I32 => {
                            let r = ChannelChunkRuntimeOp::<i32>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::I64 => {
                            let r = ChannelChunkRuntimeOp::<i64>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U8 => {
                            let r = ChannelChunkRuntimeOp::<u8>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U16 => {
                            let r = ChannelChunkRuntimeOp::<u16>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U32 => {
                            let r = ChannelChunkRuntimeOp::<u32>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::U64 => {
                            let r = ChannelChunkRuntimeOp::<u64>::new(128, chunk_c, chunk_offset);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not supported for ChannelChunk",
                                other
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: fwd_src,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                Op::ChannelBiasAdd { c } => {
                    let c = *c;
                    let (name, fwd_src, bwd_src, rop): (
                        String,
                        String,
                        String,
                        Arc<dyn RuntimeOp>,
                    ) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = ChannelBiasAddRuntimeOp::<f32>::new(128, c);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = ChannelBiasAddRuntimeOp::<f64>::new(128, c);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                r.backward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not supported for ChannelBiasAdd",
                                other
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: fwd_src,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                Op::Add => {
                    make_num_kernel!(ElemwiseAddForward(128), ElemwiseAddBackward(128), node)
                }

                // ── ONNX unary element-wise ops ─────────────────────────────
                Op::Abs => {
                    make_num_kernel!(ElemwiseAbsForward(1024), ElemwiseAbsBackward(1024), node)
                }
                Op::Neg => {
                    make_num_kernel!(ElemwiseNegForward(1024), ElemwiseNegBackward(1024), node)
                }
                Op::Sign => make_num_kernel!(ElemwiseSignForward(1024), node),
                Op::IsNaN => make_float_kernel!(ElemwiseIsnanForward(1024), node),
                Op::Ceil => make_float_kernel!(ElemwiseCeilForward(1024), node),
                Op::Floor => make_float_kernel!(ElemwiseFloorForward(1024), node),
                Op::Sqrt => {
                    make_float_kernel!(ElemwiseSqrtForward(1024), ElemwiseSqrtBackward(1024), node)
                }
                Op::Reciprocal => make_float_kernel!(
                    ElemwiseReciprocalForward(1024),
                    ElemwiseReciprocalBackward(1024),
                    node
                ),
                Op::Exp => {
                    make_float_kernel!(ElemwiseExpForward(1024), ElemwiseExpBackward(1024), node)
                }
                Op::Log => {
                    make_float_kernel!(ElemwiseLogForward(1024), ElemwiseLogBackward(1024), node)
                }
                Op::Erf => {
                    make_float_kernel!(ElemwiseErfForward(1024), ElemwiseErfBackward(1024), node)
                }
                Op::Sin => {
                    make_float_kernel!(ElemwiseSinForward(1024), ElemwiseSinBackward(1024), node)
                }
                Op::Cos => {
                    make_float_kernel!(ElemwiseCosForward(1024), ElemwiseCosBackward(1024), node)
                }
                Op::Tan => {
                    make_float_kernel!(ElemwiseTanForward(1024), ElemwiseTanBackward(1024), node)
                }
                Op::Asin => {
                    make_float_kernel!(ElemwiseAsinForward(1024), ElemwiseAsinBackward(1024), node)
                }
                Op::Acos => {
                    make_float_kernel!(ElemwiseAcosForward(1024), ElemwiseAcosBackward(1024), node)
                }
                Op::Atan => {
                    make_float_kernel!(ElemwiseAtanForward(1024), ElemwiseAtanBackward(1024), node)
                }
                Op::Sinh => {
                    make_float_kernel!(ElemwiseSinhForward(1024), ElemwiseSinhBackward(1024), node)
                }
                Op::Cosh => {
                    make_float_kernel!(ElemwiseCoshForward(1024), ElemwiseCoshBackward(1024), node)
                }
                Op::Asinh => make_float_kernel!(
                    ElemwiseAsinhForward(1024),
                    ElemwiseAsinhBackward(1024),
                    node
                ),
                Op::Acosh => make_float_kernel!(
                    ElemwiseAcoshForward(1024),
                    ElemwiseAcoshBackward(1024),
                    node
                ),
                Op::Atanh => make_float_kernel!(
                    ElemwiseAtanhForward(1024),
                    ElemwiseAtanhBackward(1024),
                    node
                ),
                Op::Round => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Round — implement rounding kernel"
                    ));
                }

                // ── ONNX binary element-wise ops ─────────────────────────────
                Op::Mul => {
                    make_num_kernel!(ElemwiseMulForward(1024), ElemwiseMulBackward(1024), node)
                }
                Op::Sub => {
                    make_num_kernel!(ElemwiseSubForward(1024), ElemwiseSubBackward(1024), node)
                }
                Op::Div => {
                    make_float_kernel!(ElemwiseDivForward(1024), ElemwiseDivBackward(1024), node)
                }
                Op::Pow => {
                    make_float_kernel!(ElemwisePowForward(1024), ElemwisePowBackward(1024), node)
                }
                Op::Mod { .. } => make_float_kernel!(ElemwiseFmodForward(1024), node),
                Op::ElemMin => {
                    make_num_kernel!(ElemwiseMinForward(1024), ElemwiseMinBackward(1024), node)
                }
                Op::ElemMax => {
                    make_num_kernel!(ElemwiseMaxForward(1024), ElemwiseMaxBackward(1024), node)
                }
                Op::ElemMean => {
                    make_float_kernel!(ElemwiseMeanForward(1024), ElemwiseMeanBackward(1024), node)
                }
                Op::ElemSum => {
                    make_num_kernel!(ElemwiseSumForward(1024), ElemwiseSumBackward(1024), node)
                }
                Op::Equal => make_num_kernel!(ElemwiseEqualForward(1024), node),
                Op::Greater => make_num_kernel!(ElemwiseGreaterForward(1024), node),
                Op::GreaterOrEqual => make_num_kernel!(ElemwiseGreaterEqualForward(1024), node),
                Op::Less => make_num_kernel!(ElemwiseLessForward(1024), node),
                Op::LessOrEqual => make_num_kernel!(ElemwiseLessEqualForward(1024), node),
                Op::Where => make_float_kernel!(
                    ElemwiseWhereForward(1024),
                    ElemwiseWhereBackward(1024),
                    node
                ),
                Op::Clip => {
                    let (name, ks, bwd_ks, rop): (String, String, String, Arc<dyn RuntimeOp>) =
                        match node.dtype {
                            DtypeRepr::F32 => {
                                let r = ClipRuntimeOp::<f32>::new(
                                    1024,
                                    f32::NEG_INFINITY,
                                    f32::INFINITY,
                                );
                                (
                                    r.kernel_name().to_string(),
                                    r.forward_source().to_string(),
                                    r.backward_source().to_string(),
                                    Arc::new(r),
                                )
                            }
                            DtypeRepr::F64 => {
                                let r = ClipRuntimeOp::<f64>::new(
                                    1024,
                                    f32::NEG_INFINITY,
                                    f32::INFINITY,
                                );
                                (
                                    r.kernel_name().to_string(),
                                    r.forward_source().to_string(),
                                    r.backward_source().to_string(),
                                    Arc::new(r),
                                )
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "{:?} is not supported for Clip",
                                    other
                                ));
                            }
                        };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_ks,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                // ── ONNX reduction ops ────────────────────────────────────────
                Op::ReduceSum { .. } => make_num_kernel!(ReduceSumForward(1024), node),
                Op::ReduceMean { .. } => make_float_kernel!(ReduceMeanForward(1024), node),
                Op::ReduceMax { .. } => make_num_kernel!(ReduceMaxForward(1024), node),
                Op::ReduceMin { .. } => make_num_kernel!(ReduceMinForward(1024), node),
                Op::ReduceProd { .. } => make_float_kernel!(ReduceProdForward(1024), node),
                Op::ReduceL1 { .. } => make_num_kernel!(ReduceL1Forward(1024), node),
                Op::ReduceL2 { .. } => make_float_kernel!(ReduceL2Forward(1024), node),
                Op::ReduceLogSum { .. } => make_float_kernel!(ReduceLogSumForward(1024), node),
                Op::ReduceLogSumExp { .. } => {
                    make_float_kernel!(ReduceLogSumExpForward(1024), node)
                }
                Op::ReduceSumSquare { .. } => make_num_kernel!(ReduceSumSquareForward(1024), node),
                Op::CumSum { .. } => make_num_kernel!(CumSumForward(1024), node),
                Op::CumProd { .. } => make_num_kernel!(CumProdForward(1024), node),
                Op::GlobalAvgPool => make_float_kernel!(GlobalAvgPoolForward(1024), node),
                Op::GlobalMaxPool => make_float_kernel!(GlobalMaxPoolForward(1024), node),
                Op::ArgMax { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ArgMax — I32Tensor output requires a custom kernel"
                    ));
                }
                Op::ArgMin { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ArgMin — I32Tensor output requires a custom kernel"
                    ));
                }

                // ── Additional activations ────────────────────────────────────
                Op::Swish => {
                    let fwd = SwishForward::new(1024);
                    let nm = fwd.name.to_string();
                    let fwd_src = fwd.source.clone();
                    #[cfg(feature = "training")]
                    let bwd_src = SwishBackward::new(1024).source.clone();
                    let rop: Arc<dyn RuntimeOp> = Arc::new(fwd);
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", nm),
                        name: nm,
                        kernel_source: fwd_src,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::PRelu => {
                    let fwd = PreluForward::new(1024);
                    let nm = fwd.name.to_string();
                    let fwd_src = fwd.source.clone();
                    #[cfg(feature = "training")]
                    let bwd_src = crate::nn::activation::extra::PreluBackward::new(1024)
                        .source
                        .clone();
                    let rop: Arc<dyn RuntimeOp> = Arc::new(fwd);
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", nm),
                        name: nm,
                        kernel_source: fwd_src,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::LogSoftmax { .. } => {
                    let n_cols = node.shape.last().and_then(|d| *d).unwrap_or(1024);
                    let block_size = n_cols.next_power_of_two() as i32;
                    let fwd = LogSoftmaxForward::new(block_size);
                    let nm = fwd.name.to_string();
                    let fwd_src = fwd.source.clone();
                    #[cfg(feature = "training")]
                    let bwd_src = LogSoftmaxBackward::new(block_size).source.clone();
                    let rop: Arc<dyn RuntimeOp> = Arc::new(fwd);
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", nm),
                        name: nm,
                        kernel_source: fwd_src,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::ThresholdedRelu { alpha } => {
                    let alpha_f = *alpha as f32;
                    let (name, ks, bwd_ks, rop): (String, String, String, Arc<dyn RuntimeOp>) =
                        match node.dtype {
                            DtypeRepr::F32 => {
                                let r = ThresholdedReluRuntimeOp::new(1024, alpha_f);
                                (
                                    r.kernel_name().to_string(),
                                    r.forward_source().to_string(),
                                    r.backward_source().to_string(),
                                    Arc::new(r),
                                )
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "{:?} is not supported for ThresholdedRelu",
                                    other
                                ));
                            }
                        };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_ks,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::Shrink { lambd, bias } => {
                    let lambd_f = *lambd as f32;
                    let bias_f = *bias as f32;
                    let (name, ks, bwd_ks, rop): (String, String, String, Arc<dyn RuntimeOp>) =
                        match node.dtype {
                            DtypeRepr::F32 => {
                                let r = ShrinkRuntimeOp::new(1024, lambd_f, bias_f);
                                (
                                    r.kernel_name().to_string(),
                                    r.forward_source().to_string(),
                                    r.backward_source().to_string(),
                                    Arc::new(r),
                                )
                            }
                            other => {
                                return Err(anyhow::anyhow!(
                                    "{:?} is not supported for Shrink",
                                    other
                                ));
                            }
                        };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_ks,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                // ── Matrix ops ────────────────────────────────────────────────
                Op::MatMul | Op::Gemm { .. } => {
                    // A: [M, K], B: [K, N], C (output): [M, N].
                    let m = node.shape.first().copied().flatten();
                    let n = node.shape.last().copied().flatten().unwrap_or(0);
                    let k = node
                        .inputs
                        .first()
                        .and_then(|&i| graph.nodes[i].shape.last().copied().flatten())
                        .unwrap_or(0);
                    let (block_m, block_n, block_k) = pick_gemm_tile_sizes(m, n, k);

                    let (name, ks, rop): (String, String, Arc<dyn RuntimeOp>) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = MatMulRuntimeOp::<f32>::new(block_m, block_n, block_k);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = MatMulRuntimeOp::<f64>::new(block_m, block_n, block_k);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not a Float dtype for MatMul",
                                other
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: Some(MATMUL_TILE_SPEC),
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                // ── ONNX ops that cannot be lowered to a single Triton kernel ─
                Op::Lstm { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Lstm — multi-step recurrent; implement as a custom loop kernel"
                    ));
                }
                Op::Gru { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Gru — multi-step recurrent; implement as a custom loop kernel"
                    ));
                }
                Op::Rnn { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Rnn — multi-step recurrent; implement as a custom loop kernel"
                    ));
                }
                Op::RotaryEmbedding => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::RotaryEmbedding — implement RoPE kernel"
                    ));
                }
                Op::MultiHeadAttention { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::MultiHeadAttention — use flash attention or a custom MHA kernel"
                    ));
                }
                Op::FlexAttention { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::FlexAttention — implement a custom attention kernel supporting an arbitrary score-modification function"
                    ));
                }
                Op::LinearAttention { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::LinearAttention — implement a linear-attention/gated-delta-rule kernel"
                    ));
                }
                Op::CausalConvWithState { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::CausalConvWithState — implement a stateful causal-conv kernel"
                    ));
                }
                Op::Reshape => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Reshape — implement as a strided view or copy kernel"
                    ));
                }
                Op::Transpose { perm } => {
                    // teenygrad-3w0.10: rank-2 only (documented non-goal —
                    // see nn::tensor::transpose's module doc). `perm` is
                    // either empty (default: reverse all axes) or the
                    // explicit rank-2 swap `[1, 0]`; anything else isn't
                    // representable by `transpose_2d_forward`.
                    if !(perm.is_empty() || perm.as_slice() == [1, 0]) {
                        return Err(anyhow::anyhow!(
                            "Op::Transpose: only rank-2 perm=[1, 0] (or empty) is supported, got {:?}",
                            perm
                        ));
                    }
                    // Tensor-descriptor-based kernel: RuntimeOp is hand-written
                    // (TransposeRuntimeOp), same precedent as Op::MatMul below.
                    const BLOCK_M: i32 = 32;
                    const BLOCK_N: i32 = 32;
                    let (name, ks, rop): (String, String, Arc<dyn RuntimeOp>) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = TransposeRuntimeOp::<f32>::new(BLOCK_M, BLOCK_N);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = TransposeRuntimeOp::<f64>::new(BLOCK_M, BLOCK_N);
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "{:?} is not a supported dtype for Op::Transpose",
                                other
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{}_entry_point", name),
                        name,
                        kernel_source: ks,
                        kernel_body: String::new(),
                        pointwise_fuse_block_size: None,
                        tile_spec: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }
                Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Squeeze/Unsqueeze — implement as a zero-copy view"
                    ));
                }
                Op::Concat { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Concat — implement as a multi-input copy kernel"
                    ));
                }
                Op::Split { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Split — implement as a multi-output slice kernel"
                    ));
                }
                Op::Slice => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Slice — implement as a strided-copy kernel"
                    ));
                }
                Op::Gather { .. } | Op::GatherElements { .. } | Op::GatherND { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Gather — implement as an index-gather kernel"
                    ));
                }
                Op::ScatterElements { .. }
                | Op::ScatterND
                | Op::Scatter { .. }
                | Op::TensorScatter => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Scatter — implement as an index-scatter kernel"
                    ));
                }
                Op::Tile => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Tile — implement as a tiled-copy kernel"
                    ));
                }
                Op::Expand => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Expand — implement as a broadcast-copy kernel"
                    ));
                }
                Op::ShapeOf { .. } | Op::SizeOf => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ShapeOf/SizeOf — output is metadata, not tensor data"
                    ));
                }
                Op::Compress { .. } | Op::NonZero => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Compress/NonZero — variable-output ops require stream compaction"
                    ));
                }
                Op::Range => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Range — implement as a fill/arange kernel"
                    ));
                }
                Op::Constant { .. } | Op::ConstantOfShape { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Constant — inline constant; should be materialised before lowering"
                    ));
                }
                Op::Trilu { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Trilu — implement as a triangular mask kernel"
                    ));
                }
                Op::Pad { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Pad — implement as a generic N-D padding kernel"
                    ));
                }
                Op::ReverseSequence { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ReverseSequence — implement as a scatter-copy kernel"
                    ));
                }
                Op::Einsum { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Einsum — parse equation and emit a fused contraction kernel"
                    ));
                }
                Op::Det => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Det — implement via LU decomposition"
                    ));
                }
                Op::QLinearMatMul
                | Op::MatMulInteger
                | Op::ConvInteger { .. }
                | Op::QLinearConv { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Q* quantised matmul/conv — implement quantised compute kernels"
                    ));
                }
                Op::ConvTranspose { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ConvTranspose — implement transposed (gradient) convolution kernel"
                    ));
                }
                Op::DeformConv { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::DeformConv — implement deformable convolution kernel"
                    ));
                }
                Op::Col2Im { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Col2Im — implement col2im (fold) kernel"
                    ));
                }
                Op::Resize { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Resize — implement nearest/bilinear/bicubic resize kernels"
                    ));
                }
                Op::GridSample { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::GridSample — implement bilinear grid sample kernel"
                    ));
                }
                Op::SpaceToDepth { .. } | Op::DepthToSpace { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::SpaceToDepth/DepthToSpace — implement pixel shuffle kernel"
                    ));
                }
                Op::RoiAlign { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::RoiAlign — implement RoI-align pooling kernel"
                    ));
                }
                Op::AffineGrid { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::AffineGrid — implement affine grid generator kernel"
                    ));
                }
                Op::MaxUnpool { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::MaxUnpool — implement max-unpool (scatter with saved indices) kernel"
                    ));
                }
                Op::CenterCropPad { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::CenterCropPad — implement center-crop-pad kernel"
                    ));
                }
                Op::NonMaxSuppression { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::NonMaxSuppression — implement NMS kernel"
                    ));
                }
                Op::TopK { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::TopK — implement radix sort / parallel selection kernel"
                    ));
                }
                Op::Unique { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Unique — implement stream-compaction unique kernel"
                    ));
                }
                Op::EyeLike { .. }
                | Op::OneHot { .. }
                | Op::Bernoulli { .. }
                | Op::RandomUniformLike { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::EyeLike/OneHot/Bernoulli/RandomUniformLike — implement generation kernels"
                    ));
                }
                Op::And | Op::Or | Op::Xor => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::And/Or/Xor — implement boolean logical kernels"
                    ));
                }
                Op::BitShift { .. }
                | Op::BitwiseAnd
                | Op::BitwiseOr
                | Op::BitwiseXor
                | Op::BitwiseNot
                | Op::Not => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Bitwise* — implement integer bitwise kernels"
                    ));
                }
                Op::QuantizeLinear { .. }
                | Op::DequantizeLinear { .. }
                | Op::DynamicQuantizeLinear => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Quantize/Dequantize — implement quantisation kernels"
                    ));
                }
                Op::LRN { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::LRN — implement local response normalisation kernel"
                    ));
                }
                Op::MeanVarianceNormalization { .. } | Op::LpNormalization { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::MvnNorm/LpNorm — implement normalisation kernels"
                    ));
                }
                Op::Dft { .. }
                | Op::Stft
                | Op::MelWeightMatrix
                | Op::HannWindow { .. }
                | Op::BlackmanWindow { .. }
                | Op::HammingWindow { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::DFT/STFT/Window — implement signal processing kernels"
                    ));
                }
                Op::NegativeLogLikelihoodLoss { .. } | Op::SoftmaxCrossEntropyLoss { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::NllLoss/SoftmaxCELoss — implement loss kernels"
                    ));
                }
                Op::SequenceAt
                | Op::SequenceConstruct
                | Op::SequenceEmpty
                | Op::SequenceErase
                | Op::SequenceInsert
                | Op::SequenceLength
                | Op::SequenceMap
                | Op::SplitToSequence { .. }
                | Op::ConcatFromSequence { .. }
                | Op::OptionalGetElement
                | Op::OptionalHasElement
                | Op::Loop
                | Op::Scan { .. }
                | Op::If => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Sequence/Control-flow — not lowerable to single Triton kernels"
                    ));
                }
                Op::Adagrad | Op::Adam | Op::Momentum | Op::Gradient => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::OnnxOptimizer — use teenygrad's own optimizer kernels instead"
                    ));
                }
                Op::StringNormalizer
                | Op::RegexFullMatch { .. }
                | Op::StringConcat
                | Op::StringSplit
                | Op::TfIdfVectorizer
                | Op::LabelEncoder
                | Op::ArrayFeatureExtractor
                | Op::Binarizer { .. }
                | Op::TreeEnsemble
                | Op::ImageDecoder => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::String/ClassicalML — not GPU-lowerable"
                    ));
                }
                Op::CastLike | Op::BitCast { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::CastLike/BitCast — implement dtype-cast kernels"
                    ));
                }
                Op::Cast { to } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Cast to {:?} — implement dtype-cast kernel",
                        to
                    ));
                }
                Op::Identity => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Identity — implement zero-copy pass-through kernel"
                    ));
                }
                Op::Dropout { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Dropout — implement inference pass-through / training dropout kernel"
                    ));
                }
                Op::IsInf { .. } => {
                    return Err(anyhow::anyhow!("TODO: Op::IsInf — implement isinf kernel"));
                }
                Op::Hardmax { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Hardmax — implement argmax + one-hot kernel"
                    ));
                }

                Op::Attention {
                    c,
                    num_heads,
                    key_dim,
                } => {
                    // PSA attention is decomposed into 13 sub-nodes inline below.
                    // The match arm is unreachable because the pre-match block handles it
                    // and calls `continue`. We return an error here as a safety net.
                    let _ = (c, num_heads, key_dim);
                    return Err(anyhow::anyhow!(
                        "Op::Attention reached the match arm — this should not happen"
                    ));
                }

                Op::Custom { data } => match data.0.lower() {
                    Some((name, kernel_source, entry_point, runtime_op)) => {
                        #[cfg(feature = "training")]
                        let backward_kernel_source = data.0.lower_backward_source();
                        #[cfg(feature = "training")]
                        let backward_entry_point = if backward_kernel_source.is_empty() {
                            String::new()
                        } else {
                            format!("{name}_backward_entry_point")
                        };
                        Box::new(KernelExecutable {
                            name,
                            kernel_source,
                            entry_point,
                            shape: node.shape.clone(),
                            dtype: node.dtype,
                            runtime_op,
                            #[cfg(feature = "training")]
                            backward_kernel_source,
                            kernel_body: String::new(),
                            pointwise_fuse_block_size: None,
                            tile_spec: None,
                            #[cfg(feature = "training")]
                            backward_entry_point,
                        })
                    }
                    None => {
                        return Err(anyhow::anyhow!(
                            "custom op '{}' is not handled — implement CustomOp::lower()",
                            data.name()
                        ));
                    }
                },
            };

            let dag_idx = dag.add_node(executable);
            graph_to_dag[node_index] = dag_idx;

            for &input_graph_idx in &node.inputs {
                dag.add_edge(graph_to_dag[input_graph_idx], dag_idx);
            }
        }

        Ok((dag, graph_to_dag, graph.clone()))
    }
}

impl<'a> Lowering<'a> for TritonLowering {
    fn lower(&self, graph: &Graph, mode: LoweringMode) -> Result<Dag<Box<dyn ExecutableOp>>> {
        TritonLowering::lower_with_mapping(self, graph, mode).map(|(dag, _, _)| dag)
    }

    fn lower_with_mapping(
        &self,
        graph: &Graph,
        mode: LoweringMode,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>, Graph)> {
        TritonLowering::lower_with_mapping(self, graph, mode)
    }

    /// Conv2d-with-bias splits one graph node into two DAG nodes (conv + biasadd).
    /// `graph_to_dag[graph_idx]` already points at the biasadd DAG node; here we
    /// propagate the same name to the conv DAG node (biasadd_dag_idx - 1) so that
    /// the conv weight parameter can be loaded under the same name prefix.
    fn extra_dag_names(&self, graph: &Graph, graph_to_dag: &[usize]) -> Vec<(usize, String)> {
        let mut extra = Vec::new();
        for (graph_idx, node) in graph.nodes.iter().enumerate() {
            if let Op::Conv2d { has_bias: true, .. } = &node.op
                && let Some(name) = graph.names.get(&graph_idx)
            {
                let biasadd_dag_idx = graph_to_dag[graph_idx];
                if biasadd_dag_idx > 0 {
                    extra.push((biasadd_dag_idx - 1, name.clone()));
                }
            }
        }
        extra
    }
}

#[cfg(test)]
mod relu_silu_tile_spec_tests {
    //! teenygrad-1nr.18: `ReluForward`/`SiluForward::tile_spec()` (emitted
    //! by `#[tiled_kernel]` from `relu_forward`/`silu_forward`'s own
    //! `#[tile(block=BLOCK_SIZE,extent=n_elements)]`-tagged `x`/`y` params)
    //! is what the `Op::Relu`/`Op::Silu` lowering arms attach today,
    //! replacing the previously hand-authored `flat_elementwise_tile_spec`
    //! call for these two ops specifically. Exercises the real lowering
    //! path (`lower_unary_op`), not just the macro output in isolation, so
    //! a mismatch between the attribute and the real signature -- e.g. a
    //! future rename of `x`/`y`/`BLOCK_SIZE`/`n_elements` without updating
    //! the attribute to match -- would fail here instead of only silently
    //! producing a stale spec.

    use super::*;

    fn assert_flat_unary_spec(spec: KernelTileSpec, in_param: &str, out_param: &str) {
        assert_eq!(spec.loop_spec, None);
        assert_eq!(spec.inputs.len(), 1);
        assert_eq!(spec.outputs.len(), 1);
        assert_eq!(spec.inputs[0].param, in_param);
        assert_eq!(spec.outputs[0].param, out_param);
        for tensor in [spec.inputs[0], spec.outputs[0]] {
            assert_eq!(tensor.rank, 1);
            assert_eq!(tensor.reduction_axis, None);
            assert_eq!(tensor.untiled_dims, &[] as &[&str]);
            assert_eq!(tensor.axes.len(), 1);
            assert_eq!(tensor.axes[0].dims, &[0]);
            assert_eq!(tensor.axes[0].block_const, "BLOCK_SIZE");
            assert_eq!(tensor.axes[0].extent_param, "n_elements");
            assert_eq!(tensor.axes[0].window, None);
            assert_eq!(tensor.axes[0].divide_by, None);
        }
    }

    #[test]
    fn relu_tile_spec_matches_its_real_tile_tagged_signature() {
        let lowering = TritonLowering::default();
        let exec = lowering
            .lower_unary_op(&Op::Relu, DtypeRepr::F32)
            .expect("Relu should lower");
        let spec = exec
            .tile_spec
            .expect("relu_forward declares #[tile(...)] on x/y");
        assert_flat_unary_spec(spec, "x", "y");
    }

    #[test]
    fn silu_tile_spec_matches_its_real_tile_tagged_signature() {
        let lowering = TritonLowering::default();
        let exec = lowering
            .lower_unary_op(&Op::Silu, DtypeRepr::F32)
            .expect("Silu should lower");
        let spec = exec
            .tile_spec
            .expect("silu_forward declares #[tile(...)] on x/y");
        assert_flat_unary_spec(spec, "x", "y");
    }
}

#[cfg(test)]
mod conv2d_grid_spec_tests {
    //! teenygrad-1nr.19: `Conv2dForward::tile_spec()`/`grid_spec()` are
    //! generated straight from `conv2d_forward`'s own multi-axis
    //! `#[tile(...)]`-tagged `x_ptr`/`y_ptr` (`kernels/teeny-kernels/src/nn/conv/conv2d.rs`)
    //! -- the first real (not synthetic) kernel to use the metadata-only,
    //! raw-pointer form of `#[tile(...)]`, since `conv2d_forward`'s body
    //! (real, hand-written `pid` decode/loop) is untouched. `tile_spec()`
    //! reproduces what the hand-authored `CONV2D_TILE_SPEC` const used to
    //! say -- see `Op::Conv2d`'s lowering arm, which now layers
    //! `CONV2D_LOOP_SPEC` on top of this macro-derived value instead of
    //! hand-authoring the whole thing.

    use super::*;
    use teeny_core::model::{GridAxisBinding, GridDim};

    #[test]
    fn tile_spec_matches_the_real_tagged_signature() {
        let spec = Conv2dForward::<f32>::tile_spec();
        assert_eq!(spec.loop_spec, None);

        assert_eq!(spec.inputs.len(), 1);
        let x = spec.inputs[0];
        assert_eq!(x.param, "x_ptr");
        assert_eq!(x.rank, 4);
        assert_eq!(x.axes, &[] as &[TileAxisBinding]);
        assert_eq!(x.untiled_dims, &["B", "C_IN", "H", "W"]);

        assert_eq!(spec.outputs.len(), 1);
        let y = spec.outputs[0];
        assert_eq!(y.param, "y_ptr");
        assert_eq!(y.rank, 4);
        assert_eq!(y.untiled_dims, &["B", "C_OUT", "OH"]);
        assert_eq!(y.axes.len(), 1);
        assert_eq!(y.axes[0].dims, &[3]);
        assert_eq!(y.axes[0].block_const, "BLOCK_OW");
        assert_eq!(y.axes[0].extent_param, "OW");
        assert_eq!(y.axes[0].window, None);
        assert_eq!(y.axes[0].divide_by, None);
    }

    #[test]
    fn grid_spec_reflects_the_real_pid_decode_order_and_shape() {
        // conv2d_forward's own body decodes one flat `pid` (all axes on
        // GridDim::X) outermost-to-innermost as (b, c_out, oh, ow_tile) --
        // see that function's own comment on its `pid` decode.
        let spec = Conv2dForward::<f32>::grid_spec();
        assert_eq!(spec.axes.len(), 4);

        let names: Vec<&str> = spec.axes.iter().map(|a| a.name).collect();
        assert_eq!(names, ["B", "C_OUT", "OH", "OW"]);
        assert!(spec.axes.iter().all(|a| matches!(a.dim, GridDim::X)));

        let [b, c_out, oh, ow] = [spec.axes[0], spec.axes[1], spec.axes[2], spec.axes[3]];
        for untiled in [b, c_out, oh] {
            assert_eq!(untiled.block_const, None);
        }
        assert_eq!(b.extent_factors, &["_B"]);
        assert_eq!(c_out.extent_factors, &["C_OUT"]);
        assert_eq!(oh.extent_factors, &["OH"]);

        assert_eq!(ow.block_const, Some("BLOCK_OW"));
        assert_eq!(ow.extent_factors, &["OW", "BLOCK_OW"]);
    }

    #[test]
    fn grid_spec_axis_matches_a_grid_axis_binding_directly() {
        // Sanity check the type itself is what a future consumer would
        // actually construct/compare against.
        let expected_ow_axis = GridAxisBinding {
            name: "OW",
            extent_factors: &["OW", "BLOCK_OW"],
            dim: GridDim::X,
            block_const: Some("BLOCK_OW"),
        };
        let spec = Conv2dForward::<f32>::grid_spec();
        assert_eq!(spec.axes[3], expected_ow_axis);
    }
}
