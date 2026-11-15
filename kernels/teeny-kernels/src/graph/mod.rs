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
    graph::{
        ConstantMap, ConstantTensor, DtypeRepr, EINSUM_MAX_RANK, EinsumPlan, Graph, GraphNode, Op,
        ResizeCoords, ResizeMode, ResizeNearest, Shape, SlicePlan, einsum_plan, matmul_plan,
        reduce_axes, reduce_transpose_perm, resize_axis_plan, slice_plan,
    },
    model::{ExecutableOp, Lowering, LoweringMode, RuntimeOp},
    utils::dag::Dag,
};

pub mod optimizer;

pub use optimizer::{Anduin, GraphOptimizer, PointwiseFuse};

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
            LeakyReluForward, LeakyReluRuntimeOp, SoftplusForward, SoftplusForwardDispatch,
            SoftshrinkForward, SoftshrinkForwardDispatch, SoftsignForward, SoftsignForwardDispatch,
            ThresholdForward, ThresholdForwardDispatch,
        },
        relu::{ReluBackward, ReluForward},
        sigmoid::{
            LogsigmoidForward, LogsigmoidForwardDispatch, SigmoidForwardDispatch,
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
        linear::{LinearBackward, LinearForward},
    },
    norm::{
        batchnorm::{BatchNorm2dNchwInferenceRuntimeOp, BatchNormForwardInference},
        groupnorm::GroupNormForwardInference,
        instancenorm::{InstanceNormForwardInference, InstanceNormRuntimeOp},
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
        constant::ConstantRuntimeOp,
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
            ElemwiseAtanhBackward, ElemwiseAtanhForward, ElemwiseCeilForward, ElemwiseCopyBackward,
            ElemwiseCopyForward, ElemwiseCosBackward, ElemwiseCosForward, ElemwiseCoshBackward,
            ElemwiseCoshForward, ElemwiseErfBackward, ElemwiseErfForward, ElemwiseExpBackward,
            ElemwiseExpForward, ElemwiseFloorForward, ElemwiseIsnanForward, ElemwiseLogBackward,
            ElemwiseLogForward, ElemwiseNegBackward, ElemwiseNegForward,
            ElemwiseReciprocalBackward, ElemwiseReciprocalForward, ElemwiseSignForward,
            ElemwiseSinBackward, ElemwiseSinForward, ElemwiseSinhBackward, ElemwiseSinhForward,
            ElemwiseSqrtBackward, ElemwiseSqrtForward, ElemwiseTanBackward, ElemwiseTanForward,
        },
        gather::{GatherDynForward, GatherDynRuntimeOp, GatherForward, GatherRuntimeOp},
        reduction::{
            CumProdForward, CumSumForward, GlobalAvgPoolForward, GlobalMaxPoolForward,
            ReduceL1Forward, ReduceL2Forward, ReduceLogSumExpForward, ReduceLogSumForward,
            ReduceMaxForward, ReduceMeanForward, ReduceMinForward, ReduceProdForward,
            ReduceSumForward, ReduceSumSquareForward,
        },
        strided_copy::{StridedCopyForward, StridedCopyRuntimeOp},
        tile::{MAX_RANK, TileForward, TileRuntimeOp},
        upsample_nearest2d::{UpsampleNearest2dBackward, UpsampleNearest2dForward},
    },
};

use crate::math::einsum::{EinsumForward, EinsumRuntimeOp};
use crate::math::gemm::MatMulRuntimeOp;
use crate::nn::conv::conv_transpose2d::ConvTranspose2dRuntimeOp;
use crate::nn::pad::pad_nd::PadNdRuntimeOp;
use crate::nn::tensor::concat::ConcatRuntimeOp;
use crate::nn::tensor::fill::FillRuntimeOp;
use crate::nn::tensor::grid_sample::GridSampleRuntimeOp;
use crate::nn::tensor::logical::{LogicalAndForward, LogicalOrForward, LogicalXorForward};
use crate::nn::tensor::resize::{MAX_AXES as RESIZE_MAX_AXES, ResizeAxis, ResizeRuntimeOp};
use crate::nn::tensor::scatter::{ScatterElementsRuntimeOp, ScatterNdRuntimeOp};

use crate::errors::{Error, Result};

#[cfg(feature = "training")]
use crate::nn::norm::batchnorm::{
    BatchNorm2dNchwBackward, BatchNormNormalizeForward, BatchNormNormalizeRuntimeOp,
    BatchNormStatsForward, BatchNormStatsRuntimeOp,
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
// Bool tensors are one byte per element, so they take the u8 instantiation: sound for
// the copies and data movement a bool goes through, and ONNX defines no arithmetic on bools.
macro_rules! make_num_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let (name, ks, body, probe_bs, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I8  => { let k = $K::<i8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I16 => { let k = $K::<i16>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I32 => { let k = $K::<i32>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::I64 => { let k = $K::<i64>::new($($arg),*); let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
            DtypeRepr::U8 | DtypeRepr::Bool => { let k = $K::<u8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
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
            DtypeRepr::U8 | DtypeRepr::Bool => { let k = $K::<u8>::new($($arg),*);  let nm = k.name.to_string(); let body = k.kernel_source.clone(); let src = k.source.clone(); let probe_bs = k.pointwise_fuse_probe().map(|p| p.block_size); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, body, probe_bs, r) }
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
            DtypeRepr::U8 | DtypeRepr::Bool  => { let k = $Bwd::<u8>::new($($barg),*);  (k.name.to_string(), k.source.clone()) }
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
/// produced by a `#[kernel(dtypes = [..])]` dispatcher.
fn exec_from(
    shape: Shape,
    dtype: DtypeRepr,
    inst: teeny_core::model::KernelInstance,
) -> Box<KernelExecutable> {
    Box::new(KernelExecutable {
        entry_point: format!("{}_entry_point", inst.name),
        name: inst.name,
        kernel_source: inst.source,
        kernel_body: inst.kernel_body,
        pointwise_fuse_block_size: inst.pointwise_fuse_block_size,
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
//
// A stub here does not mean the op cannot run: the `#[kernel]` macro generates a
// dispatch that requires one, so a kernel whose arguments the lowering supplies
// through a wrapper of its own keeps its stub as well. InstanceNormForwardInference
// and LeakyReluForward are both like that, and their wrappers are
// `InstanceNormRuntimeOp` and `LeakyReluRuntimeOp`.
impl_stub_runtime_op_float!(InstanceNormForwardInference);
impl_stub_runtime_op_float!(LeakyReluForward);
impl_stub_runtime_op_float!(BatchNormForwardInference);
impl_stub_runtime_op_float!(LayerNormForwardInference);
impl_stub_runtime_op_float!(RmsNormForward);
impl_stub_runtime_op_float!(GroupNormForwardInference);

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
impl_stub_runtime_op_float!(CeluForward);
impl_stub_runtime_op_float!(HardtanhForward);
impl_stub_runtime_op_float!(HardshrinkForward);
impl_stub_runtime_op_float!(ThresholdForward);
impl_stub_runtime_op_float!(SoftshrinkForward);
impl_stub_runtime_op_float!(SoftplusForward);

// Element-wise activations of signature `(x, y, n_elements)`: one input, no
// parameters, a 1-D grid of `BLOCK_SIZE` lanes.
macro_rules! impl_unary_runtime_op_float {
    ($T:ident) => {
        impl<D: teeny_core::dtype::Float + Send + Sync + 'static> RuntimeOp for $T<D> {
            fn n_activation_inputs(&self) -> usize {
                1
            }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
                Vec::new()
            }
            fn pack_args(
                &self,
                inputs: &[(teeny_core::model::RawPtr, &[usize])],
                _params: &[teeny_core::model::RawPtr],
                output: teeny_core::model::RawPtr,
                output_shape: &[usize],
                _output_row_stride: i32,
                visitor: &mut dyn teeny_core::device::program::ArgVisitor,
            ) {
                let n: usize = output_shape.iter().product();
                visitor.visit_ptr(inputs[0].0);
                visitor.visit_ptr(output);
                visitor.visit_i32(n as i32);
            }
            fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
                let n: usize = output_shape.iter().product();
                [n.div_ceil(self.block_size as usize) as u32, 1, 1]
            }
        }
    };
}

impl_unary_runtime_op_float!(SeluForward);
impl_unary_runtime_op_float!(MishForward);
impl_unary_runtime_op_float!(Relu6Forward);
impl_unary_runtime_op_float!(HardsigmoidForward);
impl_unary_runtime_op_float!(HardswishForward);
impl_unary_runtime_op_float!(SoftsignForward);
impl_unary_runtime_op_float!(LogsigmoidForward);
impl_unary_runtime_op_float!(TanhshrinkForward);

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
pub struct TritonLowering {
    /// Optional graph rewrite run before Op→kernel lowering (e.g. [`Anduin`]).
    optimizer: Option<Arc<dyn GraphOptimizer>>,
    /// Materialised constant values, keyed by graph node index.
    constants: Arc<ConstantMap>,
}

impl std::fmt::Debug for TritonLowering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TritonLowering")
            .field("optimizer", &self.optimizer.as_ref().map(|o| o.name()))
            .field("constants", &self.constants.len())
            .finish()
    }
}

impl TritonLowering {
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects a [`GraphOptimizer`] (e.g. [`Anduin`]) to rewrite the graph before
    /// lowering. Without an optimizer, the input graph is lowered as-is.
    pub fn with_optimizer(mut self, optimizer: impl GraphOptimizer + 'static) -> Self {
        self.optimizer = Some(Arc::new(optimizer));
        self
    }

    /// Supplies materialised constant values, as produced by a pass such as
    /// `teeny_onnx::materialize::materialize_constants`.
    ///
    /// Lowering uses them for two things. A constant consumed only in a shape-operand
    /// position (see [`Op::shape_operand_positions`]) is read here rather than
    /// allocated, so it never becomes a DAG node and its edges are dropped: that is
    /// what lets `Reshape`, `Slice` and friends lower at all, since their shape
    /// operand would otherwise have to be lowered as tensor data. A constant that
    /// does flow data still needs a buffer, and lowering reports it by name.
    pub fn with_constants(mut self, constants: ConstantMap) -> Self {
        self.constants = Arc::new(constants);
        self
    }

    /// The materialised value of a graph node, if one was supplied.
    pub fn constant(&self, node_index: usize) -> Option<&ConstantTensor> {
        self.constants.get(&node_index)
    }
}

/// Builds the executable for a constant that flows tensor data.
///
/// Shape metadata constants never reach here: lowering elides them (see
/// [`elidable_constants`]). What remains is a literal the exporter wrote into the
/// graph that something reads as a buffer, so its bytes have to reach the device.
/// [`ConstantRuntimeOp`] carries them into a parameter slot, which both drivers
/// populate from `param_init_data` right after allocation.
fn constant_executable(
    dtype: DtypeRepr,
    shape: Vec<usize>,
    bytes: Vec<u8>,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    macro_rules! build {
        ($ty:ty) => {{
            let rop = ConstantRuntimeOp::<$ty>::new(1024, shape.clone(), bytes.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Op::Constant {what}: {} bytes do not fill a {dtype:?} tensor of \
                         shape {shape:?}; the materialised value and the op disagree",
                        bytes.len(),
                    )
                })?;
            let kernel = rop.kernel();
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match dtype {
        // A bool tensor is one byte per element, and a Bool buffer is allocated at that
        // same width, so its bytes are already a valid `u8` buffer and the u8 kernel
        // copies them verbatim. `bool` is deliberately not a `Num` dtype: making it one
        // would let arithmetic kernels instantiate over it, which has no meaning. The
        // aliasing is sound only because this kernel *moves* bytes without computing on
        // them, so it stays local to the constant path rather than going in the dtype
        // dispatch macros. The node keeps its Bool dtype, so consumers still see a bool.
        DtypeRepr::Bool => build!(u8),
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I8 => build!(i8),
        DtypeRepr::I16 => build!(i16),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        DtypeRepr::U8 => build!(u8),
        DtypeRepr::U16 => build!(u16),
        DtypeRepr::U32 => build!(u32),
        DtypeRepr::U64 => build!(u64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::Constant {what}: {other:?} is not a supported dtype, so its \
                 values cannot be put in a device buffer"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: shape.into_iter().map(Some).collect(),
        dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Whether an elementwise binary node's operands all have the output's shape.
///
/// When they do, the flat kernels are right and cheaper. When they do not, reading
/// both operands at the output's offsets runs off the end of the shorter one, so the
/// broadcasting kernel is the only correct choice.
fn needs_broadcast(graph: &Graph, node: &GraphNode) -> bool {
    let Some(out): Option<Vec<usize>> = node.shape.iter().copied().collect() else {
        return false;
    };
    node.inputs.iter().any(|&src| {
        graph.nodes[src]
            .shape
            .iter()
            .copied()
            .collect::<Option<Vec<usize>>>()
            .is_none_or(|s| s != out)
    })
}

/// Which arithmetic a broadcasting elementwise node performs.
#[derive(Clone, Copy)]
enum BroadcastOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BroadcastOp {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
        }
    }
}

/// Builds the executable for an elementwise binary whose operands need broadcasting.
///
/// The operands are aligned to the output's rank first, by prepending length-1 axes,
/// which is how ONNX lines up shapes of different ranks.
fn broadcast_binary_executable(
    graph: &Graph,
    node: &GraphNode,
    which: BroadcastOp,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    use crate::nn::tensor::broadcast_binary::{
        BroadcastAddForward, BroadcastBinaryRuntimeOp, BroadcastDivForward, BroadcastMulForward,
        BroadcastSubForward,
    };

    let op = which.name();
    let out_shape = concrete_shape(&node.shape, op, "output", what)?;
    let rank = out_shape.len();
    if node.inputs.len() != 2 {
        return Err(anyhow::anyhow!(
            "Op::{op} {what}: it has {} operands, not 2",
            node.inputs.len()
        ));
    }
    let mut aligned = Vec::with_capacity(2);
    for (pos, &src) in node.inputs.iter().enumerate() {
        // An empty operand shape is read as a rank-0 scalar here, where elsewhere it is
        // refused. The IR cannot tell a scalar from a shape no pass worked out
        // (teenygrad#129), and a scalar operand is exactly what this path is for: the
        // export divides a whole tensor by one. What makes it safe to read it that way
        // is that the output's shape above is required to be concrete, and shape
        // inference resolves an operand before the node that consumes it, so a node
        // with a known output and an unresolved operand is not something it produces.
        let shape: Vec<usize> = graph.nodes[src]
            .shape
            .iter()
            .copied()
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Op::{op} {what}: operand {pos} has a dynamic extent; this kernel \
                     needs the strides when the graph is lowered"
                )
            })?;
        if shape.len() > rank {
            return Err(anyhow::anyhow!(
                "Op::{op} {what}: operand {pos} has rank {} but the output has rank \
                 {rank}",
                shape.len()
            ));
        }
        // ONNX lines shapes up from the right, so a shorter operand gains length-1
        // axes in front.
        let mut padded = vec![1usize; rank - shape.len()];
        padded.extend_from_slice(&shape);
        aligned.push(padded);
    }

    macro_rules! build_bcast {
        ($ty:ty) => {{
            let (name, source, body, erased): (String, String, String, Arc<dyn RuntimeOp>) =
                match which {
                    BroadcastOp::Add => {
                        let k = BroadcastAddForward::<$ty>::new(128);
                        let (n, s, b) = (
                            k.name.to_string(),
                            k.source.clone(),
                            k.kernel_source.clone(),
                        );
                        let rop = BroadcastBinaryRuntimeOp::<_, $ty>::new(
                            k,
                            128,
                            &out_shape,
                            &aligned[0],
                            &aligned[1],
                        );
                        (
                            n,
                            s,
                            b,
                            Arc::new(rop.ok_or_else(|| {
                                broadcast_refusal(op, what, &out_shape, &aligned)
                            })?),
                        )
                    }
                    BroadcastOp::Sub => {
                        let k = BroadcastSubForward::<$ty>::new(128);
                        let (n, s, b) = (
                            k.name.to_string(),
                            k.source.clone(),
                            k.kernel_source.clone(),
                        );
                        let rop = BroadcastBinaryRuntimeOp::<_, $ty>::new(
                            k,
                            128,
                            &out_shape,
                            &aligned[0],
                            &aligned[1],
                        );
                        (
                            n,
                            s,
                            b,
                            Arc::new(rop.ok_or_else(|| {
                                broadcast_refusal(op, what, &out_shape, &aligned)
                            })?),
                        )
                    }
                    BroadcastOp::Mul => {
                        let k = BroadcastMulForward::<$ty>::new(128);
                        let (n, s, b) = (
                            k.name.to_string(),
                            k.source.clone(),
                            k.kernel_source.clone(),
                        );
                        let rop = BroadcastBinaryRuntimeOp::<_, $ty>::new(
                            k,
                            128,
                            &out_shape,
                            &aligned[0],
                            &aligned[1],
                        );
                        (
                            n,
                            s,
                            b,
                            Arc::new(rop.ok_or_else(|| {
                                broadcast_refusal(op, what, &out_shape, &aligned)
                            })?),
                        )
                    }
                    BroadcastOp::Div => {
                        let k = BroadcastDivForward::<$ty>::new(128);
                        let (n, s, b) = (
                            k.name.to_string(),
                            k.source.clone(),
                            k.kernel_source.clone(),
                        );
                        let rop = BroadcastBinaryRuntimeOp::<_, $ty>::new(
                            k,
                            128,
                            &out_shape,
                            &aligned[0],
                            &aligned[1],
                        );
                        (
                            n,
                            s,
                            b,
                            Arc::new(rop.ok_or_else(|| {
                                broadcast_refusal(op, what, &out_shape, &aligned)
                            })?),
                        )
                    }
                };
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build_bcast!(f32),
        DtypeRepr::F64 => build_bcast!(f64),
        DtypeRepr::I32 => build_bcast!(i32),
        DtypeRepr::I64 => build_bcast!(i64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::{op} {what}: {other:?} is not a supported dtype for a broadcast"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// The error for operand shapes that are not a broadcast of the output's.
fn broadcast_refusal(
    op: &str,
    what: &str,
    out_shape: &[usize],
    aligned: &[Vec<usize>],
) -> anyhow::Error {
    anyhow::anyhow!(
        "Op::{op} {what}: operands {:?} and {:?} do not broadcast to {out_shape:?}; an \
         operand's extent on an axis has to be the output's or 1",
        aligned[0],
        aligned[1]
    )
}

/// Builds the executable for `ScatterElements`.
///
/// `indices` and `updates` have `data`'s rank and the same shape as each other, and
/// agree with `data` on every axis but the scattered one. All of that is checked: an
/// operand that disagrees off-axis would have the kernel read its elements in the
/// wrong order, which is not something the device can report.
///
/// `indices`, when the lowering has the operand's value, is checked for range here as
/// well; see [`check_scatter_indices`].
fn scatter_elements_executable(
    graph: &Graph,
    node: &GraphNode,
    axis: i64,
    indices_value: Option<&ConstantTensor>,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    if node.inputs.len() != 3 {
        return Err(anyhow::anyhow!(
            "Op::ScatterElements {what}: it has {} operands, not 3 (data, indices, \
             updates)",
            node.inputs.len()
        ));
    }
    let operand = |pos: usize, name: &str| -> Result<Vec<usize>> {
        concrete_shape(
            &graph.nodes[node.inputs[pos]].shape,
            "ScatterElements",
            name,
            what,
        )
    };
    let data = operand(0, "data")?;
    let indices = operand(1, "indices")?;
    let updates = operand(2, "updates")?;
    let out = concrete_shape(&node.shape, "ScatterElements", "output", what)?;

    let rank = data.len();
    let ax = if axis < 0 { axis + rank as i64 } else { axis };
    let ax = usize::try_from(ax)
        .ok()
        .filter(|&a| a < rank)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Op::ScatterElements {what}: axis {axis} is outside a rank-{rank} tensor"
            )
        })?;
    if out != data {
        return Err(anyhow::anyhow!(
            "Op::ScatterElements {what}: its output is {out:?} but `data` is {data:?}"
        ));
    }
    if indices != updates {
        return Err(anyhow::anyhow!(
            "Op::ScatterElements {what}: `indices` is {indices:?} and `updates` is \
             {updates:?}; they have to match"
        ));
    }
    if indices.len() != rank {
        return Err(anyhow::anyhow!(
            "Op::ScatterElements {what}: `indices` has rank {} but `data` has rank \
             {rank}",
            indices.len()
        ));
    }
    for (a, (&have, &want)) in indices.iter().zip(data.iter()).enumerate() {
        if a != ax && have != want {
            return Err(anyhow::anyhow!(
                "Op::ScatterElements {what}: `indices` has {have} on axis {a} but \
                 `data` has {want}; only axis {ax} may differ"
            ));
        }
    }
    if graph.nodes[node.inputs[1]].dtype != DtypeRepr::I64 {
        return Err(anyhow::anyhow!(
            "Op::ScatterElements {what}: `indices` is {:?}; the kernel reads them as \
             I64, which is what ONNX gives",
            graph.nodes[node.inputs[1]].dtype
        ));
    }

    if let Some(value) = indices_value {
        check_scatter_indices("ScatterElements", what, value, &[ax], &[data[ax]])?;
    }

    let inner: usize = data[ax + 1..].iter().product();
    let n_data: usize = data.iter().product();
    let n_updates: usize = updates.iter().product();

    macro_rules! build {
        ($ty:ty) => {{
            let rop = ScatterElementsRuntimeOp::<$ty>::new(
                1024,
                data[ax],
                updates[ax],
                inner,
                n_data,
                n_updates,
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Op::ScatterElements {what}: an axis of {} over an inner block of \
                     {inner} gives nothing to index",
                    data[ax]
                )
            })?;
            let kernel = rop.kernel();
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::ScatterElements {what}: {other:?} is not a supported dtype"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Checks a scatter's constant indices against the axes they index.
///
/// Index `i` addresses axis `axes[i % axes.len()]`, of extent `extents[i % axes.len()]`:
/// a `ScatterND` coordinate row runs over the first `k` axes in turn, and every
/// `ScatterElements` index addresses the one scattered axis. ONNX allows
/// `[-extent, extent)`. The kernels already drop a write whose index is outside that,
/// so this does not guard memory; it turns a model that names an invalid index into an
/// error naming it, instead of a scatter that silently writes less than it was asked
/// to.
fn check_scatter_indices(
    op: &str,
    what: &str,
    value: &ConstantTensor,
    axes: &[usize],
    extents: &[usize],
) -> Result<()> {
    let values = value
        .to_i64_vec()
        .map_err(|reason| Error::UnreadableConstant {
            context: format!("Op::{op} {what}: indices operand"),
            reason,
        })?;
    for (i, &index) in values.iter().enumerate() {
        let slot = i % axes.len();
        let (axis, extent) = (axes[slot], extents[slot]);
        // An extent beyond i64 cannot occur for a tensor that fits in memory; treat it
        // as unbounded rather than wrapping.
        let bound = i64::try_from(extent).unwrap_or(i64::MAX);
        if !(-bound..bound).contains(&index) {
            return Err(anyhow::anyhow!(
                "Op::{op} {what}: index {index} (element {i} of `indices`) is out of \
                 range for axis {axis}, of extent {extent}; ONNX allows \
                 -{extent}..{extent}"
            ));
        }
    }
    Ok(())
}

/// Builds the executable for `ScatterND`.
///
/// ONNX's shapes: `indices` has a trailing axis of width `k`, `updates` has
/// `indices`' shape with that axis replaced by `data`'s last `rank - k` axes, and the
/// output has `data`'s shape. All three are checked, because a `k` or a slice length
/// read from the wrong place would scatter into wrong positions rather than fail.
///
/// `indices`, when the lowering has the operand's value, is checked for range here as
/// well; see [`check_scatter_indices`].
fn scatter_nd_executable(
    graph: &Graph,
    node: &GraphNode,
    indices_value: Option<&ConstantTensor>,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    if node.inputs.len() != 3 {
        return Err(anyhow::anyhow!(
            "Op::ScatterND {what}: it has {} operands, not 3 (data, indices, updates)",
            node.inputs.len()
        ));
    }
    let operand = |pos: usize, name: &str| -> Result<Vec<usize>> {
        concrete_shape(
            &graph.nodes[node.inputs[pos]].shape,
            "ScatterND",
            name,
            what,
        )
    };
    let data = operand(0, "data")?;
    let indices = operand(1, "indices")?;
    let updates = operand(2, "updates")?;
    let out = concrete_shape(&node.shape, "ScatterND", "output", what)?;

    if out != data {
        return Err(anyhow::anyhow!(
            "Op::ScatterND {what}: its output is {out:?} but `data` is {data:?}; the \
             output is a copy of `data` with positions overwritten"
        ));
    }
    let k = *indices.last().ok_or_else(|| {
        anyhow::anyhow!("Op::ScatterND {what}: `indices` has no axes, so it names no coordinate")
    })?;
    if k == 0 || k > data.len() {
        return Err(anyhow::anyhow!(
            "Op::ScatterND {what}: a coordinate of width {k} cannot address a \
             rank-{} tensor",
            data.len()
        ));
    }
    // updates = indices.shape[..-1] ++ data.shape[k..]
    let mut want_updates: Vec<usize> = indices[..indices.len() - 1].to_vec();
    want_updates.extend_from_slice(&data[k..]);
    if updates != want_updates {
        return Err(anyhow::anyhow!(
            "Op::ScatterND {what}: `updates` is {updates:?} but the shapes call for \
             {want_updates:?}"
        ));
    }
    if graph.nodes[node.inputs[1]].dtype != DtypeRepr::I64 {
        return Err(anyhow::anyhow!(
            "Op::ScatterND {what}: `indices` is {:?}; the kernel reads them as I64, \
             which is what ONNX gives",
            graph.nodes[node.inputs[1]].dtype
        ));
    }

    let strides: Vec<usize> = {
        let mut all = vec![1usize; data.len()];
        for ax in (0..data.len().saturating_sub(1)).rev() {
            all[ax] = all[ax + 1] * data[ax + 1];
        }
        all[..k].to_vec()
    };
    let extents = data[..k].to_vec();
    if let Some(value) = indices_value {
        let axes: Vec<usize> = (0..k).collect();
        check_scatter_indices("ScatterND", what, value, &axes, &extents)?;
    }
    let slice_len: usize = data[k..].iter().product();
    let n_data: usize = data.iter().product();
    let n_updates: usize = updates.iter().product();

    macro_rules! build {
        ($ty:ty) => {{
            let rop = ScatterNdRuntimeOp::<$ty>::new(
                1024, &strides, &extents, slice_len, n_data, n_updates,
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Op::ScatterND {what}: a coordinate of width {k} over a slice of \
                     {slice_len} does not describe a scatter"
                )
            })?;
            let kernel = rop.kernel();
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::ScatterND {what}: {other:?} is not a supported dtype"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Builds the executable for `Concat`.
///
/// Every operand has to agree with the output on every axis but the concatenated one,
/// and their extents along it have to add up to the output's. ONNX requires both, and
/// a mismatch here would be a copy into the wrong place rather than a visible error,
/// so they are checked and named.
fn concat_executable(
    graph: &Graph,
    node: &GraphNode,
    axis: i64,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    let out_shape = concrete_shape(&node.shape, "Concat", "output", what)?;
    let rank = out_shape.len();
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    let axis = usize::try_from(axis)
        .ok()
        .filter(|&a| a < rank)
        .ok_or_else(|| {
            anyhow::anyhow!("Op::Concat {what}: axis {axis} is outside a rank-{rank} output")
        })?;
    if node.inputs.is_empty() {
        return Err(anyhow::anyhow!("Op::Concat {what}: it has no operands"));
    }

    let mut along = 0usize;
    for (pos, &src) in node.inputs.iter().enumerate() {
        let shape = concrete_shape(
            &graph.nodes[src].shape,
            "Concat",
            &format!("operand {pos}"),
            what,
        )?;
        if shape.len() != rank {
            return Err(anyhow::anyhow!(
                "Op::Concat {what}: operand {pos} has rank {} but the output has rank \
                 {rank}",
                shape.len()
            ));
        }
        for (ax, (&have, &want)) in shape.iter().zip(out_shape.iter()).enumerate() {
            if ax != axis && have != want {
                return Err(anyhow::anyhow!(
                    "Op::Concat {what}: operand {pos} has {have} on axis {ax} but the \
                     output has {want}; only axis {axis} may differ"
                ));
            }
        }
        if graph.nodes[src].dtype != node.dtype {
            return Err(anyhow::anyhow!(
                "Op::Concat {what}: operand {pos} is {:?} but the output is {:?}",
                graph.nodes[src].dtype,
                node.dtype
            ));
        }
        along += shape[axis];
    }
    if along != out_shape[axis] {
        return Err(anyhow::anyhow!(
            "Op::Concat {what}: its operands add up to {along} along axis {axis}, but \
             the output has {}",
            out_shape[axis]
        ));
    }

    let n_inputs = node.inputs.len();
    macro_rules! build {
        ($ty:ty) => {{
            let rop = ConcatRuntimeOp::<$ty>::new(128, axis, n_inputs)
                .ok_or_else(|| anyhow::anyhow!("Op::Concat {what}: it has no operands"))?;
            let kernel = rop.kernel();
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        // As in the constant path: this kernel moves bytes without computing on them,
        // so a one-byte-per-element bool tensor is copied correctly by the u8
        // instantiation.
        DtypeRepr::Bool => build!(u8),
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I8 => build!(i8),
        DtypeRepr::I16 => build!(i16),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        DtypeRepr::U8 => build!(u8),
        DtypeRepr::U16 => build!(u16),
        DtypeRepr::U32 => build!(u32),
        DtypeRepr::U64 => build!(u64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::Concat {what}: {other:?} is not a supported dtype"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Which logical operator to build; see [`logical_executable`].
#[derive(Clone, Copy)]
enum LogicalOp {
    And,
    Or,
    Xor,
}

impl LogicalOp {
    fn name(self) -> &'static str {
        match self {
            Self::And => "And",
            Self::Or => "Or",
            Self::Xor => "Xor",
        }
    }
}

/// Builds the executable for `And`, `Or` or `Xor`.
///
/// Bool only, which is what ONNX defines these on. The dtype decides both how wide
/// the buffers are and how the kernel reads them, so an operand recorded as anything
/// else is refused rather than read a byte at a time out of a wider tensor: an export
/// without value_info leaves a logical node carrying its first operand's dtype, and
/// a comparison feeding one still does (its own kernels write 0 or 1 in the operands'
/// dtype), so this is a reachable mistake and not a hypothetical one.
///
/// The kernel is flat, as the other elementwise kernels are, so operands of different
/// shapes are refused too. ONNX would broadcast them.
fn logical_executable(
    graph: &Graph,
    node: &GraphNode,
    which: LogicalOp,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    let op = which.name();
    if node.dtype != DtypeRepr::Bool {
        return Err(anyhow::anyhow!(
            "Op::{op} {what}: its output is {:?} rather than Bool; a logical operator \
             takes and returns bools, and the dtype sets the buffer's width",
            node.dtype
        ));
    }
    if node.inputs.len() != 2 {
        return Err(anyhow::anyhow!(
            "Op::{op} {what}: it has {} operands, not 2",
            node.inputs.len()
        ));
    }
    let out_shape = concrete_shape(&node.shape, op, "output", what)?;
    for (pos, &src) in node.inputs.iter().enumerate() {
        let operand = &graph.nodes[src];
        if operand.dtype != DtypeRepr::Bool {
            return Err(anyhow::anyhow!(
                "Op::{op} {what}: operand {pos} is {:?} rather than Bool",
                operand.dtype
            ));
        }
        let shape = concrete_shape(&operand.shape, op, &format!("operand {pos}"), what)?;
        if shape != out_shape {
            return Err(anyhow::anyhow!(
                "Op::{op} {what}: operand {pos} has shape {shape:?} but the output is \
                 {out_shape:?}; this kernel reads both operands at the output's own \
                 offsets and does not broadcast"
            ));
        }
    }

    // A Bool buffer is one byte per element, so the u8 instantiation reads and writes
    // it at the right width. As in the constant path, `bool` is deliberately not a
    // `Num` dtype, and the aliasing is sound here because the kernel only ever reads
    // a byte as true-or-false and writes back 1 or 0.
    let (name, kernel_source, kernel_body, runtime_op): (
        String,
        String,
        String,
        Arc<dyn RuntimeOp>,
    ) = match which {
        LogicalOp::And => {
            let k = LogicalAndForward::new(1024);
            (
                k.name.to_string(),
                k.source.clone(),
                k.kernel_source.clone(),
                Arc::new(k),
            )
        }
        LogicalOp::Or => {
            let k = LogicalOrForward::new(1024);
            (
                k.name.to_string(),
                k.source.clone(),
                k.kernel_source.clone(),
                Arc::new(k),
            )
        }
        LogicalOp::Xor => {
            let k = LogicalXorForward::new(1024);
            (
                k.name.to_string(),
                k.source.clone(),
                k.kernel_source.clone(),
                Arc::new(k),
            )
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Builds the executable for `ConstantOfShape`: one value written across the output.
///
/// Separate from [`constant_executable`] because a fill's parameter slot holds one
/// element rather than the whole tensor. The outputs are often far too large to carry
/// as data, which is why the constant folder leaves them alone and this exists.
fn fill_executable(
    node: &GraphNode,
    dtype: DtypeRepr,
    fill: &[u8],
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    let shape = concrete_shape(&node.shape, "ConstantOfShape", "output", what)?;
    macro_rules! build {
        ($ty:ty) => {{
            let rop = FillRuntimeOp::<$ty>::new(1024, fill).ok_or_else(|| {
                anyhow::anyhow!(
                    "Op::ConstantOfShape {what}: its fill value is {} bytes, which is \
                     not one {dtype:?}",
                    fill.len()
                )
            })?;
            let kernel = rop.kernel();
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match dtype {
        // As in `constant_executable`: a Bool buffer is one byte per element, and this
        // kernel moves bytes without computing on them, so the u8 instantiation writes
        // a bool tensor correctly. The node keeps its Bool dtype.
        DtypeRepr::Bool => build!(u8),
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I8 => build!(i8),
        DtypeRepr::I16 => build!(i16),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        DtypeRepr::U8 => build!(u8),
        DtypeRepr::U16 => build!(u16),
        DtypeRepr::U32 => build!(u32),
        DtypeRepr::U64 => build!(u64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::ConstantOfShape {what}: {other:?} is not a supported dtype, so a \
                 tensor of it cannot be filled on the device"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: shape.into_iter().map(Some).collect(),
        dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Builds the executable for `Gather`, whose indices come from a materialised
/// constant.
fn gather_executable(
    node: &GraphNode,
    in_shape: &[usize],
    axis: i64,
    indices: &[i64],
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    macro_rules! build {
        ($ty:ty) => {{
            let rop = GatherRuntimeOp::<$ty>::new(1024, in_shape, axis, indices)
                .map_err(|e| anyhow::anyhow!("Op::Gather {what}: {e}"))?;
            let kernel = GatherForward::<$ty>::new(1024);
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        // One byte per element: a bool is moved exactly as a u8.
        DtypeRepr::U8 | DtypeRepr::Bool => build!(u8),
        other => {
            return Err(anyhow::anyhow!(
                "Op::Gather {what}: {other:?} is not supported; the kernel is generic \
                 over FloatBytes"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Builds the executable for a `Gather` whose indices are an activation (see
/// [`GatherDynRuntimeOp`]).
fn gather_dyn_executable(
    node: &GraphNode,
    in_shape: &[usize],
    axis: i64,
    n_idx: usize,
    idx_bytes: usize,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    macro_rules! build {
        ($ty:ty) => {{
            let rop = GatherDynRuntimeOp::<$ty>::new(1024, in_shape, axis, n_idx, idx_bytes)
                .map_err(|e| anyhow::anyhow!("Op::Gather {what}: {e}"))?;
            let kernel = GatherDynForward::<$ty>::new(1024);
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        // One byte per element: a bool is moved exactly as a u8.
        DtypeRepr::U8 | DtypeRepr::Bool => build!(u8),
        other => {
            return Err(anyhow::anyhow!(
                "Op::Gather {what}: {other:?} data is not supported; the kernel is generic \
                 over FloatBytes"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Builds the executable for a converting `Cast` (see [`crate::nn::tensor::cast`]).
fn cast_executable(
    node: &GraphNode,
    from: DtypeRepr,
    to: DtypeRepr,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    use crate::nn::tensor::cast::{
        CastBoolToF32Forward, CastBoolToWordsForward, CastF32ToBoolForward,
        CastF32ToWordsForward, CastKind, CastRuntimeOp, CastWordsToBoolForward,
        CastWordsToF32Forward, CastWordsToWordsForward,
    };
    let words = |d: DtypeRepr| match d {
        DtypeRepr::I32 => Some(1),
        DtypeRepr::I64 => Some(2),
        _ => None,
    };
    let refuse = || {
        anyhow::anyhow!(
            "Op::Cast {what}: {from:?} to {to:?} has no kernel; casts between F32, I32, I64 \
             and Bool are supported"
        )
    };
    let kind = match (from, to) {
        (_, DtypeRepr::F32) if words(from).is_some() => CastKind::WordsToF32 {
            in_words: words(from).unwrap(),
        },
        (DtypeRepr::Bool, DtypeRepr::F32) => CastKind::BoolToF32,
        (_, DtypeRepr::Bool) if words(from).is_some() => CastKind::WordsToBool {
            in_words: words(from).unwrap(),
        },
        (DtypeRepr::F32, DtypeRepr::Bool) => CastKind::F32ToBool,
        (DtypeRepr::F32, _) if words(to).is_some() => CastKind::F32ToWords {
            out_words: words(to).unwrap(),
        },
        (DtypeRepr::Bool, _) if words(to).is_some() => CastKind::BoolToWords {
            out_words: words(to).unwrap(),
        },
        _ if words(from).is_some() && words(to).is_some() => CastKind::WordsToWords {
            in_words: words(from).unwrap(),
            out_words: words(to).unwrap(),
        },
        _ => return Err(refuse()),
    };
    macro_rules! k {
        ($K:ident) => {{
            let k = $K::new(1024);
            (k.name.to_string(), k.source.clone(), k.kernel_source.clone())
        }};
    }
    let (name, kernel_source, kernel_body) = match kind {
        CastKind::WordsToF32 { .. } => k!(CastWordsToF32Forward),
        CastKind::BoolToF32 => k!(CastBoolToF32Forward),
        CastKind::WordsToBool { .. } => k!(CastWordsToBoolForward),
        CastKind::F32ToBool => k!(CastF32ToBoolForward),
        CastKind::F32ToWords { .. } => k!(CastF32ToWordsForward),
        CastKind::BoolToWords { .. } => k!(CastBoolToWordsForward),
        CastKind::WordsToWords { .. } => k!(CastWordsToWordsForward),
    };
    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op: Arc::new(CastRuntimeOp::new(1024, kind)),
    }))
}

/// Builds the executable for `Tile`, whose repeats come from a materialised constant.
fn tile_executable(
    node: &GraphNode,
    in_shape: &[usize],
    repeats: &[i64],
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    macro_rules! build {
        ($ty:ty) => {{
            let rop = TileRuntimeOp::<$ty>::new(1024, in_shape, repeats).ok_or_else(|| {
                anyhow::anyhow!(
                    "Op::Tile {what}: repeats {repeats:?} do not match input shape \
                     {in_shape:?} (ONNX requires one non-negative repeat per dimension, \
                     and this kernel indexes at most {MAX_RANK} of them)"
                )
            })?;
            let kernel = TileForward::<$ty>::new(1024);
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::Tile {what}: {other:?} is not supported; the kernel is generic over \
                 FloatBytes"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Builds the executable for a two-operand contraction.
fn einsum_executable(
    node: &GraphNode,
    plan: &EinsumPlan,
    equation: &str,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    macro_rules! build {
        ($ty:ty) => {{
            let rop = EinsumRuntimeOp::<$ty>::new(1024, plan).ok_or_else(|| {
                anyhow::anyhow!(
                    "Op::Einsum {what}: '{equation}' has rank {} , beyond the \
                     {EINSUM_MAX_RANK} this contraction indexes",
                    plan.out_shape.len()
                )
            })?;
            let kernel = EinsumForward::<$ty>::new(1024);
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::Einsum {what}: {other:?} is not a supported dtype; the kernel is \
                 generic over Num"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// A node's name for an error message, or its index when it has none.
fn node_label(graph: &Graph, node_index: usize) -> String {
    graph
        .names
        .get(&node_index)
        .map_or_else(|| format!("node {node_index}"), |n| format!("'{n}'"))
}

/// Whether `perm` is a permutation of `0..perm.len()`.
fn is_permutation(perm: &[usize]) -> bool {
    let mut seen = vec![false; perm.len()];
    for &a in perm {
        match seen.get_mut(a) {
            Some(slot) if !*slot => *slot = true,
            _ => return false,
        }
    }
    true
}

/// Extents for a shape that has to be fully known, or an error naming the op.
///
/// An empty shape is refused as well as a dynamic one. `Shape` cannot distinguish a
/// rank-0 scalar from a shape nobody worked out, since both are the empty vector
/// (teenygrad#129), and reading an unknown shape as rank 0 would produce a copy of one
/// element instead of an error. Refusing costs the genuine rank-0 case, which none of
/// these ops do anything useful with, and buys an error message for every node whose
/// shape inference did not reach it.
fn concrete_shape(shape: &Shape, op: &str, what: &str, whose: &str) -> Result<Vec<usize>> {
    if shape.is_empty() {
        return Err(anyhow::anyhow!(
            "Op::{op} {what}: {whose} shape is empty, which in this IR means either a \
             rank-0 scalar or a shape no pass worked out, and they cannot be told \
             apart. Run teeny_onnx::materialize first."
        ));
    }
    shape
        .iter()
        .copied()
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Op::{op} {what}: {whose} shape {shape:?} has a dynamic or unknown \
                 dimension, so the copy's extents are not known. Run \
                 teeny_onnx::materialize first."
            )
        })
}

/// Which affine mapping a strided copy expresses.
///
/// The three cases differ only in how the output's extents and the input strides are
/// derived, which is exactly what the mapping is, so they share one kernel and one
/// executable. See [`crate::nn::tensor::strided_copy`].
enum CopyMapping<'a> {
    /// Output axis `d` is input axis `perm[d]`.
    Transpose {
        /// The input's extents.
        in_shape: &'a [usize],
        /// The output permutation of input axes.
        perm: &'a [usize],
    },
    /// A normalised `Slice`.
    Slice {
        /// The input's extents.
        in_shape: &'a [usize],
        /// Per-dimension start, step and output extent.
        plan: &'a SlicePlan,
    },
    /// Broadcast: a growing unit axis takes stride 0.
    Expand {
        /// The input's extents.
        in_shape: &'a [usize],
        /// The output's extents.
        out_shape: &'a [usize],
    },
}

impl CopyMapping<'_> {
    /// The ranks this mapping spans. Both matter: the kernel indexes at most
    /// [`MAX_RANK`] dimensions of either side.
    fn ranks(&self) -> (usize, usize) {
        match self {
            CopyMapping::Transpose { in_shape, perm } => (in_shape.len(), perm.len()),
            CopyMapping::Slice { in_shape, plan } => (in_shape.len(), plan.out_shape.len()),
            CopyMapping::Expand {
                in_shape,
                out_shape,
            } => (in_shape.len(), out_shape.len()),
        }
    }

    /// The extents this mapping writes.
    fn out_shape(&self) -> Vec<usize> {
        match self {
            CopyMapping::Transpose { in_shape, perm } => {
                perm.iter().map(|&a| in_shape[a]).collect()
            }
            CopyMapping::Slice { plan, .. } => plan.out_shape.clone(),
            CopyMapping::Expand { out_shape, .. } => out_shape.to_vec(),
        }
    }
}

/// Builds the executable for the ops that move elements under an affine index map.
///
/// The rank check is not a formality. `StridedCopyRuntimeOp` left-pads its shapes to
/// [`MAX_RANK`] and *truncates* anything longer, which would drop the fastest-varying
/// dimensions and copy the wrong elements with no error. Refusing here turns that into
/// a lowering failure naming the op.
/// A reduction's axes, keepdims and empty-axes rule, or `None` for anything else.
///
/// Grouped because the reduction kernels treat them all the same way, and so does the
/// transpose that has to come first.
/// A softmax's axis, and whether it is the log form, or `None` for anything else.
///
/// `Op::Softmax` names it `dim` and `Op::LogSoftmax` names it `axis`; both count
/// negative from the end.
fn softmax_axis(op: &Op) -> Option<(i64, AxisOp)> {
    match op {
        Op::Softmax { dim } => Some((*dim, AxisOp::Softmax)),
        Op::LogSoftmax { axis } => Some((*axis, AxisOp::LogSoftmax)),
        Op::CumSum { axis, .. } => Some((*axis, AxisOp::CumSum)),
        Op::CumProd { axis, .. } => Some((*axis, AxisOp::CumProd)),
        _ => None,
    }
}

/// An op that works along one named axis, where the kernel only does the innermost one.
///
/// All four keep their input's shape, so moving the axis means a transpose on each
/// side rather than the single one a reduction needs.
#[derive(Clone, Copy, PartialEq)]
enum AxisOp {
    Softmax,
    LogSoftmax,
    CumSum,
    CumProd,
}

/// The elements per output the reduction kernels reduce: one block, with no loop over a
/// longer row, so the lowering refuses a reduction past it rather than truncate it.
const REDUCE_BLOCK_INNER: i32 = 1024;

fn reduce_parts(op: &Op) -> Option<(&[i64], bool, bool)> {
    match op {
        Op::ReduceSum {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceMean {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceMax {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceMin {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceProd {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceL1 {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceL2 {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceLogSum {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceLogSumExp {
            axes,
            keepdims,
            noop_with_empty_axes,
        }
        | Op::ReduceSumSquare {
            axes,
            keepdims,
            noop_with_empty_axes,
        } => Some((axes, *keepdims, *noop_with_empty_axes)),
        _ => None,
    }
}

fn strided_copy_executable(
    node: &GraphNode,
    op: &str,
    mapping: CopyMapping<'_>,
    what: &str,
) -> Result<Box<dyn ExecutableOp>> {
    let (in_rank, out_rank) = mapping.ranks();
    if in_rank > MAX_RANK || out_rank > MAX_RANK {
        return Err(anyhow::anyhow!(
            "Op::{op} {what}: rank {in_rank} -> {out_rank} exceeds the {MAX_RANK} \
             dimensions this kernel indexes"
        ));
    }

    // The mapping's extents and the node's recorded shape must agree, because each is
    // used for something different: the mapping decides which input element every
    // output element reads, while the node's shape decides how large the output buffer
    // is and how many elements the kernel is launched for. When they disagree the copy
    // still runs, covers a prefix of the mapping it was built from, and writes a
    // plausibly-shaped tensor of wrong values with nothing to indicate it.
    let mapped = mapping.out_shape();
    if let Some(recorded) = node.shape.iter().copied().collect::<Option<Vec<usize>>>()
        && recorded != mapped
    {
        return Err(anyhow::anyhow!(
            "Op::{op} {what}: the operands map onto {mapped:?} but the node's shape is \
             {recorded:?}, so the copy and the output buffer would describe different \
             tensors"
        ));
    }

    macro_rules! build {
        ($ty:ty) => {{
            let rop = match &mapping {
                CopyMapping::Transpose { in_shape, perm } => {
                    StridedCopyRuntimeOp::<$ty>::transpose(1024, in_shape, perm)
                }
                CopyMapping::Slice { in_shape, plan } => StridedCopyRuntimeOp::<$ty>::slice(
                    1024,
                    in_shape,
                    &plan.out_shape,
                    &plan.starts,
                    &plan.steps,
                ),
                CopyMapping::Expand {
                    in_shape,
                    out_shape,
                } => StridedCopyRuntimeOp::<$ty>::expand(1024, in_shape, out_shape),
            };
            let kernel = StridedCopyForward::<$ty>::new(1024);
            let name = kernel.name.to_string();
            let source = kernel.source.clone();
            let body = kernel.kernel_source.clone();
            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
            (name, source, body, erased)
        }};
    }

    // Integer dtypes matter here as much as float ones: a model's shape arithmetic is
    // I64, and broadcasting it is what most of an export's `Expand` nodes do.
    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
        DtypeRepr::F32 => build!(f32),
        DtypeRepr::F64 => build!(f64),
        DtypeRepr::I8 => build!(i8),
        DtypeRepr::I16 => build!(i16),
        DtypeRepr::I32 => build!(i32),
        DtypeRepr::I64 => build!(i64),
        DtypeRepr::U8 | DtypeRepr::Bool => build!(u8),
        DtypeRepr::U16 => build!(u16),
        DtypeRepr::U32 => build!(u32),
        DtypeRepr::U64 => build!(u64),
        other => {
            return Err(anyhow::anyhow!(
                "Op::{op} {what}: {other:?} is not a supported dtype; the kernel is \
                 generic over Num"
            ));
        }
    };

    Ok(Box::new(KernelExecutable {
        entry_point: format!("{name}_entry_point"),
        name,
        kernel_source,
        kernel_body,
        pointwise_fuse_block_size: None,
        shape: node.shape.clone(),
        dtype: node.dtype,
        #[cfg(feature = "training")]
        backward_kernel_source: String::new(),
        #[cfg(feature = "training")]
        backward_entry_point: String::new(),
        runtime_op,
    }))
}

/// Decides, for each node, whether it is a constant that lowering can leave out of
/// the DAG entirely.
///
/// Public so a caller can see what lowering will need before attempting it: the
/// `false` entries that are still `Op::Constant` nodes are the ones that require a
/// device buffer.
///
/// A constant qualifies when every one of its consumers takes it in a shape-operand
/// position, so nothing ever reads it as a buffer. A constant with no consumers at
/// all qualifies too, unless it is the graph's result, which must stay to be read
/// back.
pub fn elidable_constants(graph: &Graph) -> Vec<bool> {
    let n = graph.nodes.len();
    let mut data_use = vec![false; n];
    let mut used = vec![false; n];

    for node in &graph.nodes {
        let shape_positions = node.op.shape_operand_positions();
        for (position, &input) in node.inputs.iter().enumerate() {
            used[input] = true;
            if !shape_positions.contains(&position) {
                data_use[input] = true;
            }
        }
    }

    // The graph's result is the one node nothing consumes that we must still keep.
    let result = (0..n).rev().find(|&i| !used[i]);

    (0..n)
        .map(|i| {
            // `Omitted` is always elided: it stands for an operand the model did not
            // supply, so there is nothing to put on a device. It is in the graph only
            // to hold its position, which `shape_operand_positions` and the ops that
            // read operands by index both depend on.
            matches!(graph.nodes[i].op, Op::Omitted)
                || matches!(graph.nodes[i].op, Op::Constant { .. })
                    && !data_use[i]
                    && Some(i) != result
        })
        .collect()
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
    // Where the backend cannot load a partial K tile, matmul_forward reads its last
    // one back from inside the operands, which needs a whole tile to fit, so the
    // tile is never wider than the contracted axis. Rounded down to a power of two,
    // since that is what the `dot` takes, and never below 1. Where a partial tile
    // does load, 8 stays the floor: a narrower tile makes a worse `dot`, and on
    // CUDA one below 8 gives a wrong answer outright.
    let k_floor_pow2 = 1usize << (usize::BITS - 1 - k.max(1).leading_zeros());
    let block_k = if crate::math::gemm::PARTIAL_K_TILE_IS_PADDED {
        block_k
    } else {
        block_k.min(k_floor_pow2 as i32)
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
        let (dag, map) = lowering.lower_with_mapping(&graph, LoweringMode::Inference)?;
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

    /// Like `lower` but also returns the graph-node-index → DAG-node-index mapping.
    /// Useful for middleware lowerings that need to patch specific DAG nodes after
    /// the base lowering runs.
    pub fn lower_with_mapping(
        &self,
        graph: &Graph,
        mode: LoweringMode,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)> {
        let _ = mode; // used by #[cfg(feature = "training")] branch below
        let optimized;
        let graph = match &self.optimizer {
            Some(opt) => {
                optimized = opt.optimize(graph)?;
                &optimized
            }
            None => graph,
        };
        let node_indexes = graph.topological_sort();
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        // Maps graph node index → DAG node index. Elided constants have no DAG node,
        // so their entry is never read; see `elided` below.
        let mut graph_to_dag = vec![0usize; graph.nodes.len()];
        // Constants consumed only as shape metadata, which lowering reads instead of
        // allocating. They get no DAG node and their edges are dropped.
        let elided = elidable_constants(graph);

        for node_index in node_indexes {
            let node = &graph.nodes[node_index];

            if elided[node_index] {
                continue;
            }

            // The reduction kernels walk a flattened `[outer, inner]` view and infer
            // the split from the output shape, so they can only reduce the innermost
            // axes. When the axes are somewhere else, a transpose goes in front to put
            // them there and the reduction reads that instead of the original input.
            // Nothing is needed afterwards: the axes that stay keep their relative
            // order, so the reduced elements come out in the order the recorded output
            // shape already describes, whether the reduced axes are kept as 1s or
            // dropped. Without this the reduction sums adjacent elements instead of
            // strided ones and returns a plausibly-shaped wrong answer.
            let mut reduce_input_override: Option<(usize, usize)> = None;
            // A reduction may carry a second operand, the axes, which
            // `Op::shape_operand_positions` marks as metadata: its value is on the op by
            // now and the operand is elided, so only the first input is data.
            if let Some((axes, _, noop_with_empty_axes)) = reduce_parts(&node.op)
                && let Some(&data) = node.inputs.first()
                && !elided[data]
            {
                if axes.is_empty() && node.inputs.len() > 1 {
                    let what = node_label(graph, node_index);
                    return Err(anyhow::anyhow!(
                        "Op::Reduce {what}: the axes operand was not materialised, and \
                         an empty list here would mean every axis rather than the one \
                         the file asked for. Run teeny_onnx::materialize first."
                    ));
                }
                let what = node_label(graph, node_index);
                // The axes depend only on the rank, so a dynamic extent (a batch axis,
                // say) is fine until a transpose actually has to copy the tensor.
                let data_shape = &graph.nodes[data].shape;
                let rank = data_shape.len();
                let Some(reduced) = reduce_axes(axes, rank, noop_with_empty_axes) else {
                    return Err(anyhow::anyhow!(
                        "Op::Reduce {what}: axes {axes:?} are not all within the \
                         {rank} axes of {data_shape:?}"
                    ));
                };
                // The kernels reduce one block of REDUCE_BLOCK_INNER elements per output,
                // with no loop over a longer row, so past that they silently drop the
                // rest. The reduced extents are the ones the kernel walks, so they must
                // be known even where the axes that stay may be dynamic.
                let inner: Option<usize> = reduced.iter().map(|&axis| data_shape[axis]).product();
                match inner {
                    Some(inner) if inner <= REDUCE_BLOCK_INNER as usize => {}
                    Some(inner) => {
                        return Err(anyhow::anyhow!(
                            "Op::Reduce {what}: reducing {inner} elements per output over \
                             axes {axes:?} of {data_shape:?}, but the kernel reduces at most \
                             {REDUCE_BLOCK_INNER} and would silently drop the rest. See \
                             teenygrad#151."
                        ));
                    }
                    None => {
                        return Err(anyhow::anyhow!(
                            "Op::Reduce {what}: a reduced axis of {data_shape:?} (axes \
                             {axes:?}) has an unknown extent, so it cannot be checked \
                             against the kernel's limit of {REDUCE_BLOCK_INNER} elements \
                             per output. Give the input a static shape, or run \
                             teeny_onnx::materialize first."
                        ));
                    }
                }
                if let Some(perm) = reduce_transpose_perm(&reduced, rank) {
                    let in_shape = concrete_shape(data_shape, "Reduce", &what, "its input's")?;
                    let transposed: Vec<Option<usize>> =
                        perm.iter().map(|&a| Some(in_shape[a])).collect();
                    // A stand-in for the transpose's own output, which no graph node
                    // describes: `strided_copy_executable` checks the shape it is given
                    // against the mapping, and the reduction node's shape is a
                    // different tensor's.
                    let staging = GraphNode {
                        op: Op::Transpose { perm: perm.clone() },
                        inputs: vec![data],
                        dtype: graph.nodes[data].dtype,
                        shape: transposed,
                    };
                    let executable = strided_copy_executable(
                        &staging,
                        "Reduce",
                        CopyMapping::Transpose {
                            in_shape: &in_shape,
                            perm: &perm,
                        },
                        &what,
                    )?;
                    let staging_dag_idx = dag.add_node(executable);
                    dag.add_edge(graph_to_dag[data], staging_dag_idx);
                    reduce_input_override = Some((data, staging_dag_idx));
                }
            }

            // The softmax kernels normalize a tensor's innermost axis, one CTA per row,
            // so an axis anywhere else has to be moved there first. Unlike a reduction
            // the output keeps the input's shape, so this needs a transpose on both
            // sides: in, normalize, and back. Without it the kernel normalized the last
            // axis whatever the op asked for.
            // A cumulative op carries its axis as a second operand, which
            // `Op::shape_operand_positions` marks as metadata, so only the first input
            // is data. Matching on exactly one input skipped those entirely.
            if let Some((dim, axis_op)) = softmax_axis(&node.op)
                && let Some(&data) = node.inputs.first()
                && !elided[data]
            {
                let what = node_label(graph, node_index);
                // As for a reduction: only a transpose needs every extent, so a dynamic
                // batch axis in front of an innermost softmax still lowers.
                let data_shape = &graph.nodes[data].shape;
                let rank = data_shape.len();
                let Some(axes) = reduce_axes(&[dim], rank, false) else {
                    return Err(anyhow::anyhow!(
                        "Op::Softmax {what}: axis {dim} is outside the {rank} axes of \
                         {data_shape:?}"
                    ));
                };
                if let Some(perm) = reduce_transpose_perm(&axes, rank) {
                    let in_shape = concrete_shape(data_shape, "Softmax", &what, "its input's")?;
                    let axis = axes[0];
                    let permuted: Vec<usize> = perm.iter().map(|&a| in_shape[a]).collect();
                    let transposed: Shape = permuted.iter().map(|&d| Some(d)).collect();

                    // In: the axis to normalize becomes the innermost one.
                    let staging = GraphNode {
                        op: Op::Transpose { perm: perm.clone() },
                        inputs: vec![data],
                        dtype: graph.nodes[data].dtype,
                        shape: transposed.clone(),
                    };
                    let executable = strided_copy_executable(
                        &staging,
                        "Softmax",
                        CopyMapping::Transpose {
                            in_shape: &in_shape,
                            perm: &perm,
                        },
                        &what,
                    )?;
                    let into_dag_idx = dag.add_node(executable);
                    dag.add_edge(graph_to_dag[data], into_dag_idx);

                    // The normalization itself, over the transposed tensor, whose last
                    // axis is now the one the op named.
                    let block_size = in_shape[axis].next_power_of_two() as i32;
                    let middle = GraphNode {
                        op: node.op.clone(),
                        inputs: vec![],
                        dtype: node.dtype,
                        shape: transposed,
                    };
                    let executable: Box<dyn ExecutableOp> = if axis_op == AxisOp::LogSoftmax {
                        let fwd = LogSoftmaxForward::new(block_size);
                        let nm = fwd.name.to_string();
                        let fwd_src = fwd.source.clone();
                        #[cfg(feature = "training")]
                        let bwd_src = LogSoftmaxBackward::new(block_size).source.clone();
                        let rop: Arc<dyn RuntimeOp> = Arc::new(fwd);
                        Box::new(KernelExecutable {
                            entry_point: format!("{nm}_entry_point"),
                            name: nm,
                            kernel_source: fwd_src,
                            kernel_body: String::new(),
                            pointwise_fuse_block_size: None,
                            shape: middle.shape.clone(),
                            dtype: middle.dtype,
                            #[cfg(feature = "training")]
                            backward_kernel_source: bwd_src,
                            #[cfg(feature = "training")]
                            backward_entry_point: String::new(),
                            runtime_op: rop,
                        })
                    } else if axis_op == AxisOp::Softmax {
                        make_float_kernel!(SoftmaxForward(block_size), middle)
                    } else if axis_op == AxisOp::CumSum {
                        make_num_kernel!(CumSumForward(1024), middle)
                    } else {
                        make_num_kernel!(CumProdForward(1024), middle)
                    };
                    let middle_dag_idx = dag.add_node(executable);
                    dag.add_edge(into_dag_idx, middle_dag_idx);

                    // Back: the inverse permutation restores the recorded shape, which
                    // a softmax leaves as its input's.
                    let mut inverse = vec![0usize; rank];
                    for (new_axis, &old_axis) in perm.iter().enumerate() {
                        inverse[old_axis] = new_axis;
                    }
                    let executable = strided_copy_executable(
                        node,
                        "Softmax",
                        CopyMapping::Transpose {
                            in_shape: &permuted,
                            perm: &inverse,
                        },
                        &what,
                    )?;
                    let out_dag_idx = dag.add_node(executable);
                    dag.add_edge(middle_dag_idx, out_dag_idx);
                    graph_to_dag[node_index] = out_dag_idx;
                    continue;
                }
            }

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
                // A contiguous tensor flattens to the same elements in the same order, so this
                // is the element copy Reshape and Identity are (the 2-D tiled FlattenForward
                // miscopies its u8 instantiation, which bool masks take).
                Op::Flatten { .. } => make_num_kernel!(ElemwiseCopyForward(1024), node),

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
                Op::InstanceNorm1d { eps, .. }
                | Op::InstanceNorm2d { eps, .. }
                | Op::InstanceNorm3d { eps, .. } => {
                    // The extents and the epsilon were being dropped here, and the
                    // kernel they reached had no runtime dispatch: one program per
                    // instance, which is what `grid` has to say.
                    let what = node_label(graph, node_index);
                    let shape = concrete_shape(&node.shape, "InstanceNorm", "output", &what)?;
                    if shape.len() < 2 {
                        return Err(anyhow::anyhow!(
                            "Op::InstanceNorm {what}: {shape:?} has no channel axis to \
                             normalise within"
                        ));
                    }
                    if node.inputs.len() != 3 {
                        return Err(anyhow::anyhow!(
                            "Op::InstanceNorm {what}: it has {} operands, not 3 (input, \
                             scale, shift)",
                            node.inputs.len()
                        ));
                    }
                    let (n_batch, channels) = (shape[0], shape[1]);
                    let spatial: usize = shape[2..].iter().product();
                    macro_rules! build_instance {
                        ($ty:ty) => {{
                            let rop = InstanceNormRuntimeOp::<$ty>::new(
                                256,
                                n_batch,
                                channels,
                                spatial,
                                *eps as f32,
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::InstanceNorm {what}: {shape:?} leaves an \
                                     instance with nothing to normalise"
                                )
                            })?;
                            let kernel = rop.kernel();
                            let name = kernel.name.to_string();
                            let source = kernel.source.clone();
                            let body = kernel.kernel_source.clone();
                            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
                            (name, source, body, erased)
                        }};
                    }
                    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
                        DtypeRepr::F32 => build_instance!(f32),
                        DtypeRepr::F64 => build_instance!(f64),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Op::InstanceNorm {what}: {other:?} is not a float dtype"
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{name}_entry_point"),
                        name,
                        kernel_source,
                        kernel_body,
                        pointwise_fuse_block_size: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op,
                    })
                }

                // --- Convolution ---
                Op::Conv1d {
                    kernel_l,
                    stride,
                    padding,
                    ..
                } => {
                    make_num_kernel!(
                        Conv1dForward(*kernel_l as i32, *stride as i32, *padding as i32, 32),
                        node
                    )
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
                    make_num_kernel!(
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
                    )
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
                    make_num_kernel!(
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
                    )
                }

                // --- Pooling ---
                Op::AvgPool1d { kernel_l, stride } => {
                    make_num_kernel!(Avgpool1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::AvgPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    count_include_pad,
                } => {
                    make_num_kernel!(
                        Avgpool2dForward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            *pad_h as i32,
                            *pad_w as i32,
                            i32::from(*count_include_pad),
                            16
                        ),
                        node
                    )
                }
                Op::AvgPool3d {
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    stride_d,
                    stride_h,
                    stride_w,
                } => {
                    make_num_kernel!(
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
                    )
                }
                Op::MaxPool1d { kernel_l, stride } => {
                    make_num_kernel!(Maxpool1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::MaxPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                } => {
                    make_num_kernel!(
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
                    )
                }
                Op::MaxPool3d {
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    stride_d,
                    stride_h,
                    stride_w,
                } => {
                    make_num_kernel!(
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
                    )
                }
                Op::LpPool1d {
                    kernel_l, stride, ..
                } => {
                    make_float_kernel!(Lppool1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::LpPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    ..
                } => {
                    make_float_kernel!(
                        Lppool2dForward(
                            *kernel_h as i32,
                            *kernel_w as i32,
                            *stride_h as i32,
                            *stride_w as i32,
                            16
                        ),
                        node
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
                } => {
                    make_float_kernel!(
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
                    )
                }

                // --- Padding ---
                Op::ConstantPad1d {
                    pad_left,
                    pad_right,
                    ..
                } => {
                    make_num_kernel!(
                        ConstantPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    )
                }
                Op::ConstantPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                    ..
                } => {
                    make_num_kernel!(
                        ConstantPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    )
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
                    make_num_kernel!(
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
                    )
                }
                Op::ReflectionPad1d {
                    pad_left,
                    pad_right,
                } => {
                    make_num_kernel!(
                        ReflectionPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    )
                }
                Op::ReflectionPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                } => {
                    make_num_kernel!(
                        ReflectionPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    )
                }
                Op::ReflectionPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                } => {
                    make_num_kernel!(
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
                    )
                }
                Op::ReplicationPad1d {
                    pad_left,
                    pad_right,
                } => {
                    make_num_kernel!(
                        ReplicationPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    )
                }
                Op::ReplicationPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                } => {
                    make_num_kernel!(
                        ReplicationPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    )
                }
                Op::ReplicationPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                } => {
                    make_num_kernel!(
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
                    )
                }
                Op::CircularPad1d {
                    pad_left,
                    pad_right,
                } => {
                    make_num_kernel!(
                        CircularPad1dForward(*pad_left as i32, *pad_right as i32, 32),
                        node
                    )
                }
                Op::CircularPad2d {
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                } => {
                    make_num_kernel!(
                        CircularPad2dForward(
                            *pad_t as i32,
                            *pad_b as i32,
                            *pad_l as i32,
                            *pad_r as i32,
                            16
                        ),
                        node
                    )
                }
                Op::CircularPad3d {
                    pad_d1,
                    pad_d2,
                    pad_h1,
                    pad_h2,
                    pad_w1,
                    pad_w2,
                } => {
                    make_num_kernel!(
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
                    )
                }

                // --- Activation (D: Num) ---
                Op::Relu => make_num_kernel!(ReluForward(1024), ReluBackward(1024), node),

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
                Op::LeakyRelu { negative_slope } => {
                    // The slope was being dropped here, and the kernel it reached had
                    // no runtime dispatch at all: it compiled and then panicked when
                    // the executor asked for its argument count.
                    let what = node_label(graph, node_index);
                    macro_rules! build_leaky {
                        ($ty:ty) => {{
                            let rop = LeakyReluRuntimeOp::<$ty>::new(1024, *negative_slope as f32);
                            let kernel = rop.kernel();
                            let name = kernel.name.to_string();
                            let source = kernel.source.clone();
                            let body = kernel.kernel_source.clone();
                            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
                            (name, source, body, erased)
                        }};
                    }
                    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
                        DtypeRepr::F32 => build_leaky!(f32),
                        DtypeRepr::F64 => build_leaky!(f64),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Op::LeakyRelu {what}: {other:?} is not a float dtype"
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{name}_entry_point"),
                        name,
                        kernel_source,
                        kernel_body,
                        pointwise_fuse_block_size: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op,
                    })
                }
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
                Op::Silu => exec_from(
                    node.shape.clone(),
                    node.dtype,
                    SiluForwardDispatch::dispatch(node.dtype, 1024)?,
                ),
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
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: bwd_src,
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op: rop,
                    })
                }

                Op::Add if needs_broadcast(graph, node) => broadcast_binary_executable(
                    graph,
                    node,
                    BroadcastOp::Add,
                    &node_label(graph, node_index),
                )?,
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
                Op::Mul if needs_broadcast(graph, node) => broadcast_binary_executable(
                    graph,
                    node,
                    BroadcastOp::Mul,
                    &node_label(graph, node_index),
                )?,
                Op::Mul => {
                    make_num_kernel!(ElemwiseMulForward(1024), ElemwiseMulBackward(1024), node)
                }
                Op::Sub if needs_broadcast(graph, node) => broadcast_binary_executable(
                    graph,
                    node,
                    BroadcastOp::Sub,
                    &node_label(graph, node_index),
                )?,
                Op::Sub => {
                    make_num_kernel!(ElemwiseSubForward(1024), ElemwiseSubBackward(1024), node)
                }
                Op::Div if needs_broadcast(graph, node) => broadcast_binary_executable(
                    graph,
                    node,
                    BroadcastOp::Div,
                    &node_label(graph, node_index),
                )?,
                Op::Div => {
                    make_float_kernel!(ElemwiseDivForward(1024), ElemwiseDivBackward(1024), node)
                }
                Op::Pow
                | Op::Mod { .. }
                | Op::ElemMin
                | Op::ElemMax
                | Op::Equal
                | Op::Greater
                | Op::GreaterOrEqual
                | Op::Less
                | Op::LessOrEqual
                    if needs_broadcast(graph, node) =>
                {
                    // Only Add, Sub, Mul and Div have a broadcasting kernel. The flat
                    // kernels read every operand at the output's offsets, so a smaller
                    // operand would be read past its end: refuse instead.
                    let what = node_label(graph, node_index);
                    let shapes: Vec<_> = node
                        .inputs
                        .iter()
                        .map(|&i| graph.nodes[i].shape.clone())
                        .collect();
                    return Err(anyhow::anyhow!(
                        "Op::{:?} {what}: operands {shapes:?} differ from the output {:?}, \
                         and only Add, Sub, Mul and Div have a broadcasting kernel",
                        node.op,
                        node.shape
                    ));
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
                Op::Clip { min, max } => {
                    let what = node_label(graph, node_index);
                    // Both bounds were hardcoded to infinity here, so every Clip in
                    // every model was a pass-through: the op ran, cost a kernel launch
                    // and clamped nothing. Nothing in the graph showed it, because the
                    // output has the right shape and dtype and most values in a
                    // trained network are inside their bounds anyway.
                    let bound = |pos: usize, attr: Option<f64>, fallback: f32| -> Result<f32> {
                        let value: Option<f64> = match node.inputs.get(pos) {
                            // No operand, or one the model omitted: the opset-6
                            // attribute form, or no bound on this side.
                            None => attr,
                            Some(&id) if matches!(graph.nodes[id].op, Op::Omitted) => attr,
                            Some(&id) => match self.constants.get(&id).map(|t| t.to_f64_vec()) {
                                // An empty tensor is how some exporters write "no
                                // bound" without omitting the operand.
                                Some(Ok(v)) => v.first().copied(),
                                Some(Err(reason)) => {
                                    return Err(Error::UnreadableConstant {
                                        context: format!("Op::Clip {what}: operand {pos}"),
                                        reason,
                                    }
                                    .into());
                                }
                                None => {
                                    return Err(anyhow::anyhow!(
                                        "Op::Clip {what}: operand {pos} has no value, so \
                                         the bound is not known. The kernel takes its \
                                         bounds as scalar arguments, so a computed bound \
                                         cannot be used; materialise it and pass it via \
                                         TritonLowering::with_constants."
                                    ));
                                }
                            },
                        };
                        let Some(value) = value else {
                            return Ok(fallback);
                        };
                        // The kernel takes its bounds as f32 whatever the tensor's
                        // dtype. Narrowing an f64 bound loses a little precision in the
                        // threshold, which is tolerable; narrowing one that overflows
                        // f32 would turn a real bound into no bound at all, which is
                        // not, so it is refused rather than rounded to infinity.
                        if value.is_finite() && !(value as f32).is_finite() {
                            return Err(anyhow::anyhow!(
                                "Op::Clip {what}: bound {value} at operand {pos} is \
                                 outside f32 range, and the kernel takes its bounds as \
                                 f32, so clamping would silently not happen"
                            ));
                        }
                        Ok(value as f32)
                    };
                    let min_val = bound(1, *min, f32::NEG_INFINITY)?;
                    let max_val = bound(2, *max, f32::INFINITY)?;
                    if min_val > max_val {
                        return Err(anyhow::anyhow!(
                            "Op::Clip {what}: min {min_val} is above max {max_val}, so \
                             every element would be clamped to {max_val}"
                        ));
                    }

                    let (name, ks, bwd_ks, rop): (String, String, String, Arc<dyn RuntimeOp>) =
                        match node.dtype {
                            DtypeRepr::F32 => {
                                let r = ClipRuntimeOp::<f32>::new(1024, min_val, max_val);
                                (
                                    r.kernel_name().to_string(),
                                    r.forward_source().to_string(),
                                    r.backward_source().to_string(),
                                    Arc::new(r),
                                )
                            }
                            DtypeRepr::F64 => {
                                let r = ClipRuntimeOp::<f64>::new(1024, min_val, max_val);
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
                Op::ReduceSum { .. } => {
                    make_num_kernel!(ReduceSumForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceMean { .. } => {
                    make_float_kernel!(ReduceMeanForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceMax { .. } => {
                    make_num_kernel!(ReduceMaxForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceMin { .. } => {
                    make_num_kernel!(ReduceMinForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceProd { .. } => {
                    make_float_kernel!(ReduceProdForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceL1 { .. } => make_num_kernel!(ReduceL1Forward(REDUCE_BLOCK_INNER), node),
                Op::ReduceL2 { .. } => {
                    make_float_kernel!(ReduceL2Forward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceLogSum { .. } => {
                    make_float_kernel!(ReduceLogSumForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceLogSumExp { .. } => {
                    make_float_kernel!(ReduceLogSumExpForward(REDUCE_BLOCK_INNER), node)
                }
                Op::ReduceSumSquare { .. } => {
                    make_num_kernel!(ReduceSumSquareForward(REDUCE_BLOCK_INNER), node)
                }
                Op::CumSum {
                    exclusive, reverse, ..
                }
                | Op::CumProd {
                    exclusive, reverse, ..
                } if *exclusive || *reverse => {
                    // The kernels accumulate inclusively and forwards, and nothing in
                    // them reads these, so either flag would be ignored rather than
                    // obeyed.
                    let what = node_label(graph, node_index);
                    return Err(anyhow::anyhow!(
                        "Op::CumSum/CumProd {what}: exclusive={exclusive} \
                         reverse={reverse} is not implemented; the kernel accumulates \
                         inclusively and forwards"
                    ));
                }
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
                    let what = node_label(graph, node_index);
                    let operand = |pos: usize, side: &str| -> Result<Vec<usize>> {
                        let &i = node.inputs.get(pos).ok_or_else(|| {
                            anyhow::anyhow!("Op::MatMul {what}: no {side} operand")
                        })?;
                        concrete_shape(&graph.nodes[i].shape, "MatMul", side, &what)
                    };
                    let plan =
                        matmul_plan(&operand(0, "left operand")?, &operand(1, "right operand")?)
                            .map_err(|e| anyhow::anyhow!("Op::MatMul {what}: {e}"))?;

                    let (block_m, block_n, block_k) =
                        pick_gemm_tile_sizes(Some(plan.m), plan.n, plan.k);

                    let (name, ks, rop): (String, String, Arc<dyn RuntimeOp>) = match node.dtype {
                        DtypeRepr::F32 => {
                            let r = MatMulRuntimeOp::<f32>::new(
                                block_m,
                                block_n,
                                block_k,
                                plan.clone(),
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::MatMul {what}: a K tile of {block_k} does not fit \
                                     inside a contraction of {}",
                                    plan.k
                                )
                            })?;
                            (
                                r.kernel_name().to_string(),
                                r.forward_source().to_string(),
                                Arc::new(r),
                            )
                        }
                        DtypeRepr::F64 => {
                            let r = MatMulRuntimeOp::<f64>::new(
                                block_m,
                                block_n,
                                block_k,
                                plan.clone(),
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::MatMul {what}: a K tile of {block_k} does not fit \
                                     inside a contraction of {}",
                                    plan.k
                                )
                            })?;
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
                // Shape-only: element order is unchanged, so for contiguous buffers
                // this is an element-for-element copy. A zero-copy view would be
                // better, but the executable DAG has no node type that aliases its
                // input's buffer.
                Op::Reshape => {
                    make_num_kernel!(ElemwiseCopyForward(1024), ElemwiseCopyBackward(1024), node)
                }
                Op::Transpose { perm } => {
                    let what = node_label(graph, node_index);
                    let &[data] = node.inputs.as_slice() else {
                        return Err(anyhow::anyhow!(
                            "Op::Transpose {what}: expected one input, got {}",
                            node.inputs.len()
                        ));
                    };
                    let in_shape = concrete_shape(
                        &graph.nodes[data].shape,
                        "Transpose",
                        &what,
                        "its input's",
                    )?;
                    // ONNX defaults `perm` to the reversed axis order, and the importer
                    // records the attribute's absence as an empty vector, so an empty
                    // perm here means that default rather than an identity.
                    let reversed: Vec<usize> = (0..in_shape.len()).rev().collect();
                    let perm = if perm.is_empty() { &reversed } else { perm };
                    // Two different faults, reported differently. A perm of the wrong
                    // length usually means the recorded input shape is wrong rather
                    // than the attribute, since an exporter writes one perm entry per
                    // axis, so blaming the perm sends the reader to the wrong place.
                    if perm.len() != in_shape.len() {
                        return Err(anyhow::anyhow!(
                            "Op::Transpose {what}: perm {perm:?} has {} entries but the \
                             recorded input shape {in_shape:?} has {} axes. An exporter \
                             writes one entry per axis, so the recorded shape is the \
                             likely culprit; run teeny_onnx::materialize first.",
                            perm.len(),
                            in_shape.len()
                        ));
                    }
                    if !is_permutation(perm) {
                        return Err(anyhow::anyhow!(
                            "Op::Transpose {what}: perm {perm:?} is not a permutation of \
                             the {} axes of {in_shape:?}",
                            in_shape.len()
                        ));
                    }
                    strided_copy_executable(
                        node,
                        "Transpose",
                        CopyMapping::Transpose {
                            in_shape: &in_shape,
                            perm,
                        },
                        &what,
                    )?
                }
                // Shape-only, as Op::Reshape above.
                Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
                    make_num_kernel!(ElemwiseCopyForward(1024), ElemwiseCopyBackward(1024), node)
                }
                Op::Concat { axis } => {
                    let what = node_label(graph, node_index);
                    concat_executable(graph, node, *axis, &what)?
                }
                Op::Split { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Split — implement as a multi-output slice kernel"
                    ));
                }
                Op::Slice => {
                    let what = node_label(graph, node_index);
                    let Some(&data) = node.inputs.first() else {
                        return Err(anyhow::anyhow!("Op::Slice {what}: has no input"));
                    };
                    let in_shape =
                        concrete_shape(&graph.nodes[data].shape, "Slice", &what, "its input's")?;
                    // starts and ends are required operands; axes and steps are not.
                    let operand = |pos: usize, required: bool| -> Result<Option<Vec<i64>>> {
                        let Some(&id) = node.inputs.get(pos) else {
                            if required {
                                return Err(anyhow::anyhow!(
                                    "Op::Slice {what}: operand {pos} is missing, and \
                                     starts and ends are required"
                                ));
                            }
                            return Ok(None);
                        };
                        match self.constants.get(&id).map(|t| t.to_i64_vec()) {
                            Some(Ok(v)) => Ok(Some(v)),
                            Some(Err(e)) => Err(Error::UnreadableConstant {
                                context: format!("Op::Slice {what}: operand {pos}"),
                                reason: e,
                            }
                            .into()),
                            None => Err(anyhow::anyhow!(
                                "Op::Slice {what}: operand {pos} has no integer value; \
                                 materialise it and pass it via \
                                 TritonLowering::with_constants"
                            )),
                        }
                    };
                    let starts = operand(1, true)?.unwrap_or_default();
                    let ends = operand(2, true)?.unwrap_or_default();
                    let axes = operand(3, false)?;
                    let steps = operand(4, false)?;
                    // Shared with shape inference, so the extents this copy writes and
                    // the ones the graph records cannot disagree.
                    let plan =
                        slice_plan(&in_shape, &starts, &ends, axes.as_deref(), steps.as_deref())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::Slice {what}: starts {starts:?}, ends {ends:?}, axes \
                             {axes:?} and steps {steps:?} do not describe a slice of \
                             {in_shape:?}"
                                )
                            })?;
                    strided_copy_executable(
                        node,
                        "Slice",
                        CopyMapping::Slice {
                            in_shape: &in_shape,
                            plan: &plan,
                        },
                        &what,
                    )?
                }
                // `Gather(data, indices, axis)`. An index tensor is integer-typed and a
                // node's buffers are typed from its own dtype, so there is nowhere
                // correctly-typed for one to land as an activation. The kernel reads its
                // indices from a parameter slot instead, so the operand is elided like
                // any other that never needs a buffer, and the value comes from the
                // constants supplied here. That covers nearly every Gather in practice.
                Op::Gather { axis } => {
                    let what = graph
                        .names
                        .get(&node_index)
                        .map_or_else(|| format!("node {node_index}"), |n| format!("'{n}'"));
                    let &[data, idx_node] = node.inputs.as_slice() else {
                        return Err(anyhow::anyhow!(
                            "Op::Gather {what}: expected (data, indices), got {} inputs",
                            node.inputs.len()
                        ));
                    };
                    let in_shape = graph.nodes[data]
                        .shape
                        .iter()
                        .copied()
                        .collect::<Option<Vec<usize>>>()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Op::Gather {what}: input shape {:?} has a dynamic \
                                 dimension, so the gathered extent is not known",
                                graph.nodes[data].shape
                            )
                        })?;
                    match self.constants.get(&idx_node) {
                        None => {
                            // Indices known only at run time (token ids into an embedding):
                            // read from the index tensor's own buffer.
                            let idx = &graph.nodes[idx_node];
                            let idx_bytes = match idx.dtype {
                                DtypeRepr::I64 => 8,
                                DtypeRepr::I32 => 4,
                                other => {
                                    return Err(anyhow::anyhow!(
                                        "Op::Gather {what}: runtime indices of {other:?}; only \
                                     I32 and I64 are supported"
                                    ));
                                }
                            };
                            let n_idx: usize =
                                concrete_shape(&idx.shape, "Gather", &what, "its indices'")?
                                    .iter()
                                    .product();
                            gather_dyn_executable(node, &in_shape, *axis, n_idx, idx_bytes, &what)?
                        }
                        Some(constant) => {
                            let indices = constant.to_i64_vec().map_err(|reason| {
                                Error::UnreadableConstant {
                                    context: format!("Op::Gather {what}: indices operand"),
                                    reason,
                                }
                            })?;
                            gather_executable(node, &in_shape, *axis, &indices, &what)?
                        }
                    }
                }
                Op::GatherElements { .. } | Op::GatherND { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::GatherElements/GatherND — per-element and N-D \
                         indexing; Op::Gather's slice-along-one-axis kernel does not \
                         cover either"
                    ));
                }
                Op::ScatterND => {
                    let what = node_label(graph, node_index);
                    let indices = node.inputs.get(1).and_then(|i| self.constants.get(i));
                    scatter_nd_executable(graph, node, indices, &what)?
                }
                Op::ScatterElements { axis } => {
                    let what = node_label(graph, node_index);
                    let indices = node.inputs.get(1).and_then(|i| self.constants.get(i));
                    scatter_elements_executable(graph, node, *axis, indices, &what)?
                }
                Op::Scatter { .. } | Op::TensorScatter => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Scatter/TensorScatter — the deprecated Scatter and \
                         the tensor-scatter variant; ScatterElements and ScatterND \
                         cover what the exports use"
                    ));
                }
                // `Tile(data, repeats)`. The repeats operand is shape metadata, so it
                // is elided rather than lowered; its value comes from the constants
                // supplied to this lowering. This is what materialising constants was
                // for: without the value there is no kernel to build.
                Op::Tile => {
                    let what = graph
                        .names
                        .get(&node_index)
                        .map_or_else(|| format!("node {node_index}"), |n| format!("'{n}'"));
                    let &[data, repeats_node] = node.inputs.as_slice() else {
                        return Err(anyhow::anyhow!(
                            "Op::Tile {what}: expected (data, repeats), got {} inputs",
                            node.inputs.len()
                        ));
                    };
                    let in_shape = graph.nodes[data]
                        .shape
                        .iter()
                        .copied()
                        .collect::<Option<Vec<usize>>>()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Op::Tile {what}: input shape {:?} has a dynamic \
                                 dimension, so the tiled extent is not known",
                                graph.nodes[data].shape
                            )
                        })?;
                    let repeats = self
                        .constants
                        .get(&repeats_node)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Op::Tile {what}: no value for its repeats operand; \
                                 materialise it and pass it via \
                                 TritonLowering::with_constants"
                            )
                        })?
                        .to_i64_vec()
                        .map_err(|e| Error::UnreadableConstant {
                            context: format!("Op::Tile {what}: repeats operand"),
                            reason: e,
                        })?;
                    tile_executable(node, &in_shape, &repeats, &what)?
                }
                Op::Expand => {
                    let what = node_label(graph, node_index);
                    let Some(&data) = node.inputs.first() else {
                        return Err(anyhow::anyhow!("Op::Expand {what}: has no input"));
                    };
                    let in_shape =
                        concrete_shape(&graph.nodes[data].shape, "Expand", &what, "its input's")?;
                    // The target shape is an operand, but by this point it is also the
                    // node's own inferred shape, which is the one the output buffer is
                    // allocated from. Reading it here rather than the operand keeps the
                    // copy and the allocation describing the same tensor.
                    let out_shape = concrete_shape(&node.shape, "Expand", &what, "its own output")?;
                    if out_shape.len() < in_shape.len() {
                        return Err(anyhow::anyhow!(
                            "Op::Expand {what}: cannot expand {in_shape:?} to the lower \
                             rank {out_shape:?}"
                        ));
                    }
                    // Broadcasting aligns trailing axes: each input axis must match its
                    // output axis or be 1.
                    let lead = out_shape.len() - in_shape.len();
                    if let Some(d) = (0..in_shape.len())
                        .find(|&d| in_shape[d] != 1 && in_shape[d] != out_shape[lead + d])
                    {
                        return Err(anyhow::anyhow!(
                            "Op::Expand {what}: axis {d} of {in_shape:?} is {} , which \
                             neither matches {} in {out_shape:?} nor is 1",
                            in_shape[d],
                            out_shape[lead + d]
                        ));
                    }
                    strided_copy_executable(
                        node,
                        "Expand",
                        CopyMapping::Expand {
                            in_shape: &in_shape,
                            out_shape: &out_shape,
                        },
                        &what,
                    )?
                }
                Op::ShapeOf { .. } | Op::SizeOf => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ShapeOf/SizeOf — output is metadata, not tensor data"
                    ));
                }
                Op::Compress { .. } | Op::NonZero => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Compress/NonZero {}: how many elements these write is \
                         decided by the data, and the compaction kernel is the easy half \
                         — every buffer in the executable DAG is allocated from a static \
                         shape before the graph runs, so there is nowhere to put a result \
                         whose width is not known until it has run. Where the operand is \
                         constant the folder evaluates the node instead, which is what \
                         the exports seen so far need",
                        node_label(graph, node_index)
                    ));
                }
                Op::Range => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Range — implement as a fill/arange kernel"
                    ));
                }
                // A constant consumed only as shape metadata never reaches here: it
                // is elided above. One that flows data needs its bytes on the device.
                Op::Constant { dtype, shape } => {
                    let what = graph
                        .names
                        .get(&node_index)
                        .map_or_else(|| format!("node {node_index}"), |n| format!("'{n}'"));
                    let value = self.constants.get(&node_index).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Op::Constant {what} ({dtype:?}, shape {shape:?}) flows tensor \
                             data, not shape metadata, so its values are needed, but none \
                             were supplied; pass them via TritonLowering::with_constants"
                        )
                    })?;
                    // `Op::Constant`'s own dtype and shape are the authoritative ones:
                    // they come from the constant's value in the model, whereas a node's
                    // dtype and shape can fall back to a guess when the model carries no
                    // `value_info` for it.
                    let concrete = shape
                        .iter()
                        .copied()
                        .collect::<Option<Vec<usize>>>()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Op::Constant {what} has a dynamic dimension in shape \
                                 {shape:?}; a constant's shape must be static"
                            )
                        })?;
                    constant_executable(*dtype, concrete, value.bytes.clone(), &what)?
                }
                // An operand the model did not supply. It occupies its position so
                // the operands after it keep theirs, and carries no value, so reaching
                // it here means an op read an operand it should have treated as
                // absent. That is a lowering bug in that op, not something to paper
                // over with a zero buffer.
                Op::Omitted => {
                    return Err(anyhow::anyhow!(
                        "Op::Omitted {}: an omitted optional operand reached lowering; \
                         the consuming op should skip it rather than read it",
                        node_label(graph, node_index)
                    ));
                }
                Op::ConstantOfShape { dtype, fill } => {
                    let what = node_label(graph, node_index);
                    fill_executable(node, *dtype, fill, &what)?
                }
                Op::Trilu { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::Trilu — implement as a triangular mask kernel"
                    ));
                }
                Op::Pad { mode } => {
                    let what = node_label(graph, node_index);
                    if mode != "constant" {
                        return Err(anyhow::anyhow!(
                            "TODO: Op::Pad {what}: mode {mode:?}; only a constant pad \
                             is implemented, and reflect, edge and wrap each need \
                             their own coordinate mapping"
                        ));
                    }
                    let in_shape = concrete_shape(
                        &graph.nodes[*node
                            .inputs
                            .first()
                            .ok_or_else(|| anyhow::anyhow!("Op::Pad {what}: no tensor to pad"))?]
                        .shape,
                        "Pad",
                        "operand 0",
                        &what,
                    )?;
                    let out_shape = concrete_shape(&node.shape, "Pad", "output", &what)?;
                    let rank = in_shape.len();
                    if out_shape.len() != rank {
                        return Err(anyhow::anyhow!(
                            "Op::Pad {what}: its output has rank {} but its operand has \
                             rank {rank}",
                            out_shape.len()
                        ));
                    }

                    // pads is [lo.., hi..], over all axes or over the ones `axes` names.
                    let pads = node
                        .inputs
                        .get(1)
                        .and_then(|id| self.constants.get(id))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Op::Pad {what}: its pads have no value; \
                                 materialise them and pass them via \
                                 TritonLowering::with_constants"
                            )
                        })?
                        .to_i64_vec()
                        .map_err(|e| Error::UnreadableConstant {
                            context: format!("Op::Pad {what}: its pads"),
                            reason: e,
                        })?;
                    let axes: Vec<i64> = match node
                        .inputs
                        .get(3)
                        .filter(|&&id| !matches!(graph.nodes[id].op, Op::Omitted))
                    {
                        Some(id) => self
                            .constants
                            .get(id)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::Pad {what}: it names its axes with an operand \
                                         that has no value"
                                )
                            })?
                            .to_i64_vec()
                            .map_err(|e| Error::UnreadableConstant {
                                context: format!("Op::Pad {what}: its axes operand"),
                                reason: e,
                            })?,
                        None => (0..rank as i64).collect(),
                    };
                    if pads.len() != axes.len() * 2 {
                        return Err(anyhow::anyhow!(
                            "Op::Pad {what}: {} pads for {} axes; it takes a leading \
                             and a trailing pad for each",
                            pads.len(),
                            axes.len()
                        ));
                    }

                    // Only the named axes are padded; the rest keep a pad of zero.
                    let mut lo = vec![0i64; rank];
                    for (k, &ax) in axes.iter().enumerate() {
                        let ax = if ax < 0 { ax + rank as i64 } else { ax };
                        let ax =
                            usize::try_from(ax)
                                .ok()
                                .filter(|&a| a < rank)
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Op::Pad {what}: axis {} is outside a rank-{rank} \
                                     tensor",
                                        axes[k]
                                    )
                                })?;
                        lo[ax] = pads[k];
                        let hi = pads[axes.len() + k];
                        let want = in_shape[ax] as i64 + pads[k] + hi;
                        if want != out_shape[ax] as i64 {
                            return Err(anyhow::anyhow!(
                                "Op::Pad {what}: axis {ax} is {} padded by ({}, {hi}), \
                                 which gives {want}, but the output has {}",
                                in_shape[ax],
                                pads[k],
                                out_shape[ax]
                            ));
                        }
                    }

                    // The pad value is operand 2, a constant scalar. An omitted or
                    // absent one means ONNX's default of zero.
                    let value: Vec<u8> = match node
                        .inputs
                        .get(2)
                        .filter(|&&id| !matches!(graph.nodes[id].op, Op::Omitted))
                    {
                        Some(id) => match self.constants.get(id) {
                            Some(t) if !t.bytes.is_empty() => t.bytes.clone(),
                            _ => {
                                return Err(anyhow::anyhow!(
                                    "Op::Pad {what}: its pad value is not a constant; \
                                     the kernel reads it from a parameter slot, so it \
                                     has to be known when the graph is lowered"
                                ));
                            }
                        },
                        None => Vec::new(),
                    };

                    macro_rules! build_pad {
                        ($ty:ty) => {{
                            let rop = PadNdRuntimeOp::<$ty>::new(
                                1024, &out_shape, &in_shape, &lo, &value,
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::Pad {what}: {out_shape:?} padded from \
                                     {in_shape:?} by {lo:?} with a {}-byte value does \
                                     not describe a pad",
                                    value.len()
                                )
                            })?;
                            let kernel = rop.kernel();
                            let name = kernel.name.to_string();
                            let source = kernel.source.clone();
                            let body = kernel.kernel_source.clone();
                            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
                            (name, source, body, erased)
                        }};
                    }
                    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
                        DtypeRepr::F32 => build_pad!(f32),
                        DtypeRepr::F64 => build_pad!(f64),
                        DtypeRepr::I32 => build_pad!(i32),
                        DtypeRepr::I64 => build_pad!(i64),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Op::Pad {what}: {other:?} is not a supported dtype"
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{name}_entry_point"),
                        name,
                        kernel_source,
                        kernel_body,
                        pointwise_fuse_block_size: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op,
                    })
                }
                Op::ReverseSequence { .. } => {
                    return Err(anyhow::anyhow!(
                        "TODO: Op::ReverseSequence — implement as a scatter-copy kernel"
                    ));
                }
                Op::Einsum { equation } => {
                    let what = node_label(graph, node_index);
                    let &[a, b] = node.inputs.as_slice() else {
                        return Err(anyhow::anyhow!(
                            "Op::Einsum {what}: expected two operands, got {}; only the \
                             two-operand form is supported",
                            node.inputs.len()
                        ));
                    };
                    let a_shape = concrete_shape(
                        &graph.nodes[a].shape,
                        "Einsum",
                        &what,
                        "its left operand's",
                    )?;
                    let b_shape = concrete_shape(
                        &graph.nodes[b].shape,
                        "Einsum",
                        &what,
                        "its right operand's",
                    )?;
                    // The equation is planned by the same function shape inference
                    // uses, so the extents the graph records and the strides the
                    // contraction runs under come from one reading of it.
                    let plan = einsum_plan(equation, &a_shape, &b_shape).map_err(|e| {
                        anyhow::anyhow!(
                            "Op::Einsum {what}: cannot plan '{equation}' over \
                                 {a_shape:?} and {b_shape:?}: {e}"
                        )
                    })?;
                    if let Some(recorded) =
                        node.shape.iter().copied().collect::<Option<Vec<usize>>>()
                        && recorded != plan.out_shape
                    {
                        return Err(anyhow::anyhow!(
                            "Op::Einsum {what}: '{equation}' gives {:?} but the node's \
                             shape is {recorded:?}",
                            plan.out_shape
                        ));
                    }
                    einsum_executable(node, &plan, equation, &what)?
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
                Op::ConvTranspose {
                    in_channels,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    groups,
                    has_bias,
                    ..
                } => {
                    let what = node_label(graph, node_index);
                    if *groups != 1 {
                        return Err(anyhow::anyhow!(
                            "TODO: Op::ConvTranspose {what}: {groups} groups; the \
                             kernel sums over every input channel, which is one group"
                        ));
                    }
                    let x_shape = concrete_shape(
                        &graph.nodes[*node.inputs.first().ok_or_else(|| {
                            anyhow::anyhow!("Op::ConvTranspose {what}: no input")
                        })?]
                        .shape,
                        "ConvTranspose",
                        "input",
                        &what,
                    )?;
                    let out_shape = concrete_shape(&node.shape, "ConvTranspose", "output", &what)?;
                    if x_shape.len() != 4 || out_shape.len() != 4 {
                        return Err(anyhow::anyhow!(
                            "Op::ConvTranspose {what}: {x_shape:?} to {out_shape:?}; \
                             this kernel is 2-D, over a rank-4 NCHW tensor"
                        ));
                    }
                    if x_shape[1] != *in_channels || out_shape[1] != *out_channels {
                        return Err(anyhow::anyhow!(
                            "Op::ConvTranspose {what}: it says {in_channels} to \
                             {out_channels} channels but the shapes are {} to {}",
                            x_shape[1],
                            out_shape[1]
                        ));
                    }
                    let operands = node.inputs.len();
                    if *has_bias && operands < 3 {
                        return Err(anyhow::anyhow!(
                            "Op::ConvTranspose {what}: it has a bias but only \
                             {operands} operands"
                        ));
                    }

                    macro_rules! build_convt {
                        ($ty:ty) => {{
                            let rop = ConvTranspose2dRuntimeOp::<$ty>::new(
                                256,
                                x_shape[0],
                                *in_channels,
                                *out_channels,
                                x_shape[2],
                                x_shape[3],
                                out_shape[2],
                                out_shape[3],
                                *kernel_h,
                                *kernel_w,
                                *stride_h,
                                *stride_w,
                                *padding_h,
                                *padding_w,
                                *has_bias,
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::ConvTranspose {what}: a {kernel_h}x{kernel_w} \
                                     kernel at stride {stride_h}x{stride_w} over \
                                     {in_channels} channels leaves nothing to sum"
                                )
                            })?;
                            let kernel = rop.kernel();
                            let name = kernel.name.to_string();
                            let source = kernel.source.clone();
                            let body = kernel.kernel_source.clone();
                            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
                            (name, source, body, erased)
                        }};
                    }
                    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
                        DtypeRepr::F32 => build_convt!(f32),
                        DtypeRepr::F64 => build_convt!(f64),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Op::ConvTranspose {what}: {other:?} is not a float dtype"
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{name}_entry_point"),
                        name,
                        kernel_source,
                        kernel_body,
                        pointwise_fuse_block_size: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op,
                    })
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
                Op::Resize {
                    mode,
                    coordinate_transformation_mode,
                    nearest_mode,
                    antialias,
                } => {
                    let what = node_label(graph, node_index);
                    let in_shape = concrete_shape(
                        &graph.nodes[*node.inputs.first().ok_or_else(|| {
                            anyhow::anyhow!("Op::Resize {what}: no tensor to resize")
                        })?]
                        .shape,
                        "Resize",
                        "operand 0",
                        &what,
                    )?;
                    let out_shape = concrete_shape(&node.shape, "Resize", "output", &what)?;
                    let rank = in_shape.len();
                    if out_shape.len() != rank {
                        return Err(anyhow::anyhow!(
                            "Op::Resize {what}: its output has rank {} but its operand \
                             has rank {rank}",
                            out_shape.len()
                        ));
                    }
                    if *antialias {
                        return Err(anyhow::anyhow!(
                            "TODO: Op::Resize {what}: antialias; downsampling with a \
                             filter needs a wider tap set than the plan builds"
                        ));
                    }

                    let interpolation = match mode.as_str() {
                        "nearest" => {
                            let rounding = match nearest_mode.as_str() {
                                "floor" => ResizeNearest::Floor,
                                "ceil" => ResizeNearest::Ceil,
                                "round_prefer_floor" => ResizeNearest::RoundPreferFloor,
                                "round_prefer_ceil" => ResizeNearest::RoundPreferCeil,
                                other => {
                                    return Err(anyhow::anyhow!(
                                        "Op::Resize {what}: nearest_mode {other:?} is \
                                         not one ONNX defines"
                                    ));
                                }
                            };
                            ResizeMode::Nearest(rounding)
                        }
                        "linear" => ResizeMode::Linear,
                        "cubic" => ResizeMode::Cubic,
                        other => {
                            return Err(anyhow::anyhow!(
                                "TODO: Op::Resize {what}: mode {other:?}; nearest, \
                                 linear and cubic are implemented"
                            ));
                        }
                    };
                    let coords = match coordinate_transformation_mode.as_str() {
                        "half_pixel" | "pytorch_half_pixel" => ResizeCoords::HalfPixel,
                        "align_corners" => ResizeCoords::AlignCorners,
                        "asymmetric" => ResizeCoords::Asymmetric,
                        other => {
                            return Err(anyhow::anyhow!(
                                "TODO: Op::Resize {what}: coordinate transformation \
                                 {other:?}; half_pixel, align_corners and asymmetric \
                                 are implemented"
                            ));
                        }
                    };

                    // Scales, when the resize was given them rather than sizes: ONNX
                    // uses the scale itself for the coordinate transform, which is not
                    // always the ratio of the extents.
                    let scales: Option<Vec<f64>> = node
                        .inputs
                        .get(2)
                        .filter(|&&id| !matches!(graph.nodes[id].op, Op::Omitted))
                        .and_then(|id| self.constants.get(id))
                        .map(|t| t.to_f64_vec())
                        .transpose()
                        .map_err(|e| Error::UnreadableConstant {
                            context: format!("Op::Resize {what}: its scales"),
                            reason: e,
                        })?;

                    // Only the trailing axes may be resized, and at most three of them.
                    let first_resized = (0..rank)
                        .find(|&ax| in_shape[ax] != out_shape[ax])
                        .unwrap_or(rank);
                    let resized = rank - first_resized;
                    if resized > RESIZE_MAX_AXES {
                        return Err(anyhow::anyhow!(
                            "Op::Resize {what}: {in_shape:?} to {out_shape:?} changes \
                             {resized} axes; this kernel resizes at most \
                             {RESIZE_MAX_AXES}, and they have to be the trailing ones"
                        ));
                    }
                    // A resize that changes nothing still has to copy, so keep one axis.
                    let first_resized = first_resized.min(rank - 1);

                    let mut strides = vec![1usize; rank];
                    for ax in (0..rank.saturating_sub(1)).rev() {
                        strides[ax] = strides[ax + 1] * in_shape[ax + 1];
                    }

                    let mut axes = Vec::new();
                    for ax in first_resized..rank {
                        let plan = resize_axis_plan(
                            interpolation,
                            coords,
                            in_shape[ax],
                            out_shape[ax],
                            scales.as_ref().and_then(|s| s.get(ax).copied()),
                        )
                        .map_err(|e| anyhow::anyhow!("Op::Resize {what}: axis {ax}: {e}"))?;
                        axes.push(ResizeAxis {
                            out_ext: out_shape[ax],
                            stride: strides[ax],
                            taps: plan.taps,
                            indices: plan.indices,
                            weights: plan.weights,
                        });
                    }
                    let in_block: usize = in_shape[first_resized..].iter().product();
                    let n_out: usize = out_shape.iter().product();

                    macro_rules! build_resize {
                        ($ty:ty) => {{
                            let rop = ResizeRuntimeOp::<$ty>::new(1024, &axes, in_block, n_out)
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Op::Resize {what}: {} axes do not describe a \
                                         resize this kernel can run",
                                        axes.len()
                                    )
                                })?;
                            let kernel = rop.kernel();
                            let name = kernel.name.to_string();
                            let source = kernel.source.clone();
                            let body = kernel.kernel_source.clone();
                            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
                            (name, source, body, erased)
                        }};
                    }
                    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
                        DtypeRepr::F32 => build_resize!(f32),
                        DtypeRepr::F64 => build_resize!(f64),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Op::Resize {what}: {other:?} is not a float dtype; the \
                                 weights it interpolates with are floats"
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{name}_entry_point"),
                        name,
                        kernel_source,
                        kernel_body,
                        pointwise_fuse_block_size: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op,
                    })
                }
                Op::GridSample {
                    mode,
                    padding_mode,
                    align_corners,
                } => {
                    let what = node_label(graph, node_index);
                    if mode != "bilinear" && mode != "linear" {
                        return Err(anyhow::anyhow!(
                            "TODO: Op::GridSample {what}: mode {mode:?}; only bilinear \
                             is implemented, and nearest and bicubic need their own \
                             tap sets"
                        ));
                    }
                    if padding_mode != "zeros" {
                        return Err(anyhow::anyhow!(
                            "TODO: Op::GridSample {what}: padding mode \
                             {padding_mode:?}; only zeros is implemented, which is a \
                             masked read, where border and reflection each need a \
                             coordinate mapping"
                        ));
                    }
                    if node.inputs.len() != 2 {
                        return Err(anyhow::anyhow!(
                            "Op::GridSample {what}: it has {} operands, not 2 (input, \
                             grid)",
                            node.inputs.len()
                        ));
                    }
                    let x_shape = concrete_shape(
                        &graph.nodes[node.inputs[0]].shape,
                        "GridSample",
                        "input",
                        &what,
                    )?;
                    let grid_shape = concrete_shape(
                        &graph.nodes[node.inputs[1]].shape,
                        "GridSample",
                        "grid",
                        &what,
                    )?;
                    let out_shape = concrete_shape(&node.shape, "GridSample", "output", &what)?;
                    if x_shape.len() != 4 || grid_shape.len() != 4 || out_shape.len() != 4 {
                        return Err(anyhow::anyhow!(
                            "Op::GridSample {what}: {x_shape:?} sampled by \
                             {grid_shape:?} into {out_shape:?}; this kernel samples a \
                             rank-4 NCHW tensor with a rank-4 grid"
                        ));
                    }
                    // grid is [N, H_out, W_out, 2] and the output [N, C, H_out, W_out].
                    let (n, c) = (x_shape[0], x_shape[1]);
                    let (h_out, w_out) = (grid_shape[1], grid_shape[2]);
                    if grid_shape[0] != n || grid_shape[3] != 2 {
                        return Err(anyhow::anyhow!(
                            "Op::GridSample {what}: its grid is {grid_shape:?}; it has \
                             to be [{n}, H_out, W_out, 2]"
                        ));
                    }
                    if out_shape != vec![n, c, h_out, w_out] {
                        return Err(anyhow::anyhow!(
                            "Op::GridSample {what}: its output is {out_shape:?} but the \
                             input and grid call for {:?}",
                            vec![n, c, h_out, w_out]
                        ));
                    }
                    if graph.nodes[node.inputs[1]].dtype != node.dtype {
                        return Err(anyhow::anyhow!(
                            "Op::GridSample {what}: its grid is {:?} but the tensor is \
                             {:?}; the kernel reads both as the same dtype",
                            graph.nodes[node.inputs[1]].dtype,
                            node.dtype
                        ));
                    }

                    macro_rules! build_grid {
                        ($ty:ty) => {{
                            let rop = GridSampleRuntimeOp::<$ty>::new(
                                1024,
                                n,
                                c,
                                x_shape[2],
                                x_shape[3],
                                h_out,
                                w_out,
                                *align_corners,
                            )
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Op::GridSample {what}: {x_shape:?} has an axis \
                                     with no extent to sample along"
                                )
                            })?;
                            let kernel = rop.kernel();
                            let name = kernel.name.to_string();
                            let source = kernel.source.clone();
                            let body = kernel.kernel_source.clone();
                            let erased: Arc<dyn RuntimeOp> = Arc::new(rop);
                            (name, source, body, erased)
                        }};
                    }
                    let (name, kernel_source, kernel_body, runtime_op) = match node.dtype {
                        DtypeRepr::F32 => build_grid!(f32),
                        DtypeRepr::F64 => build_grid!(f64),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Op::GridSample {what}: {other:?} is not a float dtype; \
                                 the weights it samples with are floats"
                            ));
                        }
                    };
                    Box::new(KernelExecutable {
                        entry_point: format!("{name}_entry_point"),
                        name,
                        kernel_source,
                        kernel_body,
                        pointwise_fuse_block_size: None,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        #[cfg(feature = "training")]
                        backward_kernel_source: String::new(),
                        #[cfg(feature = "training")]
                        backward_entry_point: String::new(),
                        runtime_op,
                    })
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
                    let which = match node.op {
                        Op::And => LogicalOp::And,
                        Op::Or => LogicalOp::Or,
                        _ => LogicalOp::Xor,
                    };
                    let what = node_label(graph, node_index);
                    logical_executable(graph, node, which, &what)?
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
                    // A cast to the dtype the operand already has moves bytes and
                    // nothing else, so it is the same copy as Identity. Exporters emit
                    // these in quantity: all 300 in the FoundationStereo export are
                    // F32 to F32, and refusing them would have held up the whole graph
                    // for a conversion none of them asks for.
                    let from = node
                        .inputs
                        .first()
                        .map(|&i| graph.nodes[i].dtype)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Op::Cast {}: no operand to cast",
                                node_label(graph, node_index)
                            )
                        })?;
                    if node.dtype != *to {
                        return Err(anyhow::anyhow!(
                            "Op::Cast {}: casts to {to:?} but the node is {:?}; the \
                             node's dtype sets its buffer's width",
                            node_label(graph, node_index),
                            node.dtype
                        ));
                    }
                    if from == *to {
                        make_num_kernel!(ElemwiseCopyForward(1024), node)
                    } else {
                        cast_executable(node, from, *to, &node_label(graph, node_index))?
                    }
                }
                // Pass-through, as Op::Reshape above.
                Op::Identity => {
                    make_num_kernel!(ElemwiseCopyForward(1024), ElemwiseCopyBackward(1024), node)
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
                // An elided constant has no DAG node to draw an edge from. Its value
                // reached this op through the lowering above, not through the DAG.
                if elided[input_graph_idx] {
                    continue;
                }
                // A reduction whose axes needed moving reads the transpose above
                // instead of its own input.
                let from = match reduce_input_override {
                    Some((original, staging)) if original == input_graph_idx => staging,
                    _ => graph_to_dag[input_graph_idx],
                };
                dag.add_edge(from, dag_idx);
            }
        }

        Ok((dag, graph_to_dag))
    }
}

impl<'a> Lowering<'a> for TritonLowering {
    fn lower(&self, graph: &Graph, mode: LoweringMode) -> Result<Dag<Box<dyn ExecutableOp>>> {
        TritonLowering::lower_with_mapping(self, graph, mode).map(|(dag, _)| dag)
    }

    fn lower_with_mapping(
        &self,
        graph: &Graph,
        mode: LoweringMode,
    ) -> Result<(Dag<Box<dyn ExecutableOp>>, Vec<usize>)> {
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
mod pick_gemm_tile_sizes_tests {
    use super::pick_gemm_tile_sizes;

    #[test]
    fn test_small_shape_gets_smallest_tiles() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 16), (64, 64, 8));
    }

    #[test]
    fn test_large_m_and_n_get_the_largest_tile() {
        // Largest safe tile: 256×128 would exceed the sm_120 opt-in shared-memory ceiling.
        assert_eq!(pick_gemm_tile_sizes(Some(512), 256, 256), (128, 128, 32));
    }

    #[test]
    fn test_large_m_only_widens_block_m_not_block_n() {
        assert_eq!(pick_gemm_tile_sizes(Some(512), 32, 64), (128, 64, 16));
    }

    #[test]
    fn test_large_n_only_widens_block_n_not_block_m() {
        assert_eq!(pick_gemm_tile_sizes(Some(32), 512, 64), (64, 128, 16));
    }

    /// Where the backend needs `matmul_forward` to read its last K tile back from
    /// inside the operands, a tile wider than the contracted axis has nowhere to be
    /// read from.
    #[test]
    #[cfg_attr(feature = "cuda", ignore = "a partial K tile loads on CUDA")]
    fn test_block_k_never_exceeds_the_contracted_extent() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 4).2, 4);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 6).2, 4);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 1).2, 1);
    }

    /// A contraction wide enough for the tile is left alone on every backend.
    #[test]
    fn test_block_k_is_untouched_when_the_contraction_is_wider() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 12).2, 8);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 1921).2, 32);
    }

    #[test]
    fn test_unknown_dynamic_batch_treated_as_small() {
        // m=None should behave the same as an explicit small m, not a large one.
        assert_eq!(
            pick_gemm_tile_sizes(None, 64, 16),
            pick_gemm_tile_sizes(Some(64), 64, 16)
        );
    }

    /// 8 is the narrowest tile worth giving the `dot`, but a tile wider than the
    /// contracted axis has nowhere to be read back from, and `matmul_forward` needs
    /// one whole K tile to exist. Correctness wins: a contraction of 1 gets a tile
    /// of 1, and the few shapes that narrow are small ones anyway.
    #[test]
    fn test_block_k_stays_at_eight_unless_the_contraction_is_narrower() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 8).2, 8);
        let narrow = pick_gemm_tile_sizes(Some(64), 64, 1).2;
        if crate::math::gemm::PARTIAL_K_TILE_IS_PADDED {
            assert_eq!(narrow, 8);
        } else {
            assert_eq!(narrow, 1);
        }
    }

    #[test]
    fn test_block_k_grows_with_k() {
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 8).2, 8);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 32).2, 16);
        assert_eq!(pick_gemm_tile_sizes(Some(64), 64, 128).2, 32);
    }
}
