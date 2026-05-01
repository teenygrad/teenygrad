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

use std::sync::Arc;
use teeny_core::{
    graph::{DtypeRepr, Graph, Op, Shape},
    model::{ExecutableOp, Lowering, LoweringMode, RuntimeOp},
    utils::dag::Dag,
};

use crate::nn::{
    activation::{
        elu::{CeluForward, EluForward, SeluForward},
        gelu::{GeluForward, MishForward},
        hard::{
            HardshrinkForward, HardsigmoidForward, HardswishForward, HardtanhForward, Relu6Forward,
        },
        misc::{
            LeakyReluForward, SoftplusForward, SoftshrinkForward, SoftsignForward, ThresholdForward,
        },
        relu::{ReluBackward, ReluForward},
        sigmoid::{LogsigmoidForward, SigmoidForward, SiluForward},
        softmax::SoftmaxForward,
        tanh::{TanhForward, TanhshrinkForward},
    },
    conv::{conv1d::Conv1dForward, conv2d::Conv2dForward, conv3d::Conv3dForward},
    mlp::{flatten::FlattenForward, linear::{LinearBackward, LinearForward}},
    norm::{
        batchnorm::BatchNormForwardInference,
        groupnorm::GroupNormForwardInference,
        instancenorm::InstanceNormForwardInference,
        layernorm::LayerNormForwardInference,
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
        avgpool1d::Avgpool1dForward, avgpool2d::Avgpool2dForward, avgpool3d::Avgpool3dForward,
        lppool1d::Lppool1dForward, lppool2d::Lppool2dForward, lppool3d::Lppool3dForward,
        maxpool1d::Maxpool1dForward, maxpool2d::Maxpool2dForward, maxpool3d::Maxpool3dForward,
    },
};

use crate::errors::Result;

#[cfg(feature = "training")]
use crate::nn::norm::batchnorm::{
    BatchNormNormalizeForward, BatchNormNormalizeRuntimeOp,
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
// make_untyped_kernel! — for kernels without a D type parameter
// ---------------------------------------------------------------------------

/// Dispatch to a D: Num kernel based on `$node.dtype`.
/// Usage: `make_num_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_num_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let (name, ks, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I8  => { let k = $K::<i8>::new($($arg),*);  let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I16 => { let k = $K::<i16>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I32 => { let k = $K::<i32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I64 => { let k = $K::<i64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U8  => { let k = $K::<u8>::new($($arg),*);  let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U16 => { let k = $K::<u16>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U32 => { let k = $K::<u32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U64 => { let k = $K::<u64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($K))),
        };
        Box::new(KernelExecutable {
            name,
            kernel_source: ks,
            entry_point: "entry_point".to_string(),
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: String::new(),
            #[cfg(feature = "training")]
            backward_entry_point: "entry_point".to_string(),
            runtime_op: rop,
        })
    }};
    // Variant with explicit backward kernel type (for ops that have backward support)
    ($K:ident ($($arg:expr),*), $Bwd:ident ($($barg:expr),*), $node:expr) => {{
        let (name, ks, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I8  => { let k = $K::<i8>::new($($arg),*);  let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I16 => { let k = $K::<i16>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I32 => { let k = $K::<i32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::I64 => { let k = $K::<i64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U8  => { let k = $K::<u8>::new($($arg),*);  let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U16 => { let k = $K::<u16>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U32 => { let k = $K::<u32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::U64 => { let k = $K::<u64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($K))),
        };
        #[cfg(feature = "training")]
        let bwd_ks = match $node.dtype {
            DtypeRepr::F32 => $Bwd::<f32>::new($($barg),*).source.clone(),
            DtypeRepr::F64 => $Bwd::<f64>::new($($barg),*).source.clone(),
            DtypeRepr::I8  => $Bwd::<i8>::new($($barg),*).source.clone(),
            DtypeRepr::I16 => $Bwd::<i16>::new($($barg),*).source.clone(),
            DtypeRepr::I32 => $Bwd::<i32>::new($($barg),*).source.clone(),
            DtypeRepr::I64 => $Bwd::<i64>::new($($barg),*).source.clone(),
            DtypeRepr::U8  => $Bwd::<u8>::new($($barg),*).source.clone(),
            DtypeRepr::U16 => $Bwd::<u16>::new($($barg),*).source.clone(),
            DtypeRepr::U32 => $Bwd::<u32>::new($($barg),*).source.clone(),
            DtypeRepr::U64 => $Bwd::<u64>::new($($barg),*).source.clone(),
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($Bwd))),
        };
        Box::new(KernelExecutable {
            name,
            kernel_source: ks,
            entry_point: "entry_point".to_string(),
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: bwd_ks,
            #[cfg(feature = "training")]
            backward_entry_point: "entry_point".to_string(),
            runtime_op: rop,
        })
    }};
}

/// Dispatch to a D: Float kernel based on `$node.dtype`.
/// Usage: `make_float_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_float_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let (name, ks, rop) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); let nm = k.name.to_string(); let src = k.source.clone(); let r: Arc<dyn RuntimeOp> = Arc::new(k); (nm, src, r) }
            other => return Err(anyhow::anyhow!("{:?} is not a Float dtype for {}", other, stringify!($K))),
        };
        Box::new(KernelExecutable {
            name,
            kernel_source: ks,
            entry_point: "entry_point".to_string(),
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: String::new(),
            #[cfg(feature = "training")]
            backward_entry_point: "entry_point".to_string(),
            runtime_op: rop,
        })
    }};
}

/// Build a KernelExecutable for kernels without a D type parameter (hardcoded f32).
/// Usage: `make_untyped_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_untyped_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let k = $K::new($($arg),*);
        let nm = k.name.to_string();
        let src = k.source.clone();
        let rop: Arc<dyn RuntimeOp> = Arc::new(k);
        Box::new(KernelExecutable {
            name: nm,
            kernel_source: src,
            entry_point: "entry_point".to_string(),
            shape: $node.shape.clone(),
            dtype: $node.dtype,
            #[cfg(feature = "training")]
            backward_kernel_source: String::new(),
            #[cfg(feature = "training")]
            backward_entry_point: "entry_point".to_string(),
            runtime_op: rop,
        })
    }};
}

// ---------------------------------------------------------------------------
// KernelExecutable — compilable unit produced by TritonLowering
// ---------------------------------------------------------------------------

/// A lowered op that carries the kernel source needed for compilation.
///
/// Callers that have `teeny-compiler` as a dependency can pass `kernel_source`
/// and `kernel_entry_point` to `compile_kernel` along with a chosen `Target`.
pub struct KernelExecutable {
    pub name: String,
    pub kernel_source: String,
    pub entry_point: String,
    pub shape: Shape,
    pub dtype: DtypeRepr,
    /// Runtime dispatch object: how to pack args and compute the launch grid.
    /// `Input` nodes carry a no-op implementation.
    pub runtime_op: Arc<dyn RuntimeOp>,
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
            fn n_activation_inputs(&self) -> usize { unimplemented!(concat!(stringify!($T), " has no runtime support")) }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { unimplemented!() }
            fn pack_args(&self, _: &[(teeny_core::model::RawPtr, &[usize])], _: &[teeny_core::model::RawPtr], _: teeny_core::model::RawPtr, _: &[usize], _: i32, _: &mut dyn teeny_core::device::program::ArgVisitor) { unimplemented!() }
            fn block(&self) -> [u32; 3] { unimplemented!() }
            fn grid(&self, _: &[usize]) -> [u32; 3] { unimplemented!() }
        }
    };
}

macro_rules! impl_stub_runtime_op_float {
    ($T:ident) => {
        impl<D: teeny_core::dtype::Float + Send + Sync + 'static> RuntimeOp for $T<D> {
            fn n_activation_inputs(&self) -> usize { unimplemented!(concat!(stringify!($T), " has no runtime support")) }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { unimplemented!() }
            fn pack_args(&self, _: &[(teeny_core::model::RawPtr, &[usize])], _: &[teeny_core::model::RawPtr], _: teeny_core::model::RawPtr, _: &[usize], _: i32, _: &mut dyn teeny_core::device::program::ArgVisitor) { unimplemented!() }
            fn block(&self) -> [u32; 3] { unimplemented!() }
            fn grid(&self, _: &[usize]) -> [u32; 3] { unimplemented!() }
        }
    };
}

macro_rules! impl_stub_runtime_op_untyped {
    ($T:ident) => {
        impl RuntimeOp for $T {
            fn n_activation_inputs(&self) -> usize { unimplemented!(concat!(stringify!($T), " has no runtime support")) }
            fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { unimplemented!() }
            fn pack_args(&self, _: &[(teeny_core::model::RawPtr, &[usize])], _: &[teeny_core::model::RawPtr], _: teeny_core::model::RawPtr, _: &[usize], _: i32, _: &mut dyn teeny_core::device::program::ArgVisitor) { unimplemented!() }
            fn block(&self) -> [u32; 3] { unimplemented!() }
            fn grid(&self, _: &[usize]) -> [u32; 3] { unimplemented!() }
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
impl_stub_runtime_op_num!(Maxpool2dForward);
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

// Activation — untyped (no D type parameter; hardcoded f32)
impl_stub_runtime_op_untyped!(EluForward);
impl_stub_runtime_op_untyped!(SeluForward);
impl_stub_runtime_op_untyped!(CeluForward);
impl_stub_runtime_op_untyped!(GeluForward);
impl_stub_runtime_op_untyped!(MishForward);
impl_stub_runtime_op_untyped!(HardtanhForward);
impl_stub_runtime_op_untyped!(Relu6Forward);
impl_stub_runtime_op_untyped!(HardsigmoidForward);
impl_stub_runtime_op_untyped!(HardswishForward);
impl_stub_runtime_op_untyped!(HardshrinkForward);
impl_stub_runtime_op_untyped!(LeakyReluForward);
impl_stub_runtime_op_untyped!(ThresholdForward);
impl_stub_runtime_op_untyped!(SoftsignForward);
impl_stub_runtime_op_untyped!(SoftshrinkForward);
impl_stub_runtime_op_untyped!(SoftplusForward);
impl_stub_runtime_op_untyped!(SigmoidForward);
impl_stub_runtime_op_untyped!(SiluForward);
impl_stub_runtime_op_untyped!(LogsigmoidForward);
impl_stub_runtime_op_untyped!(TanhForward);
impl_stub_runtime_op_untyped!(TanhshrinkForward);

// ---------------------------------------------------------------------------
// No-op RuntimeOp for Input placeholder nodes
// ---------------------------------------------------------------------------

struct InputRuntimeOp;

impl RuntimeOp for InputRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 0 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }
    fn pack_args(&self, _: &[(teeny_core::model::RawPtr, &[usize])], _: &[teeny_core::model::RawPtr], _: teeny_core::model::RawPtr, _: &[usize], _: i32, _: &mut dyn teeny_core::device::program::ArgVisitor) {}
    fn block(&self) -> [u32; 3] { [1, 1, 1] }
    fn grid(&self, _: &[usize]) -> [u32; 3] { [0, 0, 0] }
}

// ---------------------------------------------------------------------------
// TritonLowering
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct TritonLowering {}

impl TritonLowering {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<'a> Lowering<'a> for TritonLowering {
    fn lower(&self, graph: &Graph, mode: LoweringMode) -> Result<Dag<Box<dyn ExecutableOp>>> {
        let _ = mode; // used by #[cfg(feature = "training")] branch below
        let node_indexes = graph.topological_sort();
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        // Maps graph node index → DAG node index (one-to-one since we add every node)
        let mut graph_to_dag = vec![0usize; graph.nodes.len()];

        for node_index in node_indexes {
            let node = &graph.nodes[node_index];

            // Training BatchNorm needs two sequential DAG nodes: stats then normalize.
            #[cfg(feature = "training")]
            if mode == LoweringMode::Training {
                if let Op::BatchNorm1d { num_features, eps, momentum, .. }
                     | Op::BatchNorm2d { num_features, eps, momentum, .. }
                     | Op::BatchNorm3d { num_features, eps, momentum, .. } = &node.op
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
                            other => return Err(anyhow::anyhow!(
                                "{:?} is not a Float dtype for BatchNormStatsForward", other
                            )),
                        };

                    let stats_node = Box::new(KernelExecutable {
                        name: stats_name,
                        kernel_source: stats_src,
                        entry_point: "entry_point".to_string(),
                        shape: vec![Some(2 * c)],
                        dtype: node.dtype,
                        backward_kernel_source: String::new(),
                        backward_entry_point: "entry_point".to_string(),
                        runtime_op: stats_rop,
                    }) as Box<dyn ExecutableOp>;

                    let stats_dag_idx = dag.add_node(stats_node);
                    for &input_graph_idx in &node.inputs {
                        dag.add_edge(graph_to_dag[input_graph_idx], stats_dag_idx);
                    }

                    let (norm_name, norm_src, norm_rop): (String, String, Arc<dyn RuntimeOp>) =
                        match node.dtype {
                            DtypeRepr::F32 => {
                                let k = BatchNormNormalizeForward::<f32>::new(BLOCK_N);
                                let src = k.source.clone();
                                let rop: Arc<dyn RuntimeOp> = Arc::new(
                                    BatchNormNormalizeRuntimeOp::<f32>::new(BLOCK_N),
                                );
                                (k.name.to_string(), src, rop)
                            }
                            DtypeRepr::F64 => {
                                let k = BatchNormNormalizeForward::<f64>::new(BLOCK_N);
                                let src = k.source.clone();
                                let rop: Arc<dyn RuntimeOp> = Arc::new(
                                    BatchNormNormalizeRuntimeOp::<f64>::new(BLOCK_N),
                                );
                                (k.name.to_string(), src, rop)
                            }
                            other => return Err(anyhow::anyhow!(
                                "{:?} is not a Float dtype for BatchNormNormalizeForward", other
                            )),
                        };

                    let norm_node = Box::new(KernelExecutable {
                        name: norm_name,
                        kernel_source: norm_src,
                        entry_point: "entry_point".to_string(),
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                        backward_kernel_source: String::new(),
                        backward_entry_point: "entry_point".to_string(),
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
            }

            let executable: Box<dyn ExecutableOp> = match &node.op {
                Op::Input => Box::new(KernelExecutable {
                    name: "input".to_string(),
                    kernel_source: String::new(),
                    entry_point: String::new(),
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                    #[cfg(feature = "training")]
                    backward_kernel_source: String::new(),
                    #[cfg(feature = "training")]
                    backward_entry_point: "entry_point".to_string(),
                    runtime_op: Arc::new(InputRuntimeOp),
                }),

                // --- Linear / MLP ---
                Op::Linear { has_bias, .. } => {
                    make_num_kernel!(LinearForward(*has_bias, 32, 64, 32, 8), LinearBackward(*has_bias, 32, 64, 32, 8), node)
                }
                Op::Flatten => make_num_kernel!(FlattenForward(32, 256), node),

                // --- Normalisation ---
                Op::BatchNorm1d { .. } | Op::BatchNorm2d { .. } | Op::BatchNorm3d { .. } => {
                    make_float_kernel!(BatchNormForwardInference(64), node)
                }
                Op::LayerNorm { .. } => {
                    make_float_kernel!(LayerNormForwardInference(1024), node)
                }
                Op::RmsNorm { .. } => {
                    make_float_kernel!(RmsNormForward(1024), node)
                }
                Op::GroupNorm { .. } => {
                    make_float_kernel!(GroupNormForwardInference(256), node)
                }
                Op::InstanceNorm1d { .. } | Op::InstanceNorm2d { .. } | Op::InstanceNorm3d { .. } => {
                    make_float_kernel!(InstanceNormForwardInference(256), node)
                }

                // --- Convolution ---
                Op::Conv1d { kernel_l, stride, padding, .. } => {
                    make_num_kernel!(Conv1dForward(*kernel_l as i32, *stride as i32, *padding as i32, 32), node)
                }
                Op::Conv2d { kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, .. } => {
                    make_num_kernel!(
                        Conv2dForward(*kernel_h as i32, *kernel_w as i32, *stride_h as i32, *stride_w as i32, *padding_h as i32, *padding_w as i32, 16),
                        node
                    )
                }
                Op::Conv3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, .. } => {
                    make_num_kernel!(
                        Conv3dForward(*kernel_d as i32, *kernel_h as i32, *kernel_w as i32, *stride_d as i32, *stride_h as i32, *stride_w as i32, *padding_d as i32, *padding_h as i32, *padding_w as i32, 8),
                        node
                    )
                }

                // --- Pooling ---
                Op::AvgPool1d { kernel_l, stride } => {
                    make_num_kernel!(Avgpool1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::AvgPool2d { kernel_h, kernel_w, stride_h, stride_w } => {
                    make_num_kernel!(
                        Avgpool2dForward(*kernel_h as i32, *kernel_w as i32, *stride_h as i32, *stride_w as i32, 16),
                        node
                    )
                }
                Op::AvgPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w } => {
                    make_num_kernel!(
                        Avgpool3dForward(*kernel_d as i32, *kernel_h as i32, *kernel_w as i32, *stride_d as i32, *stride_h as i32, *stride_w as i32, 8),
                        node
                    )
                }
                Op::MaxPool1d { kernel_l, stride } => {
                    make_num_kernel!(Maxpool1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::MaxPool2d { kernel_h, kernel_w, stride_h, stride_w } => {
                    make_num_kernel!(
                        Maxpool2dForward(*kernel_h as i32, *kernel_w as i32, *stride_h as i32, *stride_w as i32, 16),
                        node
                    )
                }
                Op::MaxPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w } => {
                    make_num_kernel!(
                        Maxpool3dForward(*kernel_d as i32, *kernel_h as i32, *kernel_w as i32, *stride_d as i32, *stride_h as i32, *stride_w as i32, 8),
                        node
                    )
                }
                Op::LpPool1d { kernel_l, stride, .. } => {
                    make_float_kernel!(Lppool1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::LpPool2d { kernel_h, kernel_w, stride_h, stride_w, .. } => {
                    make_float_kernel!(
                        Lppool2dForward(*kernel_h as i32, *kernel_w as i32, *stride_h as i32, *stride_w as i32, 16),
                        node
                    )
                }
                Op::LpPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, .. } => {
                    make_float_kernel!(
                        Lppool3dForward(*kernel_d as i32, *kernel_h as i32, *kernel_w as i32, *stride_d as i32, *stride_h as i32, *stride_w as i32, 8),
                        node
                    )
                }

                // --- Padding ---
                Op::ConstantPad1d { pad_left, pad_right, .. } => {
                    make_num_kernel!(ConstantPad1dForward(*pad_left as i32, *pad_right as i32, 32), node)
                }
                Op::ConstantPad2d { pad_l, pad_r, pad_t, pad_b, .. } => {
                    make_num_kernel!(
                        ConstantPad2dForward(*pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16),
                        node
                    )
                }
                Op::ConstantPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2, .. } => {
                    make_num_kernel!(
                        ConstantPad3dForward(*pad_d1 as i32, *pad_d2 as i32, *pad_h1 as i32, *pad_h2 as i32, *pad_w1 as i32, *pad_w2 as i32, 8),
                        node
                    )
                }
                Op::ReflectionPad1d { pad_left, pad_right } => {
                    make_num_kernel!(ReflectionPad1dForward(*pad_left as i32, *pad_right as i32, 32), node)
                }
                Op::ReflectionPad2d { pad_l, pad_r, pad_t, pad_b } => {
                    make_num_kernel!(
                        ReflectionPad2dForward(*pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16),
                        node
                    )
                }
                Op::ReflectionPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
                    make_num_kernel!(
                        ReflectionPad3dForward(*pad_d1 as i32, *pad_d2 as i32, *pad_h1 as i32, *pad_h2 as i32, *pad_w1 as i32, *pad_w2 as i32, 8),
                        node
                    )
                }
                Op::ReplicationPad1d { pad_left, pad_right } => {
                    make_num_kernel!(ReplicationPad1dForward(*pad_left as i32, *pad_right as i32, 32), node)
                }
                Op::ReplicationPad2d { pad_l, pad_r, pad_t, pad_b } => {
                    make_num_kernel!(
                        ReplicationPad2dForward(*pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16),
                        node
                    )
                }
                Op::ReplicationPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
                    make_num_kernel!(
                        ReplicationPad3dForward(*pad_d1 as i32, *pad_d2 as i32, *pad_h1 as i32, *pad_h2 as i32, *pad_w1 as i32, *pad_w2 as i32, 8),
                        node
                    )
                }
                Op::CircularPad1d { pad_left, pad_right } => {
                    make_num_kernel!(CircularPad1dForward(*pad_left as i32, *pad_right as i32, 32), node)
                }
                Op::CircularPad2d { pad_l, pad_r, pad_t, pad_b } => {
                    make_num_kernel!(
                        CircularPad2dForward(*pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16),
                        node
                    )
                }
                Op::CircularPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
                    make_num_kernel!(
                        CircularPad3dForward(*pad_d1 as i32, *pad_d2 as i32, *pad_h1 as i32, *pad_h2 as i32, *pad_w1 as i32, *pad_w2 as i32, 8),
                        node
                    )
                }

                // --- Activation (D: Num) ---
                Op::Relu => make_num_kernel!(ReluForward(1024), ReluBackward(1024), node),

                // --- Activation (hardcoded f32 — no D type parameter) ---
                Op::Elu { .. }       => make_untyped_kernel!(EluForward(1024), node),
                Op::Selu             => make_untyped_kernel!(SeluForward(1024), node),
                Op::Celu { .. }      => make_untyped_kernel!(CeluForward(1024), node),
                Op::Gelu             => make_untyped_kernel!(GeluForward(1024), node),
                Op::Mish             => make_untyped_kernel!(MishForward(1024), node),
                Op::Hardtanh { .. }  => make_untyped_kernel!(HardtanhForward(1024), node),
                Op::Relu6            => make_untyped_kernel!(Relu6Forward(1024), node),
                Op::Hardsigmoid      => make_untyped_kernel!(HardsigmoidForward(1024), node),
                Op::Hardswish        => make_untyped_kernel!(HardswishForward(1024), node),
                Op::Hardshrink { .. }  => make_untyped_kernel!(HardshrinkForward(1024), node),
                Op::LeakyRelu { .. }   => make_untyped_kernel!(LeakyReluForward(1024), node),
                Op::Threshold { .. }   => make_untyped_kernel!(ThresholdForward(1024), node),
                Op::Softsign           => make_untyped_kernel!(SoftsignForward(1024), node),
                Op::Softshrink { .. }  => make_untyped_kernel!(SoftshrinkForward(1024), node),
                Op::Softplus { .. }    => make_untyped_kernel!(SoftplusForward(1024), node),
                Op::Sigmoid            => make_untyped_kernel!(SigmoidForward(1024), node),
                Op::Silu               => make_untyped_kernel!(SiluForward(1024), node),
                Op::Logsigmoid         => make_untyped_kernel!(LogsigmoidForward(1024), node),
                Op::Tanh               => make_untyped_kernel!(TanhForward(1024), node),
                Op::Tanhshrink         => make_untyped_kernel!(TanhshrinkForward(1024), node),

                // --- Activation (D: Float) ---
                Op::Softmax { .. } => {
                    // BLOCK_SIZE must be >= n_cols (the last dim), rounded up to next power of 2.
                    let n_cols = node.shape.last().and_then(|d| *d).unwrap_or(1024);
                    let block_size = n_cols.next_power_of_two() as i32;
                    make_float_kernel!(SoftmaxForward(block_size), node)
                }
            };

            let dag_idx = dag.add_node(executable);
            graph_to_dag[node_index] = dag_idx;

            for &input_graph_idx in &node.inputs {
                dag.add_edge(graph_to_dag[input_graph_idx], dag_idx);
            }
        }

        Ok(dag)
    }
}
