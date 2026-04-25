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

use teeny_core::{
    graph::{DtypeRepr, Graph, Op, Shape},
    model::{ExecutableOp, Lowering},
    utils::dag::Dag,
};

use crate::{
    activation::{
        elu::{CeluForward, EluForward, SeluForward},
        gelu::{GeluForward, MishForward},
        hard::{
            HardshrinkForward, HardsigmoidForward, HardswishForward, HardtanhForward, Relu6Forward,
        },
        misc::{
            LeakyReluForward, SoftplusForward, SoftshrinkForward, SoftsignForward, ThresholdForward,
        },
        relu::ReluForward,
        sigmoid::{LogsigmoidForward, SigmoidForward, SiluForward},
        softmax::SoftmaxForward,
        tanh::{TanhForward, TanhshrinkForward},
    },
    conv::{conv1d::Conv1dForward, conv2d::Conv2dForward, conv3d::Conv3dForward},
    mlp::{flatten::FlattenForward, linear::LinearForward},
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
        let (ks, ep) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::I8  => { let k = $K::<i8>::new($($arg),*);  (k.kernel_source, k.entry_point) }
            DtypeRepr::I16 => { let k = $K::<i16>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::I32 => { let k = $K::<i32>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::I64 => { let k = $K::<i64>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::U8  => { let k = $K::<u8>::new($($arg),*);  (k.kernel_source, k.entry_point) }
            DtypeRepr::U16 => { let k = $K::<u16>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::U32 => { let k = $K::<u32>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::U64 => { let k = $K::<u64>::new($($arg),*); (k.kernel_source, k.entry_point) }
            other => return Err(anyhow::anyhow!("{:?} is not a supported Num dtype for {}", other, stringify!($K))),
        };
        Box::new(KernelExecutable {
            kernel_source: ks,
            entry_point: ep,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
        })
    }};
}

/// Dispatch to a D: Float kernel based on `$node.dtype`.
/// Usage: `make_float_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_float_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let (ks, ep) = match $node.dtype {
            DtypeRepr::F32 => { let k = $K::<f32>::new($($arg),*); (k.kernel_source, k.entry_point) }
            DtypeRepr::F64 => { let k = $K::<f64>::new($($arg),*); (k.kernel_source, k.entry_point) }
            other => return Err(anyhow::anyhow!("{:?} is not a Float dtype for {}", other, stringify!($K))),
        };
        Box::new(KernelExecutable {
            kernel_source: ks,
            entry_point: ep,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
        })
    }};
}

/// Build a KernelExecutable for kernels without a D type parameter (hardcoded f32).
/// Usage: `make_untyped_kernel!(KernelType(arg1, arg2, ...), node)`
macro_rules! make_untyped_kernel {
    ($K:ident ($($arg:expr),*), $node:expr) => {{
        let k = $K::new($($arg),*);
        Box::new(KernelExecutable {
            kernel_source: k.kernel_source,
            entry_point: k.entry_point,
            shape: $node.shape.clone(),
            dtype: $node.dtype,
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
    pub kernel_source: String,
    pub entry_point: String,
    pub shape: Shape,
    pub dtype: DtypeRepr,
}

impl ExecutableOp for KernelExecutable {
    fn kernel_source(&self) -> &str {
        &self.kernel_source
    }

    fn kernel_entry_point(&self) -> &str {
        &self.entry_point
    }

    fn output_shape(&self) -> &Shape {
        &self.shape
    }

    fn output_dtype(&self) -> DtypeRepr {
        self.dtype
    }
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
    fn lower(&self, graph: &Graph) -> Result<Dag<Box<dyn ExecutableOp>>> {
        let node_indexes = graph.topological_sort();
        let mut dag: Dag<Box<dyn ExecutableOp>> = Dag::new();
        // Maps graph node index → DAG node index (one-to-one since we add every node)
        let mut graph_to_dag = vec![0usize; graph.nodes.len()];

        for node_index in node_indexes {
            let node = &graph.nodes[node_index];

            let executable: Box<dyn ExecutableOp> = match &node.op {
                Op::Input => Box::new(KernelExecutable {
                    kernel_source: String::new(),
                    entry_point: String::new(),
                    shape: node.shape.clone(),
                    dtype: node.dtype,
                }),

                // --- Linear / MLP ---
                Op::Linear { has_bias, .. } => {
                    make_num_kernel!(LinearForward(*has_bias, 32, 64, 32, 8), node)
                }
                Op::Flatten => make_num_kernel!(FlattenForward(32, 256), node),

                // --- Convolution ---
                Op::Conv1d { kernel_l, stride, .. } => {
                    make_num_kernel!(Conv1dForward(*kernel_l as i32, *stride as i32, 32), node)
                }
                Op::Conv2d { kernel_h, kernel_w, stride_h, stride_w, .. } => {
                    make_num_kernel!(
                        Conv2dForward(*kernel_h as i32, *kernel_w as i32, *stride_h as i32, *stride_w as i32, 16),
                        node
                    )
                }
                Op::Conv3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, .. } => {
                    make_num_kernel!(
                        Conv3dForward(*kernel_d as i32, *kernel_h as i32, *kernel_w as i32, *stride_d as i32, *stride_h as i32, *stride_w as i32, 8),
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
                Op::Relu => make_num_kernel!(ReluForward(1024), node),

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
                Op::Softmax { .. } => make_float_kernel!(SoftmaxForward(1024), node),
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
