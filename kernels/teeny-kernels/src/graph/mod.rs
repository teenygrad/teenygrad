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
        hard::{HardshrinkForward, HardsigmoidForward, HardswishForward, HardtanhForward, Relu6Forward},
        misc::{LeakyReluForward, SoftplusForward, SoftshrinkForward, SoftsignForward, ThresholdForward},
        relu::ReluForward,
        sigmoid::{LogsigmoidForward, SigmoidForward, SiluForward},
        softmax::SoftmaxForward,
        tanh::{TanhForward, TanhshrinkForward},
    },
    conv::{
        conv1d::Conv1dForward,
        conv2d::Conv2dForward,
        conv3d::Conv3dForward,
    },
    mlp::{flatten::FlattenForward, linear::LinearForward},
    pad::{
        circular_pad1d::CircularPad1dForward,
        circular_pad2d::CircularPad2dForward,
        circular_pad3d::CircularPad3dForward,
        constant_pad1d::ConstantPad1dForward,
        constant_pad2d::ConstantPad2dForward,
        constant_pad3d::ConstantPad3dForward,
        reflection_pad1d::ReflectionPad1dForward,
        reflection_pad2d::ReflectionPad2dForward,
        reflection_pad3d::ReflectionPad3dForward,
        replication_pad1d::ReplicationPad1dForward,
        replication_pad2d::ReplicationPad2dForward,
        replication_pad3d::ReplicationPad3dForward,
    },
    pool::{
        avgpool1d::Avgpool1dForward,
        avgpool2d::Avgpool2dForward,
        avgpool3d::Avgpool3dForward,
        lppool1d::Lppool1dForward,
        lppool2d::Lppool2dForward,
        lppool3d::Lppool3dForward,
        maxpool1d::Maxpool1dForward,
        maxpool2d::Maxpool2dForward,
        maxpool3d::Maxpool3dForward,
    },
};

use crate::errors::Result;

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
                    let k = LinearForward::<f32>::new(*has_bias, 32, 64, 32, 8);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Flatten => {
                    let k = FlattenForward::<f32>::new(32, 256);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }

                // --- Convolution ---
                Op::Conv1d { kernel_l, stride, .. } => {
                    let k = Conv1dForward::<f32>::new(*kernel_l as i32, *stride as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Conv2d { kernel_h, kernel_w, stride_h, stride_w, .. } => {
                    let k = Conv2dForward::<f32>::new(
                        *kernel_h as i32, *kernel_w as i32,
                        *stride_h as i32, *stride_w as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Conv3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, .. } => {
                    let k = Conv3dForward::<f32>::new(
                        *kernel_d as i32, *kernel_h as i32, *kernel_w as i32,
                        *stride_d as i32, *stride_h as i32, *stride_w as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }

                // --- Pooling ---
                Op::AvgPool1d { kernel_l, stride } => {
                    let k = Avgpool1dForward::<f32>::new(*kernel_l as i32, *stride as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::AvgPool2d { kernel_h, kernel_w, stride_h, stride_w } => {
                    let k = Avgpool2dForward::<f32>::new(
                        *kernel_h as i32, *kernel_w as i32,
                        *stride_h as i32, *stride_w as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::AvgPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w } => {
                    let k = Avgpool3dForward::<f32>::new(
                        *kernel_d as i32, *kernel_h as i32, *kernel_w as i32,
                        *stride_d as i32, *stride_h as i32, *stride_w as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::MaxPool1d { kernel_l, stride } => {
                    let k = Maxpool1dForward::<f32>::new(*kernel_l as i32, *stride as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::MaxPool2d { kernel_h, kernel_w, stride_h, stride_w } => {
                    let k = Maxpool2dForward::<f32>::new(
                        *kernel_h as i32, *kernel_w as i32,
                        *stride_h as i32, *stride_w as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::MaxPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w } => {
                    let k = Maxpool3dForward::<f32>::new(
                        *kernel_d as i32, *kernel_h as i32, *kernel_w as i32,
                        *stride_d as i32, *stride_h as i32, *stride_w as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::LpPool1d { kernel_l, stride, .. } => {
                    let k = Lppool1dForward::<f32>::new(*kernel_l as i32, *stride as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::LpPool2d { kernel_h, kernel_w, stride_h, stride_w, .. } => {
                    let k = Lppool2dForward::<f32>::new(
                        *kernel_h as i32, *kernel_w as i32,
                        *stride_h as i32, *stride_w as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::LpPool3d { kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, .. } => {
                    let k = Lppool3dForward::<f32>::new(
                        *kernel_d as i32, *kernel_h as i32, *kernel_w as i32,
                        *stride_d as i32, *stride_h as i32, *stride_w as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }

                // --- Padding ---
                Op::ConstantPad1d { pad_left, pad_right, .. } => {
                    let k = ConstantPad1dForward::<f32>::new(*pad_left as i32, *pad_right as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ConstantPad2d { pad_l, pad_r, pad_t, pad_b, .. } => {
                    let k = ConstantPad2dForward::<f32>::new(
                        *pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ConstantPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2, .. } => {
                    let k = ConstantPad3dForward::<f32>::new(
                        *pad_d1 as i32, *pad_d2 as i32,
                        *pad_h1 as i32, *pad_h2 as i32,
                        *pad_w1 as i32, *pad_w2 as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ReflectionPad1d { pad_left, pad_right } => {
                    let k = ReflectionPad1dForward::<f32>::new(*pad_left as i32, *pad_right as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ReflectionPad2d { pad_l, pad_r, pad_t, pad_b } => {
                    let k = ReflectionPad2dForward::<f32>::new(
                        *pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ReflectionPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
                    let k = ReflectionPad3dForward::<f32>::new(
                        *pad_d1 as i32, *pad_d2 as i32,
                        *pad_h1 as i32, *pad_h2 as i32,
                        *pad_w1 as i32, *pad_w2 as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ReplicationPad1d { pad_left, pad_right } => {
                    let k = ReplicationPad1dForward::<f32>::new(*pad_left as i32, *pad_right as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ReplicationPad2d { pad_l, pad_r, pad_t, pad_b } => {
                    let k = ReplicationPad2dForward::<f32>::new(
                        *pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::ReplicationPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
                    let k = ReplicationPad3dForward::<f32>::new(
                        *pad_d1 as i32, *pad_d2 as i32,
                        *pad_h1 as i32, *pad_h2 as i32,
                        *pad_w1 as i32, *pad_w2 as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::CircularPad1d { pad_left, pad_right } => {
                    let k = CircularPad1dForward::<f32>::new(*pad_left as i32, *pad_right as i32, 32);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::CircularPad2d { pad_l, pad_r, pad_t, pad_b } => {
                    let k = CircularPad2dForward::<f32>::new(
                        *pad_t as i32, *pad_b as i32, *pad_l as i32, *pad_r as i32, 16,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::CircularPad3d { pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2 } => {
                    let k = CircularPad3dForward::<f32>::new(
                        *pad_d1 as i32, *pad_d2 as i32,
                        *pad_h1 as i32, *pad_h2 as i32,
                        *pad_w1 as i32, *pad_w2 as i32, 8,
                    );
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }

                // --- Activation ---
                Op::Relu => {
                    let k = ReluForward::<f32>::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Elu { .. } => {
                    let k = EluForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Selu => {
                    let k = SeluForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Celu { .. } => {
                    let k = CeluForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Gelu => {
                    let k = GeluForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Mish => {
                    let k = MishForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Hardtanh { .. } => {
                    let k = HardtanhForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Relu6 => {
                    let k = Relu6Forward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Hardsigmoid => {
                    let k = HardsigmoidForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Hardswish => {
                    let k = HardswishForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Hardshrink { .. } => {
                    let k = HardshrinkForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::LeakyRelu { .. } => {
                    let k = LeakyReluForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Threshold { .. } => {
                    let k = ThresholdForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Softsign => {
                    let k = SoftsignForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Softshrink { .. } => {
                    let k = SoftshrinkForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Softplus { .. } => {
                    let k = SoftplusForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Sigmoid => {
                    let k = SigmoidForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Silu => {
                    let k = SiluForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Logsigmoid => {
                    let k = LogsigmoidForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Tanh => {
                    let k = TanhForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Tanhshrink => {
                    let k = TanhshrinkForward::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
                }
                Op::Softmax { .. } => {
                    let k = SoftmaxForward::<f32>::new(1024);
                    Box::new(KernelExecutable {
                        kernel_source: k.kernel_source,
                        entry_point: k.entry_point,
                        shape: node.shape.clone(),
                        dtype: node.dtype,
                    })
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
