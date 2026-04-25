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

use crate::errors::Result;
use teeny_core::{
    graph::{Graph, Op},
    model::{ExecutableOp, Lowering},
    utils::dag::Dag,
};

#[derive(Debug, Default)]
pub struct TritonLowering {}

impl TritonLowering {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<'a> Lowering<'a> for TritonLowering {
    fn lower(&self, graph: &Graph) -> Result<Dag<Box<&'static dyn ExecutableOp>>> {
        let node_indexes = graph.topological_sort();
        let mut dag = Dag::new();

        for node_index in node_indexes {
            let node = &graph.nodes[node_index];
            match node.op {
                Op::Input => todo!(),

                // Linear / MLP
                Op::Linear { .. } => todo!(),
                Op::Flatten => todo!(),

                // Convolution
                Op::Conv1d { .. } => todo!(),
                Op::Conv2d { .. } => todo!(),
                Op::Conv3d { .. } => todo!(),

                // Pooling
                Op::AvgPool1d { .. } => todo!(),
                Op::AvgPool2d { .. } => todo!(),
                Op::AvgPool3d { .. } => todo!(),
                Op::MaxPool1d { .. } => todo!(),
                Op::MaxPool2d { .. } => todo!(),
                Op::MaxPool3d { .. } => todo!(),
                Op::LpPool1d { .. } => todo!(),
                Op::LpPool2d { .. } => todo!(),
                Op::LpPool3d { .. } => todo!(),

                // Padding
                Op::ConstantPad1d { .. } => todo!(),
                Op::ConstantPad2d { .. } => todo!(),
                Op::ConstantPad3d { .. } => todo!(),
                Op::ReflectionPad1d { .. } => todo!(),
                Op::ReflectionPad2d { .. } => todo!(),
                Op::ReflectionPad3d { .. } => todo!(),
                Op::ReplicationPad1d { .. } => todo!(),
                Op::ReplicationPad2d { .. } => todo!(),
                Op::ReplicationPad3d { .. } => todo!(),
                Op::CircularPad1d { .. } => todo!(),
                Op::CircularPad2d { .. } => todo!(),
                Op::CircularPad3d { .. } => todo!(),

                // Activation
                Op::Relu => todo!(),
                Op::Elu { .. } => todo!(),
                Op::Selu => todo!(),
                Op::Celu { .. } => todo!(),
                Op::Gelu => todo!(),
                Op::Mish => todo!(),
                Op::Hardtanh { .. } => todo!(),
                Op::Relu6 => todo!(),
                Op::Hardsigmoid => todo!(),
                Op::Hardswish => todo!(),
                Op::Hardshrink { .. } => todo!(),
                Op::LeakyRelu { .. } => todo!(),
                Op::Threshold { .. } => todo!(),
                Op::Softsign => todo!(),
                Op::Softshrink { .. } => todo!(),
                Op::Softplus { .. } => todo!(),
                Op::Sigmoid => todo!(),
                Op::Silu => todo!(),
                Op::Logsigmoid => todo!(),
                Op::Tanh => todo!(),
                Op::Tanhshrink => todo!(),
                Op::Softmax { .. } => todo!(),
            }
        }

        Ok(dag)
    }
}
