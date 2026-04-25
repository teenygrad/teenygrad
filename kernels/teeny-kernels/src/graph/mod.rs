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
                Op::Linear {
                    in_features,
                    out_features,
                    has_bias,
                } => todo!(),
                Op::Conv2d {
                    in_channels,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    has_bias,
                } => todo!(),
                Op::AvgPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                } => todo!(),
                Op::Flatten => todo!(),
                Op::Relu => todo!(),
                Op::Softmax { dim } => todo!(),
            }
        }

        Ok(dag)
    }
}
