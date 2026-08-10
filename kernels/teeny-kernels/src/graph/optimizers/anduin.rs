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

//! Anduin — first Triton graph optimizer (LotR river names).

use teeny_core::graph::{Graph, GraphNode, Op};

use crate::errors::Result;
use crate::graph::optimizer::GraphOptimizer;

/// Anduin: Triton-side graph rewrites before lowering.
///
/// Currently fuses `Conv2d(no bias) → BatchNorm2d → Silu` into
/// [`Op::Conv2dBnSilu`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Anduin;

impl GraphOptimizer for Anduin {
    fn name(&self) -> &str {
        "anduin"
    }

    fn optimize(&self, graph: &Graph) -> Result<Graph> {
        Ok(fuse_conv_bn_silu(graph))
    }
}

/// Rewrite `Conv2d(no bias) → BatchNorm2d → Silu` into a single [`Op::Conv2dBnSilu`].
fn fuse_conv_bn_silu(graph: &Graph) -> Graph {
    let n = graph.nodes.len();

    let mut n_consumers = vec![0usize; n];
    for node in &graph.nodes {
        for &inp in &node.inputs {
            n_consumers[inp] += 1;
        }
    }

    let mut dead = vec![false; n];
    let mut node_override: Vec<Option<(Op, Vec<usize>)>> = vec![None; n];

    // `silu_idx` indexes multiple parallel collections below.
    #[allow(clippy::needless_range_loop)]
    for silu_idx in 0..n {
        if !matches!(graph.nodes[silu_idx].op, Op::Silu) {
            continue;
        }
        if graph.nodes[silu_idx].inputs.len() != 1 {
            continue;
        }

        let bn_idx = graph.nodes[silu_idx].inputs[0];
        if !matches!(graph.nodes[bn_idx].op, Op::BatchNorm2d { .. }) {
            continue;
        }
        if n_consumers[bn_idx] != 1 {
            continue;
        }
        if graph.nodes[bn_idx].inputs.len() != 1 {
            continue;
        }

        let conv_idx = graph.nodes[bn_idx].inputs[0];
        if !matches!(
            graph.nodes[conv_idx].op,
            Op::Conv2d {
                has_bias: false,
                ..
            }
        ) {
            continue;
        }
        if n_consumers[conv_idx] != 1 {
            continue;
        }

        let (
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            groups,
        ) = if let Op::Conv2d {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            groups,
            ..
        } = graph.nodes[conv_idx].op
        {
            (
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                groups,
            )
        } else {
            unreachable!()
        };

        let bn_eps = if let Op::BatchNorm2d { eps, .. } = graph.nodes[bn_idx].op {
            eps
        } else {
            unreachable!()
        };

        dead[conv_idx] = true;
        dead[bn_idx] = true;
        node_override[silu_idx] = Some((
            Op::Conv2dBnSilu {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                groups,
                bn_eps,
            },
            graph.nodes[conv_idx].inputs.clone(),
        ));
    }

    let mut old_to_new = vec![0usize; n];
    let mut new_count = 0usize;
    for i in 0..n {
        if !dead[i] {
            old_to_new[i] = new_count;
            new_count += 1;
        }
    }

    let mut new_graph = Graph::new();
    for old_idx in 0..n {
        if dead[old_idx] {
            continue;
        }
        let node = &graph.nodes[old_idx];
        let (op, inputs) = if let Some((fused_op, fused_inputs)) = node_override[old_idx].clone() {
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
        if let Some(name) = graph.names.get(&old_idx) {
            new_graph.names.insert(new_idx, name.clone());
        }
    }

    new_graph
}
