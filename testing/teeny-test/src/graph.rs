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

//! Ready-made [`Graph`]s that more than one test drives through lowering,
//! scheduling or a real device.
//!
//! Where `teeny-core` has an `nn` layer for every op, a graph is built
//! through those layers (`SymTensor` + [`Layer::call`]), the same way a real
//! model traces itself, so every node's output shape is inferred by its op
//! rather than restated by hand. [`DecomposedSoftmax`] has no such layers to
//! trace through, so it adds its nodes directly. Only structural knobs have
//! defaults; the sizes are each test's own, since they usually have to match
//! that test's fixtures.

use std::rc::Rc;

use anyhow::{Result, anyhow};
use derive_builder::Builder;
use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_core::nn::Layer;
use teeny_core::nn::activation::sigmoid::Silu;
use teeny_core::nn::batchnorm::BatchNorm2d;
use teeny_core::nn::conv2d::Conv2d;

/// `input -> conv2d (no bias) -> batchnorm2d -> silu`, all f32 NCHW -- the
/// classic conv block, and the chain `nn::fused::conv2d_bn_silu` fuses by
/// hand.
///
/// ```ignore
/// let block = Conv2dBnSiluBuilder::default()
///     .batch(2)
///     .in_channels(3)
///     .out_channels(8)
///     .height(32)
///     .width(32)
///     .build()?
///     .graph()?;
/// // block.graph, plus block.conv / block.bn to upload their params
/// ```
#[derive(Debug, Clone, Builder)]
pub struct Conv2dBnSilu {
    /// Batch size (`N`).
    pub batch: usize,
    /// Input channels (`C_in`).
    pub in_channels: usize,
    /// Output channels (`C_out`), and so BatchNorm2d's `num_features`.
    pub out_channels: usize,
    /// Input height (`H`).
    pub height: usize,
    /// Input width (`W`).
    pub width: usize,
    /// Conv kernel `(height, width)`.
    #[builder(default = "(3, 3)")]
    pub kernel: (usize, usize),
    /// Conv stride `(height, width)`.
    #[builder(default = "(1, 1)")]
    pub stride: (usize, usize),
    /// Conv zero-padding `(height, width)`. The default `(1, 1)`, with the
    /// default 3x3 kernel and stride 1, keeps the output `H x W`.
    #[builder(default = "(1, 1)")]
    pub padding: (usize, usize),
    /// BatchNorm2d's numerical stability constant (PyTorch's default).
    #[builder(default = "1e-5")]
    pub eps: f64,
}

/// A traced [`Conv2dBnSilu`] graph plus the id of each of its nodes, so a
/// test can find the nodes whose parameters it uploads (`conv`'s weight;
/// `bn`'s weight, bias, running mean and running var).
#[derive(Debug, Clone)]
pub struct Conv2dBnSiluGraph {
    /// The traced graph.
    pub graph: Graph,
    /// The `Input` node.
    pub input: usize,
    /// The `Conv2d` node.
    pub conv: usize,
    /// The `BatchNorm2d` node.
    pub bn: usize,
    /// The `Silu` node -- the graph's output.
    pub silu: usize,
}

impl Conv2dBnSilu {
    /// Traces this block into a fresh [`Graph`].
    ///
    /// # Errors
    ///
    /// Returns an error when a layer rejects its input shape -- e.g. a
    /// kernel larger than the padded input.
    pub fn graph(&self) -> Result<Conv2dBnSiluGraph> {
        let (input, graph) = SymTensor::input(
            DtypeRepr::F32,
            vec![
                Some(self.batch),
                Some(self.in_channels),
                Some(self.height),
                Some(self.width),
            ],
        );
        let input_id = input.node_id;

        let conv = Layer::call(
            &Conv2d::<f32, SymTensor, SymTensor, 4>::new(
                self.in_channels,
                self.out_channels,
                self.kernel,
                self.stride,
                self.padding,
                false,
            ),
            input,
        )?;
        let conv_id = conv.node_id;

        let bn = Layer::call(
            &BatchNorm2d::<f32, SymTensor, SymTensor, 4>::new(self.out_channels)
                .with_eps(self.eps),
            conv,
        )?;
        let bn_id = bn.node_id;

        let silu = Layer::call(&Silu::<f32, SymTensor, 4>::new(), bn)?;
        let silu_id = silu.node_id;

        // Every SymTensor holds an Rc to the graph; drop the last one so the
        // graph can be moved out instead of cloned.
        drop(silu);
        let graph = Rc::try_unwrap(graph)
            .map_err(|_| anyhow!("Conv2dBnSilu: a SymTensor still borrows the traced graph"))?
            .into_inner();

        Ok(Conv2dBnSiluGraph {
            graph,
            input: input_id,
            conv: conv_id,
            bn: bn_id,
            silu: silu_id,
        })
    }
}

/// `softmax(x)` along the last axis of a `[rows, cols]` f32 input, built from
/// primitive ops instead of one `Softmax` node:
///
/// ```text
/// x -> max -> sub(x, max) -> exp -> sum -> div(exp, sum)
/// ```
///
/// `max` and `sum` keep their reduced axis (`[rows, 1]`), and `sub`/`div`
/// broadcast it back over the columns. This is the graph a tile-graph
/// scheduler has to fuse to recover a single softmax kernel.
///
/// ```ignore
/// let softmax = DecomposedSoftmaxBuilder::default()
///     .rows(64)
///     .cols(1024)
///     .build()?
///     .graph();
/// // softmax.graph, plus softmax.max / softmax.sum / ... to find each node
/// ```
#[derive(Debug, Clone, Builder)]
pub struct DecomposedSoftmax {
    /// Rows: the axis softmax is computed independently along.
    pub rows: usize,
    /// Columns: the axis softmax normalises over.
    pub cols: usize,
}

/// A [`DecomposedSoftmax`] graph plus the id of each of its nodes.
#[derive(Debug, Clone)]
pub struct DecomposedSoftmaxGraph {
    /// The graph.
    pub graph: Graph,
    /// The `Input` node, `x`.
    pub input: usize,
    /// The `ReduceMax` node: each row's max.
    pub max: usize,
    /// The `Sub` node: `x - max`.
    pub sub: usize,
    /// The `Exp` node.
    pub exp: usize,
    /// The `ReduceSum` node: each row's sum of `exp`.
    pub sum: usize,
    /// The `Div` node: `exp / sum` -- the graph's output.
    pub div: usize,
}

impl DecomposedSoftmax {
    /// Builds this softmax into a fresh [`Graph`].
    ///
    /// Shapes are given explicitly: `Op::infer_output_shape` has no reduce
    /// axes to go on, so it would give `max` and `sum` the input's full
    /// shape rather than `[rows, 1]`.
    pub fn graph(&self) -> DecomposedSoftmaxGraph {
        let shape = vec![Some(self.rows), Some(self.cols)];
        let reduced_shape = vec![Some(self.rows), Some(1)];
        let reduce = |graph: &mut Graph, op: Op, input: usize| {
            graph.add_node(op, vec![input], DtypeRepr::F32, reduced_shape.clone())
        };

        let mut graph = Graph::new();
        let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        let max = reduce(
            &mut graph,
            Op::ReduceMax {
                keepdims: true,
                noop_with_empty_axes: false,
            },
            input,
        );
        let sub = graph.add_node(Op::Sub, vec![input, max], DtypeRepr::F32, shape.clone());
        let exp = graph.add_node(Op::Exp, vec![sub], DtypeRepr::F32, shape.clone());
        let sum = reduce(
            &mut graph,
            Op::ReduceSum {
                keepdims: true,
                noop_with_empty_axes: false,
            },
            exp,
        );
        let div = graph.add_node(Op::Div, vec![exp, sum], DtypeRepr::F32, shape);

        DecomposedSoftmaxGraph {
            graph,
            input,
            max,
            sub,
            exp,
            sum,
            div,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Conv2dBnSiluBuilder, DecomposedSoftmaxBuilder};
    use teeny_core::graph::Op;

    #[test]
    fn test_conv2d_bn_silu_graph_is_four_nodes_in_chain_order() {
        let block = Conv2dBnSiluBuilder::default()
            .batch(2)
            .in_channels(3)
            .out_channels(8)
            .height(32)
            .width(32)
            .build()
            .expect("every required field is set")
            .graph()
            .expect("a 3x3 same-padded conv fits a 32x32 input");

        let nodes = &block.graph.nodes;
        assert_eq!(nodes.len(), 4);
        assert_eq!(
            (block.input, block.conv, block.bn, block.silu),
            (0, 1, 2, 3)
        );
        assert!(matches!(nodes[block.input].op, Op::Input));
        assert!(matches!(
            nodes[block.conv].op,
            Op::Conv2d {
                in_channels: 3,
                out_channels: 8,
                has_bias: false,
                ..
            }
        ));
        assert!(matches!(
            nodes[block.bn].op,
            Op::BatchNorm2d { num_features: 8, eps, .. } if eps == 1e-5
        ));
        assert!(matches!(nodes[block.silu].op, Op::Silu));
        assert_eq!(nodes[block.conv].inputs, vec![block.input]);
        assert_eq!(nodes[block.bn].inputs, vec![block.conv]);
        assert_eq!(nodes[block.silu].inputs, vec![block.bn]);

        // Same padding: every node after the conv is [N, C_out, H, W].
        let out = vec![Some(2), Some(8), Some(32), Some(32)];
        for id in [block.conv, block.bn, block.silu] {
            assert_eq!(nodes[id].shape, out, "node {id}");
        }
    }

    #[test]
    fn test_conv2d_bn_silu_builder_rejects_a_missing_size() {
        let err = Conv2dBnSiluBuilder::default()
            .batch(2)
            .in_channels(3)
            .out_channels(8)
            .height(32)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("width"), "unexpected error: {err}");
    }

    #[test]
    fn test_decomposed_softmax_graph_wires_max_sub_exp_sum_div() {
        let softmax = DecomposedSoftmaxBuilder::default()
            .rows(64)
            .cols(1024)
            .build()
            .expect("every required field is set")
            .graph();

        let nodes = &softmax.graph.nodes;
        assert_eq!(nodes.len(), 6);
        assert!(matches!(nodes[softmax.input].op, Op::Input));
        assert!(matches!(
            nodes[softmax.max].op,
            Op::ReduceMax { keepdims: true, .. }
        ));
        assert!(matches!(nodes[softmax.sub].op, Op::Sub));
        assert!(matches!(nodes[softmax.exp].op, Op::Exp));
        assert!(matches!(
            nodes[softmax.sum].op,
            Op::ReduceSum { keepdims: true, .. }
        ));
        assert!(matches!(nodes[softmax.div].op, Op::Div));

        // `x` and `exp` each feed a reduction and the op that consumes it.
        assert_eq!(nodes[softmax.max].inputs, vec![softmax.input]);
        assert_eq!(nodes[softmax.sub].inputs, vec![softmax.input, softmax.max]);
        assert_eq!(nodes[softmax.exp].inputs, vec![softmax.sub]);
        assert_eq!(nodes[softmax.sum].inputs, vec![softmax.exp]);
        assert_eq!(nodes[softmax.div].inputs, vec![softmax.exp, softmax.sum]);

        let full = vec![Some(64), Some(1024)];
        let reduced = vec![Some(64), Some(1)];
        for id in [softmax.input, softmax.sub, softmax.exp, softmax.div] {
            assert_eq!(nodes[id].shape, full, "node {id}");
        }
        for id in [softmax.max, softmax.sum] {
            assert_eq!(nodes[id].shape, reduced, "node {id}");
        }
    }
}
