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
//! Built through `teeny-core`'s `nn` layers (`SymTensor` + [`Layer::call`]),
//! the same way a real model traces itself, so every node's output shape is
//! inferred by its op rather than restated by hand. Only structural knobs
//! have defaults; the sizes are each test's own, since they usually have to
//! match that test's fixtures.

use std::rc::Rc;

use anyhow::{Result, anyhow};
use derive_builder::Builder;
use teeny_core::graph::{DtypeRepr, Graph, SymTensor};
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

#[cfg(test)]
mod tests {
    use super::Conv2dBnSiluBuilder;
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
}
