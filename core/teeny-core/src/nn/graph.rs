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

use alloc::{rc::Rc, vec, vec::Vec};
use core::cell::RefCell;

use crate::{
    dtype::{Dtype, Float, RankedTensor, Tensor},
    nn::{
        Layer,
        activation::{relu::Relu, softmax::Softmax},
        conv2d::Conv2d,
        flatten::Flatten,
        linear::Linear,
        pool::AvgPool2d,
    },
};

// ---------------------------------------------------------------------------
// Runtime dtype tag — used in the graph since D is erased at the node level
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DtypeRepr {
    Bool,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, BF16, F32, F64,
}

// ---------------------------------------------------------------------------
// Graph IR
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Op {
    /// Model input placeholder.
    Input,
    Linear { in_features: usize, out_features: usize, has_bias: bool },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        has_bias: bool,
    },
    AvgPool2d { kernel_h: usize, kernel_w: usize, stride_h: usize, stride_w: usize },
    /// Collapse spatial dims `[N, C, H, W]` → `[N, C*H*W]`.
    Flatten,
    Relu,
    Softmax { dim: usize },
}

#[derive(Debug)]
pub struct GraphNode {
    pub op: Op,
    /// Indices of producer nodes in `Graph::nodes`; empty for `Input`.
    pub inputs: Vec<usize>,
    pub dtype: DtypeRepr,
    pub rank: usize,
}

#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<usize>,
        dtype: DtypeRepr,
        rank: usize,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GraphNode { op, inputs, dtype, rank });
        id
    }

    /// Returns node indices in topological order (producers before consumers)
    /// using Kahn's algorithm. Panics if the graph contains a cycle.
    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];

        for (id, node) in self.nodes.iter().enumerate() {
            for &input in &node.inputs {
                in_degree[id] += 1;
                dependents[input].push(id);
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(id) = queue.pop() {
            order.push(id);
            for &dep in &dependents[id] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push(dep);
                }
            }
        }

        assert_eq!(order.len(), n, "graph contains a cycle");
        order
    }
}

// ---------------------------------------------------------------------------
// SymTensor — a tensor that writes to the graph on every operation
// ---------------------------------------------------------------------------

/// A symbolic tensor handle. Every layer operation on a `SymTensor` records
/// itself in the shared `Graph` and returns a new `SymTensor` pointing to
/// the new node. Cloning is cheap — it shares the graph via `Rc`.
#[derive(Clone)]
pub struct SymTensor {
    pub node_id: usize,
    pub graph: Rc<RefCell<Graph>>,
    pub dtype: DtypeRepr,
    pub rank: usize,
}

impl SymTensor {
    /// Create an input placeholder, returning both the tensor and the shared
    /// graph handle. Keep the graph handle to inspect the result after tracing.
    pub fn input(dtype: DtypeRepr, rank: usize) -> (Self, Rc<RefCell<Graph>>) {
        let graph = Rc::new(RefCell::new(Graph::new()));
        let node_id = graph.borrow_mut().add_node(Op::Input, vec![], dtype, rank);
        let tensor = Self { node_id, graph: graph.clone(), dtype, rank };
        (tensor, graph)
    }

    fn record(&self, op: Op) -> Self {
        self.record_with_rank(op, self.rank)
    }

    /// Like `record`, but the output node carries a different rank (e.g. after Flatten).
    fn record_with_rank(&self, op: Op, rank: usize) -> Self {
        let node_id = self.graph.borrow_mut().add_node(
            op,
            vec![self.node_id],
            self.dtype,
            rank,
        );
        Self { node_id, graph: self.graph.clone(), dtype: self.dtype, rank }
    }
}

// SymTensor satisfies Tensor<D, RANK> for any D and RANK — shape is dynamic.
impl<D: Dtype, const RANK: usize> RankedTensor<D, RANK> for SymTensor {
    const SHAPE: [usize; RANK] = [0; RANK];
}
impl<D: Dtype, const RANK: usize> Tensor<D, RANK> for SymTensor {}

// ---------------------------------------------------------------------------
// Layer<SymTensor> impls — one per layer type, record op instead of computing
// ---------------------------------------------------------------------------

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Linear<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Linear {
            in_features: self.in_features,
            out_features: self.out_features,
            has_bias: self.has_bias,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Conv2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Conv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            padding_h: self.padding_h,
            padding_w: self.padding_w,
            has_bias: self.has_bias,
        })
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for AvgPool2d<D, SymTensor, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::AvgPool2d {
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
        })
    }
}

impl<D: Dtype> Layer<SymTensor> for Flatten<D, SymTensor, SymTensor> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record_with_rank(Op::Flatten, 2)
    }
}

impl<D: Dtype, const RANK: usize> Layer<SymTensor> for Relu<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Relu)
    }
}

impl<D: Float, const RANK: usize> Layer<SymTensor> for Softmax<D, SymTensor, RANK> {
    type Output = SymTensor;
    fn call(&self, input: SymTensor) -> SymTensor {
        input.record(Op::Softmax { dim: self.dim })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{
            activation::{relu::Relu, softmax::Softmax},
            conv2d::Conv2d,
            linear::Linear,
        },
        sequential,
    };

    #[test]
    fn test_sequential_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, 2);

        let model = sequential![
            Linear::<f32, SymTensor, SymTensor, 2>::new(784, 128, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(128, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input);

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 5);
        assert!(matches!(g.nodes[0].op, Op::Input));
        assert!(matches!(g.nodes[1].op, Op::Linear { in_features: 784, out_features: 128, .. }));
        assert!(matches!(g.nodes[2].op, Op::Relu));
        assert!(matches!(g.nodes[3].op, Op::Linear { in_features: 128, out_features: 10, .. }));
        assert!(matches!(g.nodes[4].op, Op::Softmax { dim: 1 }));
    }

    #[test]
    fn test_topological_sort_linear_chain() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, 2);

        let model = sequential![
            Linear::<f32, SymTensor, SymTensor, 2>::new(784, 128, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(128, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input);

        let g = graph.borrow();
        let order = g.topological_sort();
        // For a linear chain every node must appear before any node that consumes it.
        assert_eq!(order.len(), g.nodes.len());
        for (pos, &id) in order.iter().enumerate() {
            for &input_id in &g.nodes[id].inputs {
                let input_pos = order.iter().position(|&x| x == input_id).unwrap();
                assert!(input_pos < pos, "producer {input_id} must come before consumer {id}");
            }
        }
    }

    /// Residual connection: two branches from the same input converge at an add.
    /// Demonstrates that SymTensor cloning naturally represents forks in the graph.
    #[test]
    fn test_residual_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, 2);

        let main = Linear::<f32, SymTensor, SymTensor, 2>::new(64, 64, true).call(input.clone());
        let main = Relu::<f32, SymTensor, 2>::new().call(main);

        // Skip connection re-uses the original input node (node_id 0)
        let skip = Linear::<f32, SymTensor, SymTensor, 2>::new(64, 64, false).call(input);

        // In a real impl this would be an Add op; we just verify both branches
        // share the same graph and the skip connection references the input.
        assert!(Rc::ptr_eq(&main.graph, &skip.graph));

        let g = graph.borrow();
        // Input, Linear(main), Relu, Linear(skip)
        assert_eq!(g.nodes.len(), 4);
        // Both the main and skip linear ops take node 0 (Input) as their input
        assert_eq!(g.nodes[1].inputs, vec![0]); // main branch
        assert_eq!(g.nodes[3].inputs, vec![0]); // skip branch
    }

    #[test]
    fn test_conv2d_graph_extraction() {
        // Input: rank-4 tensor [N, C, H, W]
        let (input, graph) = SymTensor::input(DtypeRepr::F32, 4);

        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(
            3,           // in_channels
            64,          // out_channels
            (3, 3),      // kernel_size
            (1, 1),      // stride
            (1, 1),      // padding
            true,        // has_bias
        );
        let _out = Layer::call(&conv, input);

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 2);
        assert!(matches!(g.nodes[0].op, Op::Input));
        assert!(matches!(
            g.nodes[1].op,
            Op::Conv2d {
                in_channels: 3,
                out_channels: 64,
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                padding_h: 1,
                padding_w: 1,
                has_bias: true,
            }
        ));
        assert_eq!(g.nodes[1].inputs, vec![0]);
        assert_eq!(g.nodes[1].rank, 4);
    }
}
