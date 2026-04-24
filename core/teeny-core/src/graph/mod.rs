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

pub mod compiler;

// ---------------------------------------------------------------------------
// Shape — dynamic tensor shape used throughout the graph IR
// ---------------------------------------------------------------------------

/// A dynamic shape vector. Each element is either a known size (`Some(n)`) or a
/// dynamic/unknown dimension (`None`), e.g. a batch axis whose size is determined
/// at runtime.
pub type Shape = Vec<Option<usize>>;

// ---------------------------------------------------------------------------
// Runtime dtype tag — used in the graph since D is erased at the node level
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DtypeRepr {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

// ---------------------------------------------------------------------------
// Graph IR
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Op {
    /// Model input placeholder.
    Input,
    Linear {
        in_features: usize,
        out_features: usize,
        has_bias: bool,
    },
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
    AvgPool2d {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    /// Collapse spatial dims `[N, C, H, W]` → `[N, C*H*W]`.
    Flatten,
    Relu,
    Softmax {
        dim: usize,
    },
}

#[derive(Debug)]
pub struct GraphNode {
    pub op: Op,
    /// Indices of producer nodes in `Graph::nodes`; empty for `Input`.
    pub inputs: Vec<usize>,
    pub dtype: DtypeRepr,
    /// Output shape of this node. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
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
        shape: Shape,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            op,
            inputs,
            dtype,
            shape,
        });
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
    /// Output shape of this tensor. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
}

// ---------------------------------------------------------------------------
// Shape inference — computes the output shape for each Op given an input shape
// ---------------------------------------------------------------------------

fn infer_output_shape(op: &Op, input: &Shape) -> Shape {
    match op {
        Op::Input => input.clone(),
        Op::Relu | Op::Softmax { .. } => input.clone(),

        Op::Linear { out_features, .. } => {
            // [..., in_features] → [..., out_features]
            let mut out = input[..input.len() - 1].to_vec();
            out.push(Some(*out_features));
            out
        }

        Op::Conv2d {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ..
        } => {
            // [N, C_in, H, W] → [N, C_out, H_out, W_out]
            let h_out = input[2].map(|h| (h + 2 * padding_h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w + 2 * padding_w - kernel_w) / stride_w + 1);
            vec![input[0], Some(*out_channels), h_out, w_out]
        }

        Op::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        } => {
            // [N, C, H, W] → [N, C, H_out, W_out]
            let h_out = input[2].map(|h| (h - kernel_h) / stride_h + 1);
            let w_out = input[3].map(|w| (w - kernel_w) / stride_w + 1);
            vec![input[0], input[1], h_out, w_out]
        }

        Op::Flatten => {
            // [N, C, H, W, ...] → [N, C*H*W*...]
            // The flattened dimension is known only when every dim after the
            // first is known; otherwise it stays dynamic.
            let rest = &input[1..];
            let flat: Option<usize> = rest
                .iter()
                .try_fold(1usize, |acc, dim| dim.map(|d| acc * d));
            vec![input[0], flat]
        }
    }
}

// ---------------------------------------------------------------------------
// SymTensor methods
// ---------------------------------------------------------------------------

impl SymTensor {
    /// Create an input placeholder, returning both the tensor and the shared
    /// graph handle. Keep the graph handle to inspect the result after tracing.
    ///
    /// Use `None` for dynamic dimensions (e.g. the batch axis):
    /// ```ignore
    /// SymTensor::input(DtypeRepr::F32, vec![None, Some(784)])
    /// ```
    pub fn input(dtype: DtypeRepr, shape: Shape) -> (Self, Rc<RefCell<Graph>>) {
        let graph = Rc::new(RefCell::new(Graph::new()));
        let node_id = graph
            .borrow_mut()
            .add_node(Op::Input, vec![], dtype, shape.clone());
        let tensor = Self {
            node_id,
            graph: graph.clone(),
            dtype,
            shape,
        };
        (tensor, graph)
    }

    /// Number of dimensions of this tensor.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    fn record(&self, op: Op) -> Self {
        let output_shape = infer_output_shape(&op, &self.shape);
        self.record_with_shape(op, output_shape)
    }

    fn record_with_shape(&self, op: Op, shape: Shape) -> Self {
        let node_id =
            self.graph
                .borrow_mut()
                .add_node(op, vec![self.node_id], self.dtype, shape.clone());
        Self {
            node_id,
            graph: self.graph.clone(),
            dtype: self.dtype,
            shape,
        }
    }
}

// SymTensor satisfies Tensor<D, RANK> for any D and RANK — shape is tracked
// dynamically at runtime; the compile-time SHAPE constant is zeroed (unused).
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
        input.record(Op::Flatten)
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
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);

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
        assert_eq!(g.nodes[0].shape, vec![None, Some(784)]);

        assert!(matches!(
            g.nodes[1].op,
            Op::Linear {
                in_features: 784,
                out_features: 128,
                ..
            }
        ));
        assert_eq!(g.nodes[1].shape, vec![None, Some(128)]);

        assert!(matches!(g.nodes[2].op, Op::Relu));
        assert_eq!(g.nodes[2].shape, vec![None, Some(128)]);

        assert!(matches!(
            g.nodes[3].op,
            Op::Linear {
                in_features: 128,
                out_features: 10,
                ..
            }
        ));
        assert_eq!(g.nodes[3].shape, vec![None, Some(10)]);

        assert!(matches!(g.nodes[4].op, Op::Softmax { dim: 1 }));
        assert_eq!(g.nodes[4].shape, vec![None, Some(10)]);
    }

    #[test]
    fn test_topological_sort_linear_chain() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);

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
                assert!(
                    input_pos < pos,
                    "producer {input_id} must come before consumer {id}"
                );
            }
        }
    }

    /// Residual connection: two branches from the same input converge at an add.
    /// Demonstrates that SymTensor cloning naturally represents forks in the graph.
    #[test]
    fn test_residual_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(64)]);

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
        // Input: [N, 3, 32, 32]
        let (input, graph) =
            SymTensor::input(DtypeRepr::F32, vec![None, Some(3), Some(32), Some(32)]);

        let conv = Conv2d::<f32, SymTensor, SymTensor, 4>::new(
            3,      // in_channels
            64,     // out_channels
            (3, 3), // kernel_size
            (1, 1), // stride
            (1, 1), // padding
            true,   // has_bias
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
        // (32 + 2*1 - 3) / 1 + 1 = 32 — same-padding preserves spatial dims
        assert_eq!(g.nodes[1].shape, vec![None, Some(64), Some(32), Some(32)]);
    }

    #[test]
    fn test_lenet5_shapes() {
        // Trace the LeNet-5 forward pass and verify shapes at every node.
        // Input: [N, 1, 28, 28]
        let (input, graph) =
            SymTensor::input(DtypeRepr::F32, vec![None, Some(1), Some(28), Some(28)]);

        use crate::{
            nn::{flatten::Flatten, pool::AvgPool2d},
            sequential,
        };

        let model = sequential![
            // Block 1
            Conv2d::<f32, SymTensor, SymTensor, 4>::new(1, 6, (5, 5), (1, 1), (2, 2), true),
            Relu::<f32, SymTensor, 4>::new(),
            AvgPool2d::<f32, SymTensor, SymTensor, 4>::new((2, 2), (2, 2)),
            // Block 2
            Conv2d::<f32, SymTensor, SymTensor, 4>::new(6, 16, (5, 5), (1, 1), (0, 0), true),
            Relu::<f32, SymTensor, 4>::new(),
            AvgPool2d::<f32, SymTensor, SymTensor, 4>::new((2, 2), (2, 2)),
            // Flatten + classifier
            Flatten::<f32, SymTensor, SymTensor>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(400, 120, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(120, 84, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(84, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input);

        let g = graph.borrow();
        // 1 input + 13 ops = 14 nodes
        assert_eq!(g.nodes.len(), 14);

        // node 0: Input  [N, 1, 28, 28]
        assert_eq!(g.nodes[0].shape, vec![None, Some(1), Some(28), Some(28)]);
        // node 1: Conv2d(1→6, 5×5, pad=2) → [N, 6, 28, 28]
        assert_eq!(g.nodes[1].shape, vec![None, Some(6), Some(28), Some(28)]);
        // node 2: Relu → [N, 6, 28, 28]
        assert_eq!(g.nodes[2].shape, vec![None, Some(6), Some(28), Some(28)]);
        // node 3: AvgPool2d(2×2, stride=2) → [N, 6, 14, 14]
        assert_eq!(g.nodes[3].shape, vec![None, Some(6), Some(14), Some(14)]);
        // node 4: Conv2d(6→16, 5×5, pad=0) → [N, 16, 10, 10]
        assert_eq!(g.nodes[4].shape, vec![None, Some(16), Some(10), Some(10)]);
        // node 5: Relu → [N, 16, 10, 10]
        assert_eq!(g.nodes[5].shape, vec![None, Some(16), Some(10), Some(10)]);
        // node 6: AvgPool2d(2×2, stride=2) → [N, 16, 5, 5]
        assert_eq!(g.nodes[6].shape, vec![None, Some(16), Some(5), Some(5)]);
        // node 7: Flatten → [N, 400]
        assert_eq!(g.nodes[7].shape, vec![None, Some(400)]);
        // node 8: Linear(400→120) → [N, 120]
        assert_eq!(g.nodes[8].shape, vec![None, Some(120)]);
        // node 9: Relu → [N, 120]
        assert_eq!(g.nodes[9].shape, vec![None, Some(120)]);
        // node 10: Linear(120→84) → [N, 84]
        assert_eq!(g.nodes[10].shape, vec![None, Some(84)]);
        // node 11: Relu → [N, 84]
        assert_eq!(g.nodes[11].shape, vec![None, Some(84)]);
        // node 12: Linear(84→10) → [N, 10]
        assert_eq!(g.nodes[12].shape, vec![None, Some(10)]);
        // node 13: Softmax(dim=1) → [N, 10]
        assert_eq!(g.nodes[13].shape, vec![None, Some(10)]);
    }
}
