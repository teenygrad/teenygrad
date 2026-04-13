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
        linear::Linear,
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
        let node_id = self.graph.borrow_mut().add_node(
            op,
            vec![self.node_id],
            self.dtype,
            self.rank,
        );
        Self { node_id, graph: self.graph.clone(), dtype: self.dtype, rank: self.rank }
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
        nn::{activation::{relu::Relu, softmax::Softmax}, linear::Linear},
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
}
