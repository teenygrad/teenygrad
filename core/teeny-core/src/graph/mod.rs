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

use alloc::{collections::BTreeMap, rc::Rc, string::String, sync::Arc, vec, vec::Vec};
use core::cell::RefCell;

use crate::dtype::{Dtype, RankedTensor, Tensor};
use crate::errors::Result;

/// Graph-to-FXGraph lowering.
pub mod compiler;
pub mod layer;
pub mod op;

pub use op::{CustomOp, Op};

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

/// Runtime dtype tag: the graph-level (type-erased) representation of a tensor's dtype,
/// mirroring `dtype::Dtype`'s implementors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DtypeRepr {
    /// `bool`.
    Bool,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// 16-bit float.
    F16,
    /// `bfloat16`.
    BF16,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

// ---------------------------------------------------------------------------
// Graph IR
// ---------------------------------------------------------------------------

/// One node in a [`Graph`]: an [`Op`] plus its producer indices, dtype, and output shape.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// This node's operation.
    pub op: Op,
    /// Indices of producer nodes in `Graph::nodes`; empty for `Input`.
    pub inputs: Vec<usize>,
    /// This node's output dtype.
    pub dtype: DtypeRepr,
    /// Output shape of this node. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
}

/// The traced computational graph: a list of [`GraphNode`]s plus optional node names.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    /// The graph's nodes, in the order they were recorded.
    pub nodes: Vec<GraphNode>,
    /// Node index → dotted name captured from [`crate::name_scope`] at recording time.
    pub names: BTreeMap<usize, String>,
}

impl Graph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a new node to the graph, returning its index.
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
        #[cfg(feature = "std")]
        if let Some(name) = crate::name_scope::current_scope() {
            self.names.insert(id, name);
        }
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
    /// This tensor's node index in `graph`.
    pub node_id: usize,
    /// The shared graph this tensor's operations record into.
    pub graph: Rc<RefCell<Graph>>,
    /// This tensor's dtype.
    pub dtype: DtypeRepr,
    /// Output shape of this tensor. `None` in a slot means a dynamic/unknown
    /// dimension (e.g. the batch axis).
    pub shape: Shape,
}

// SymTensor satisfies Tensor<D, RANK> for any D and RANK — shape is tracked
// dynamically at runtime; the compile-time SHAPE constant is zeroed (unused).
impl<D: Dtype, const RANK: usize> RankedTensor<D, RANK> for SymTensor {
    const SHAPE: [usize; RANK] = [0; RANK];
}
impl<D: Dtype, const RANK: usize> Tensor<D, RANK> for SymTensor {}

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

    /// Records `op` with the shape inferred from this tensor.
    ///
    /// `Layer::call` returns a `SymTensor` rather than a `Result`, so a shape
    /// the op cannot produce — a conv window that does not fit, a zero stride —
    /// panics here with the error's own message. Callers that want to handle
    /// it instead should ask [`Op::infer_output_shape`] directly.
    fn record(&self, op: Op) -> Result<Self> {
        let output_shape = op.infer_output_shape(&[&self.shape])?;
        Ok(self.record_with_shape(op, output_shape))
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

    /// Record a custom op whose output shape is determined by [`CustomOp::infer_output_shape`].
    ///
    /// `self` is the primary (first) input.  Pass additional inputs via
    /// `other_inputs`.  Pass `dtype` to override the output element type;
    /// defaults to the primary input's dtype.
    ///
    /// # Errors
    ///
    /// Returns the op's own error if it rejects these input shapes.
    pub fn record_custom(
        &self,
        data: Arc<dyn CustomOp>,
        other_inputs: &[&SymTensor],
        dtype: Option<DtypeRepr>,
    ) -> Result<Self> {
        let mut shapes: Vec<&Shape> = vec![&self.shape];
        shapes.extend(other_inputs.iter().map(|t| &t.shape));
        let output_shape = data.infer_output_shape(&shapes)?;

        let mut input_ids: Vec<usize> = vec![self.node_id];
        input_ids.extend(other_inputs.iter().map(|t| t.node_id));

        let out_dtype = dtype.unwrap_or(self.dtype);
        let node_id = self.graph.borrow_mut().add_node(
            Op::Custom { data },
            input_ids,
            out_dtype,
            output_shape.clone(),
        );
        Ok(Self {
            node_id,
            graph: self.graph.clone(),
            dtype: out_dtype,
            shape: output_shape,
        })
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
            Layer,
            activation::{relu::Relu, softmax::Softmax},
            linear::Linear,
        },
        sequential,
    };

    #[test]
    fn test_topological_sort_linear_chain() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(784)]);

        let model = sequential![
            Linear::<f32, SymTensor, SymTensor, 2>::new(784, 128, true),
            Relu::<f32, SymTensor, 2>::new(),
            Linear::<f32, SymTensor, SymTensor, 2>::new(128, 10, true),
            Softmax::<f32, SymTensor, 2>::new(1)
        ];

        let _out = Layer::call(&model, input).unwrap();

        let g = graph.borrow();
        let order = g.topological_sort();
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

    #[test]
    fn test_residual_graph_extraction() {
        let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(64)]);

        let main = Linear::<f32, SymTensor, SymTensor, 2>::new(64, 64, true)
            .call(input.clone())
            .unwrap();
        let main = Relu::<f32, SymTensor, 2>::new().call(main).unwrap();
        let skip = Linear::<f32, SymTensor, SymTensor, 2>::new(64, 64, false)
            .call(input)
            .unwrap();

        assert!(Rc::ptr_eq(&main.graph, &skip.graph));

        let g = graph.borrow();
        assert_eq!(g.nodes.len(), 4);
        assert_eq!(g.nodes[1].inputs, vec![0]);
        assert_eq!(g.nodes[3].inputs, vec![0]);
    }
}
