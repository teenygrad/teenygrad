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

use alloc::{rc::Rc, vec, vec::Vec};
use core::cell::RefCell;

use super::{CustomData, DtypeRepr, Graph, Op, Shape, infer_output_shape};
use crate::dtype::{Dtype, RankedTensor, Tensor};

/// `Layer<SymTensor>` impls: each records its op instead of computing it.
mod layers;

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

    fn record(&self, op: Op) -> Self {
        let output_shape = infer_output_shape(&op, &[&self.shape]);
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

    /// Record a custom op whose output shape is determined by [`CustomOp::infer_output_shape`](super::CustomOp::infer_output_shape).
    ///
    /// `self` is the primary (first) input.  Pass additional inputs via
    /// `other_inputs`.  Pass `dtype` to override the output element type;
    /// defaults to the primary input's dtype.
    pub fn record_custom(
        &self,
        data: CustomData,
        other_inputs: &[&SymTensor],
        dtype: Option<DtypeRepr>,
    ) -> Self {
        let mut shapes: Vec<&Shape> = vec![&self.shape];
        shapes.extend(other_inputs.iter().map(|t| &t.shape));
        let output_shape = data.infer_output_shape(&shapes);

        let mut input_ids: Vec<usize> = vec![self.node_id];
        input_ids.extend(other_inputs.iter().map(|t| t.node_id));

        let out_dtype = dtype.unwrap_or(self.dtype);
        let node_id = self.graph.borrow_mut().add_node(
            Op::Custom { data },
            input_ids,
            out_dtype,
            output_shape.clone(),
        );
        Self {
            node_id,
            graph: self.graph.clone(),
            dtype: out_dtype,
            shape: output_shape,
        }
    }
}
