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

use std::ops::BitAnd;
use std::ops::BitOr;
use std::sync::Arc;

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::graph::Node;
use crate::graph::NodeOp;
use crate::graph::ops::Op;
use crate::graph::ops::and::AndOp;
use crate::graph::ops::cumsum::CumSumOp;
use crate::graph::ops::diff::DiffOp;
use crate::graph::ops::dot::DotOp;
use crate::graph::ops::eq::EqOp;
use crate::graph::ops::expand::ExpandOp;
use crate::graph::ops::index::IndexOp;
use crate::graph::ops::leq::LeqOp;
use crate::graph::ops::neq::NotEqOp;
use crate::graph::ops::or::OrOp;
use crate::graph::ops::pad::PadOp;
use crate::graph::ops::powi::Powi;
use crate::graph::ops::slice::SliceOp;
use crate::graph::ops::slice::TensorIndex;
use crate::graph::ops::to_dtype::ToDtype;
use crate::graph::ops::transpose::TransposeOp;
use crate::graph::ops::unsqueeze::UnsqueezeOp;
use crate::graph::scalar;
use crate::graph::shape::DynamicShape;
use crate::value::Value;

pub mod add;
pub mod div;
pub mod mul;
pub mod sub;

#[derive(Debug, Clone)]
pub struct NodeRef<'data>(pub Arc<Node<'data>>);

impl<'data> NodeRef<'data> {
    pub fn realize(&self) -> Result<Vec<Value>> {
        todo!()
    }

    pub fn t(&self) -> Self {
        NodeOp::Transpose(TransposeOp::new(self.clone(), &[1, 0])).into()
    }

    pub fn transpose(self, dims: &[isize]) -> Self {
        NodeOp::Transpose(TransposeOp::new(self, dims)).into()
    }

    pub fn to_dtype(self, dtype: DtypeEnum) -> Self {
        NodeOp::ToDtype(ToDtype::new(self, dtype)).into()
    }

    pub fn unsqueeze(self, dim: isize) -> Self {
        NodeOp::Unsqueeze(UnsqueezeOp::new(self, dim)).into()
    }

    pub fn expand(self, dims: &[isize]) -> Self {
        NodeOp::Expand(ExpandOp::new(self, dims)).into()
    }

    pub fn powi(&self, exp: Value) -> Self {
        NodeOp::Powi(Powi::new(self.clone(), exp)).into()
    }

    pub fn dot(&self, other: &NodeRef<'data>) -> Self {
        NodeOp::Dot(DotOp::new(self.clone(), other.clone())).into()
    }

    pub fn slice(&self, indices: &[TensorIndex<'data>]) -> Self {
        NodeOp::Slice(SliceOp::new(self, indices.to_vec())).into()
    }

    pub fn diff(&self, dummy_value: &NodeRef<'data>, dim: isize) -> Self {
        NodeOp::Diff(DiffOp::new(self.clone(), dummy_value.clone(), dim)).into()
    }

    pub fn cumsum(&self, dim: isize) -> Self {
        NodeOp::CumSum(CumSumOp::new(self.clone(), dim)).into()
    }

    pub fn neq(&self, other: &NodeRef<'data>) -> Self {
        NodeOp::NotEq(NotEqOp::new(self.clone(), other.clone())).into()
    }

    pub fn eq(&self, other: &NodeRef<'data>) -> Self {
        NodeOp::Eq(EqOp::new(self.clone(), other.clone())).into()
    }

    pub fn leq(&self, other: &NodeRef<'data>) -> Self {
        NodeOp::Leq(LeqOp::new(self.clone(), other.clone())).into()
    }

    pub fn index<I: IntoIterator<Item = NodeRef<'data>>>(&self, indices: I) -> Self {
        NodeOp::Index(IndexOp::new(self.clone(), indices.into_iter().collect())).into()
    }

    pub fn pad(&self, pad: &[usize]) -> Self {
        NodeOp::Pad(PadOp::new(self.clone(), pad.to_vec())).into()
    }
}

impl<'data> From<NodeOp<'data>> for NodeRef<'data> {
    fn from(op: NodeOp<'data>) -> Self {
        NodeRef(Arc::new(Node::new(op, true, false)))
    }
}

impl<'data> From<f32> for NodeRef<'data> {
    fn from(value: f32) -> Self {
        scalar(value)
    }
}

impl<'data> Op for NodeRef<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        self.0.op.shape()
    }

    fn dtype(&self) -> DtypeEnum {
        self.0.op.dtype()
    }
}

impl<'data> BitAnd<NodeRef<'data>> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn bitand(self, rhs: NodeRef<'data>) -> Self::Output {
        NodeOp::And(AndOp::new(self, rhs)).into()
    }
}

impl<'data> BitOr<NodeRef<'data>> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn bitor(self, rhs: NodeRef<'data>) -> Self::Output {
        NodeOp::Or(OrOp::new(self, rhs)).into()
    }
}
