/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::ops::Index;
use std::sync::Arc;

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::graph::Node;
use crate::graph::NodeOp;
use crate::graph::ops::Op;
use crate::graph::ops::cumsum::CumSumOp;
use crate::graph::ops::diff::DiffOp;
use crate::graph::ops::dot::DotOp;
use crate::graph::ops::eq::EqOp;
use crate::graph::ops::expand::ExpandOp;
use crate::graph::ops::neq::NotEqOp;
use crate::graph::ops::powi::Powi;
use crate::graph::ops::slice::SliceOp;
use crate::graph::ops::slice::TensorIndex;
use crate::graph::ops::to_dtype::ToDtype;
use crate::graph::ops::transpose::TransposeOp;
use crate::graph::ops::unsqueeze::UnsqueezeOp;
use crate::graph::scalar;
use crate::tensor::shape::DynamicShape;
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
        NodeRef(Arc::new(Node::new(
            NodeOp::Transpose(TransposeOp::new(self.clone(), &[1, 0])),
            true,
            false,
        )))
    }

    pub fn transpose(self, dims: &[isize]) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Transpose(TransposeOp::new(self, dims)),
            true,
            false,
        )))
    }

    pub fn to_dtype(self, dtype: DtypeEnum) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::ToDtype(ToDtype::new(self, dtype)),
            true,
            false,
        )))
    }

    pub fn unsqueeze(self, dim: isize) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Unsqueeze(UnsqueezeOp::new(self, dim)),
            true,
            false,
        )))
    }

    pub fn expand(self, dims: &[isize]) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Expand(ExpandOp::new(self, dims)),
            true,
            false,
        )))
    }

    pub fn powi(&self, exp: Value) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Powi(Powi::new(self.clone(), exp)),
            true,
            false,
        )))
    }

    pub fn dot(&self, other: &NodeRef<'data>) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Dot(DotOp::new(self.clone(), other.clone())),
            true,
            false,
        )))
    }

    pub fn slice(&self, indices: &[TensorIndex]) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Slice(SliceOp::new(self, indices.to_vec())),
            true,
            false,
        )))
    }

    pub fn diff(&self, dummy_value: &NodeRef<'data>, dim: isize) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Diff(DiffOp::new(self.clone(), dummy_value.clone(), dim)),
            true,
            false,
        )))
    }

    pub fn cumsum(&self, dim: isize) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::CumSum(CumSumOp::new(self.clone(), dim)),
            true,
            false,
        )))
    }

    pub fn neq(&self, other: &NodeRef<'data>) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::NotEq(NotEqOp::new(self.clone(), other.clone())),
            true,
            false,
        )))
    }

    pub fn eq(&self, other: &NodeRef<'data>) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Eq(EqOp::new(self.clone(), other.clone())),
            true,
            false,
        )))
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

impl<'data> Index<(usize, usize)> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        todo!()
    }
}
