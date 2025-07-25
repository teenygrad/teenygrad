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

use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::Result;
use crate::graph::Node;
use crate::graph::NodeOp;
use crate::graph::ops::dot::DotOp;
use crate::graph::ops::powi::Powi;
use crate::graph::ops::sqrt::SqrtOp;
use crate::graph::ops::transpose::TransposeOp;
use crate::graph::scalar;
use crate::tensor::shape::DynamicShape;

pub mod add;
pub mod div;
pub mod mul;
pub mod sub;

#[derive(Debug, Clone)]
pub struct NodeRef<'data, N: Dtype>(pub Arc<Node<'data, N>>);

impl<'data, N: Dtype> NodeRef<'data, N> {
    pub fn realize(&self) -> Result<Vec<N>> {
        todo!()
    }

    pub fn t(&self) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Transpose(TransposeOp::new(self.clone())),
            true,
            false,
        )))
    }

    pub fn shape(&self) -> Result<DynamicShape> {
        self.0.shape()
    }

    pub fn powi(&self, exp: N) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Powi(Powi::new(self.clone(), exp)),
            true,
            false,
        )))
    }

    pub fn sqrt(&self) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Sqrt(SqrtOp::new(self.clone())),
            true,
            false,
        )))
    }

    pub fn dot(&self, other: &NodeRef<N>) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Dot(DotOp::new(self.clone(), other.clone())),
            true,
            false,
        )))
    }
}

impl<'data, N: Dtype> From<NodeOp<'data, N>> for NodeRef<'data, N> {
    fn from(op: NodeOp<'data, N>) -> Self {
        NodeRef(Arc::new(Node::new(op, true, false)))
    }
}

impl<'data, N: Dtype> From<f32> for NodeRef<'data, N> {
    fn from(value: f32) -> Self {
        scalar(N::from_f32(value))
    }
}
