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

pub mod add;
pub mod mul;
pub mod sub;

use std::sync::Arc;

use crate::dtype::Dtype;
use crate::graph::Node;
use crate::graph::NodeOp;
use crate::graph::ops::transpose::TransposeOp;
use crate::tensor::shape::DynamicShape;

#[derive(Debug, Clone)]
pub struct NodeRef<N: Dtype>(pub Arc<Node<N>>);

impl<N: Dtype> NodeRef<N> {
    pub fn t(&self) -> Self {
        NodeRef(Arc::new(Node::new(
            NodeOp::Transpose(TransposeOp::new(self.clone())),
            true,
            false,
        )))
    }

    pub fn shape(&self) -> DynamicShape {
        self.0.shape()
    }
}

impl<N: Dtype> From<NodeOp<N>> for NodeRef<N> {
    fn from(op: NodeOp<N>) -> Self {
        NodeRef(Arc::new(Node::new(op, true, false)))
    }
}
