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
use std::ops::Add;
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::graph::ops::add::AddOp;
use crate::graph::{Node, NodeOp, NodeRef};

impl<N: Dtype> Add<&NodeRef<N>> for &NodeRef<N> {
    type Output = NodeRef<N>;

    fn add(self, rhs: &NodeRef<N>) -> Self::Output {
        let lhs = NodeRef(self.0.clone());
        let rhs = NodeRef(rhs.0.clone());

        NodeRef(Arc::new(Node::new(
            NodeOp::Add(AddOp::new(lhs, rhs)),
            true,
            false,
        )))
    }
}

impl<N: Dtype> Add<NodeRef<N>> for NodeRef<N> {
    type Output = NodeRef<N>;

    fn add(self, rhs: NodeRef<N>) -> Self::Output {
        let lhs = NodeRef(self.0);
        let rhs = NodeRef(rhs.0);

        NodeRef(Arc::new(Node::new(
            NodeOp::Add(AddOp::new(lhs, rhs)),
            true,
            false,
        )))
    }
}

impl<N: Dtype> Add<&NodeRef<N>> for NodeRef<N> {
    type Output = NodeRef<N>;

    fn add(self, rhs: &NodeRef<N>) -> Self::Output {
        let lhs = NodeRef(self.0);
        let rhs = NodeRef(rhs.0.clone());

        NodeRef(Arc::new(Node::new(
            NodeOp::Add(AddOp::new(lhs, rhs)),
            true,
            false,
        )))
    }
}

impl<N: Dtype> Add<NodeRef<N>> for &NodeRef<N> {
    type Output = NodeRef<N>;

    fn add(self, rhs: NodeRef<N>) -> Self::Output {
        let lhs = NodeRef(self.0.clone());
        let rhs = NodeRef(rhs.0);

        NodeRef(Arc::new(Node::new(
            NodeOp::Add(AddOp::new(lhs, rhs)),
            true,
            false,
        )))
    }
}
