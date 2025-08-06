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

use std::ops::{Neg, Sub};
use std::sync::Arc;

use crate::graph::ops::neg::NegOp;
use crate::graph::ops::sub::SubOp;
use crate::graph::{Node, NodeOp, NodeRef};

impl<'data> Sub<NodeRef<'data>> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn sub(self, rhs: NodeRef<'data>) -> Self::Output {
        let lhs = NodeRef(self.0);
        let rhs = NodeRef(rhs.0);

        NodeRef(Arc::new(Node::new(
            NodeOp::Sub(SubOp::new(lhs, rhs)),
            true,
            false,
        )))
    }
}

impl<'data> Sub<&NodeRef<'data>> for &NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn sub(self, rhs: &NodeRef<'data>) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<'data> Sub<&NodeRef<'data>> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn sub(self, rhs: &NodeRef<'data>) -> Self::Output {
        self - rhs.clone()
    }
}

impl<'data> Sub<NodeRef<'data>> for &NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn sub(self, rhs: NodeRef<'data>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'data> Neg for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn neg(self) -> Self::Output {
        let lhs = NodeRef(self.0);

        NodeRef(Arc::new(Node::new(
            NodeOp::Neg(NegOp::new(lhs)),
            true,
            false,
        )))
    }
}

impl<'data> Neg for &NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}
