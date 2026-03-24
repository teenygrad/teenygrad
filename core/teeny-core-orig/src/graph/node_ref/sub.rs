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
