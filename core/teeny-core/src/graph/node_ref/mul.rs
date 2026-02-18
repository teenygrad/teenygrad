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

use std::ops::Mul;
use std::sync::Arc;

use crate::graph::ops::mult::MultOp;
use crate::graph::{Node, NodeOp, NodeRef};

impl<'data> Mul<NodeRef<'data>> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn mul(self, rhs: NodeRef<'data>) -> Self::Output {
        let lhs = NodeRef(self.0);
        let rhs = NodeRef(rhs.0);

        NodeRef(Arc::new(Node::new(
            NodeOp::Mult(MultOp::new(lhs, rhs)),
            true,
            false,
        )))
    }
}

impl<'data> Mul<&NodeRef<'data>> for &NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn mul(self, rhs: &NodeRef<'data>) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<'data> Mul<&NodeRef<'data>> for NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn mul(self, rhs: &NodeRef<'data>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<'data> Mul<NodeRef<'data>> for &NodeRef<'data> {
    type Output = NodeRef<'data>;

    fn mul(self, rhs: NodeRef<'data>) -> Self::Output {
        self.clone() * rhs
    }
}
