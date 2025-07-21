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

use crate::dtype::Dtype;
use crate::error::Result;
use crate::graph::ops::OpShape;
use crate::graph::{NodeOp, NodeRef};
use crate::tensor::shape::{DynamicShape, Shape};

#[derive(Debug, Clone)]
pub struct DotOp<N: Dtype> {
    pub lhs: NodeRef<N>,
    pub rhs: NodeRef<N>,
}

impl<N: Dtype> DotOp<N> {
    pub fn new(lhs: NodeRef<N>, rhs: NodeRef<N>) -> Self {
        Self { lhs, rhs }
    }
}

impl<N: Dtype> OpShape for DotOp<N> {
    fn shape(&self) -> Result<DynamicShape> {
        let lhs_shape = self.lhs.shape()?;
        let rhs_shape = self.rhs.shape()?;
        Ok(lhs_shape.broadcast(&rhs_shape))
    }
}

impl<N: Dtype> From<DotOp<N>> for NodeRef<N> {
    fn from(op: DotOp<N>) -> Self {
        NodeOp::Dot(op).into()
    }
}
