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
use crate::error::Error;
use crate::error::Result;
use crate::graph::ops::OpShape;
use crate::graph::{NodeOp, NodeRef};
use crate::tensor::shape::{DynamicShape, Shape};

#[derive(Debug, Clone)]
pub struct MultOp<'data, N: Dtype> {
    pub lhs: NodeRef<'data, N>,
    pub rhs: NodeRef<'data, N>,
}

impl<'data, N: Dtype> MultOp<'data, N> {
    pub fn new(lhs: NodeRef<'data, N>, rhs: NodeRef<'data, N>) -> Self {
        Self { lhs, rhs }
    }
}

impl<'data, N: Dtype> OpShape for MultOp<'data, N> {
    fn shape(&self) -> Result<DynamicShape> {
        let lhs_shape = self.lhs.shape()?;
        let rhs_shape = self.rhs.shape()?;

        match (lhs_shape.dims.len(), rhs_shape.dims.len()) {
            (2, 2) => {
                if lhs_shape.dims[1] == rhs_shape.dims[0] {
                    Ok(DynamicShape::new(&[lhs_shape.dims[0], rhs_shape.dims[1]]))
                } else {
                    Ok(lhs_shape.broadcast(&rhs_shape))
                }
            }
            _ => Err(Error::InvalidShape(format!(
                "Invalid shape for mult: {lhs_shape:?}, {rhs_shape:?}"
            ))
            .into()),
        }
    }
}

impl<'data, N: Dtype> From<MultOp<'data, N>> for NodeRef<'data, N> {
    fn from(op: MultOp<'data, N>) -> Self {
        NodeOp::Mult(op).into()
    }
}
