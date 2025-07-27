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

use crate::error::Result;
use crate::{
    dtype::Dtype,
    graph::{NodeOp, NodeRef, ops::OpShape},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct MeanOp<'data, N: Dtype> {
    pub input: NodeRef<'data, N>,
    pub dim: Option<usize>,
}

impl<'data, N: Dtype> MeanOp<'data, N> {
    pub fn new(input: NodeRef<'data, N>, dim: Option<usize>) -> Self {
        Self { input, dim }
    }
}

impl<'data, N: Dtype> OpShape for MeanOp<'data, N> {
    fn shape(&self) -> Result<DynamicShape> {
        let input_shape = self.input.shape()?;

        match self.dim {
            Some(dim) => {
                // Validate dimension index
                if dim >= input_shape.dims.len() {
                    panic!(
                        "Dimension {} is out of bounds for tensor with {} dimensions",
                        dim,
                        input_shape.dims.len()
                    );
                }

                // Remove the specified dimension
                let mut new_shape = input_shape.dims.clone();
                new_shape.remove(dim);
                Ok(DynamicShape { dims: new_shape })
            }
            None => {
                // No dimension specified - compute mean across all dimensions (scalar)
                Ok(DynamicShape::new(&[]))
            }
        }
    }
}

impl<'data, N: Dtype> From<MeanOp<'data, N>> for NodeRef<'data, N> {
    fn from(op: MeanOp<'data, N>) -> Self {
        NodeOp::Mean(op).into()
    }
}
