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

use crate::{
    dtype::Dtype,
    graph::{NodeOp, NodeRef, ops::OpShape},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct TransposeOp<N: Dtype> {
    pub input: NodeRef<N>,
}

impl<N: Dtype> TransposeOp<N> {
    pub fn new(input: NodeRef<N>) -> Self {
        Self { input }
    }
}

impl<N: Dtype> OpShape for TransposeOp<N> {
    fn shape(&self) -> DynamicShape {
        let input_shape = self.input.shape();
        let mut new_shape = input_shape.dims.clone();
        new_shape.reverse();

        DynamicShape::new(&new_shape)
    }
}

impl<N: Dtype> From<TransposeOp<N>> for NodeRef<N> {
    fn from(op: TransposeOp<N>) -> Self {
        NodeOp::Transpose(op).into()
    }
}
