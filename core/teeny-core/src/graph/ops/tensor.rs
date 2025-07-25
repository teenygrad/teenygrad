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

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;

use crate::error::Result;
use crate::{
    dtype::Dtype,
    graph::{NodeOp, NodeRef, ops::OpShape},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct TensorOp<N: Dtype> {
    #[cfg(feature = "ndarray")]
    pub input: ndarray::Array<N, IxDyn>,
}

impl<N: Dtype> TensorOp<N> {
    #[cfg(feature = "ndarray")]
    pub fn new(input: ndarray::Array<N, IxDyn>) -> Self {
        Self { input }
    }
}

impl<N: Dtype> OpShape for TensorOp<N> {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::new(&[1, self.input.len()]))
    }
}

impl<N: Dtype> From<TensorOp<N>> for NodeRef<'static, N> {
    fn from(op: TensorOp<N>) -> Self {
        NodeOp::Tensor(op).into()
    }
}
