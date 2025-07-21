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

use std::marker::PhantomData;

use crate::error::Result;
use crate::{
    dtype::Dtype,
    graph::{NodeOp, NodeRef, ops::OpShape},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct RandnOp<N: Dtype> {
    pub shape: DynamicShape,
    _marker: PhantomData<N>,
}

impl<N: Dtype> RandnOp<N> {
    pub fn new(shape: DynamicShape) -> Self {
        Self {
            shape,
            _marker: PhantomData,
        }
    }
}

impl<N: Dtype> OpShape for RandnOp<N> {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(self.shape.clone())
    }
}

impl<N: Dtype> From<RandnOp<N>> for NodeRef<N> {
    fn from(op: RandnOp<N>) -> Self {
        NodeOp::Randn(op).into()
    }
}
