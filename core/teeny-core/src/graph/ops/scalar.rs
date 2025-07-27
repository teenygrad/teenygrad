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
pub struct ScalarOp<'data, N: Dtype> {
    pub scalar: N,
    pub _marker: PhantomData<&'data ()>,
}

impl<'data, N: Dtype> ScalarOp<'data, N> {
    pub fn new(scalar: N) -> Self {
        Self {
            scalar,
            _marker: PhantomData,
        }
    }
}

impl<'data, N: Dtype> OpShape for ScalarOp<'data, N> {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::new(&[]))
    }
}

impl<'data, N: Dtype> From<ScalarOp<'data, N>> for NodeRef<'data, N> {
    fn from(op: ScalarOp<'data, N>) -> Self {
        NodeOp::Scalar(op).into()
    }
}
