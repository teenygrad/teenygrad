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
pub struct SqrtOp<'data, N: Dtype> {
    pub input: NodeRef<'data, N>,
}

impl<'data, N: Dtype> SqrtOp<'data, N> {
    pub fn new(input: NodeRef<'data, N>) -> Self {
        Self { input }
    }
}

impl<'data, N: Dtype> OpShape for SqrtOp<'data, N> {
    fn shape(&self) -> Result<DynamicShape> {
        self.input.shape()
    }
}

impl<'data, N: Dtype> From<SqrtOp<'data, N>> for NodeRef<'data, N> {
    fn from(op: SqrtOp<'data, N>) -> Self {
        NodeOp::Sqrt(op).into()
    }
}
