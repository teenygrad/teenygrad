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
pub struct ArangeOp<'data, N: Dtype> {
    pub start: N,
    pub end: N,
    pub step: N,
    _marker: PhantomData<&'data ()>,
}

impl<'data, N: Dtype> ArangeOp<'data, N> {
    pub fn new(start: N, end: N, step: N) -> Self {
        Self {
            start,
            end,
            step,
            _marker: PhantomData,
        }
    }
}

impl<'data, N: Dtype> OpShape for ArangeOp<'data, N> {
    fn shape(&self) -> Result<DynamicShape> {
        // Calculate the length of the arange sequence
        // Formula: ceil((end - start) / step)
        let length = ((self.end.to_f32() - self.start.to_f32()) / self.step.to_f32()).ceil();
        let length = length as usize;
        Ok(DynamicShape::new(&[length]))
    }
}

impl<'data, N: Dtype> From<ArangeOp<'data, N>> for NodeRef<'data, N> {
    fn from(op: ArangeOp<'data, N>) -> Self {
        NodeOp::Arange(op).into()
    }
}
