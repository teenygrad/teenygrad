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
pub struct ArangeOp<N: Dtype> {
    pub start: N,
    pub end: N,
    pub step: N,
}

impl<N: Dtype> ArangeOp<N> {
    pub fn new(start: N, end: N, step: N) -> Self {
        Self { start, end, step }
    }
}

impl<N: Dtype> OpShape for ArangeOp<N> {
    fn shape(&self) -> DynamicShape {
        // Calculate the length of the arange sequence
        // Formula: ceil((end - start) / step)
        let length = ((self.end - self.start) / self.step).ceil();
        let length = length.to_f32().unwrap_or(0.0) as usize;
        DynamicShape::new(&[length])
    }
}

impl<N: Dtype> From<ArangeOp<N>> for NodeRef<N> {
    fn from(op: ArangeOp<N>) -> Self {
        NodeOp::Arange(op).into()
    }
}
