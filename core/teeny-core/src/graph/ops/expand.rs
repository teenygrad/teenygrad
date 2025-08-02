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

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::graph::{NodeOp, NodeRef, ops::Op};
use crate::tensor::shape::{DynamicShape, Shape};

#[derive(Debug, Clone)]
pub struct ExpandOp<'data> {
    pub input: NodeRef<'data>,
    pub dims: Vec<isize>,
}

impl<'data> ExpandOp<'data> {
    pub fn new(input: NodeRef<'data>, dims: &[isize]) -> Self {
        Self {
            input,
            dims: dims.to_vec(),
        }
    }
}

impl<'data> Op for ExpandOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let mut shape = self.input.shape()?;
        for dim in &self.dims {
            shape = shape.unsqueeze(*dim);
        }
        Ok(shape)
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<ExpandOp<'data>> for NodeRef<'data> {
    fn from(op: ExpandOp<'data>) -> Self {
        NodeOp::Expand(op).into()
    }
}
