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
use crate::tensor::shape::DynamicShape;
use crate::{
    graph::{NodeOp, NodeRef, ops::Op},
    tensor::shape::Shape,
};

#[derive(Debug, Clone)]
pub struct TransposeOp<'data> {
    pub input: NodeRef<'data>,
    pub dims: Vec<isize>,
}

impl<'data> TransposeOp<'data> {
    pub fn new(input: NodeRef<'data>, dims: &[isize]) -> Self {
        Self {
            input,
            dims: dims.to_vec(),
        }
    }
}

impl<'data> Op for TransposeOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let shape = self.input.shape()?;
        let new_shape = shape.permute(&self.dims);
        Ok(new_shape)
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<TransposeOp<'data>> for NodeRef<'data> {
    fn from(op: TransposeOp<'data>) -> Self {
        NodeOp::Transpose(op).into()
    }
}
