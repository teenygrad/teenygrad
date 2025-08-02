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
pub struct MeanOp<'data> {
    pub input: NodeRef<'data>,
    pub dim: Option<isize>,
}

impl<'data> MeanOp<'data> {
    pub fn new(input: NodeRef<'data>, dim: Option<isize>) -> Self {
        Self { input, dim }
    }
}

impl<'data> Op for MeanOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let shape = self.input.shape()?;
        if let Some(dim) = self.dim {
            let new_shape = shape.unsqueeze(dim);
            Ok(new_shape)
        } else {
            Ok(shape)
        }
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<MeanOp<'data>> for NodeRef<'data> {
    fn from(op: MeanOp<'data>) -> Self {
        NodeOp::Mean(op).into()
    }
}
