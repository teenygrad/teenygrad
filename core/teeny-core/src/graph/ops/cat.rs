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
pub struct CatOp<'data> {
    pub inputs: Vec<NodeRef<'data>>,
    pub dim: isize,
}

impl<'data> CatOp<'data> {
    pub fn new(inputs: &[NodeRef<'data>], dim: isize) -> Self {
        Self {
            inputs: inputs.to_vec(),
            dim,
        }
    }
}

impl<'data> Op for CatOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let mut shape = self.inputs[0].shape()?;
        for input in &self.inputs[1..] {
            shape = shape.broadcast(&input.shape()?);
        }
        Ok(shape)
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<CatOp<'data>> for NodeRef<'data> {
    fn from(op: CatOp<'data>) -> Self {
        NodeOp::Cat(op).into()
    }
}
