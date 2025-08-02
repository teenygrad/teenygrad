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
pub struct UnsqueezeOp<'data> {
    pub input: NodeRef<'data>,
    pub dim: isize,
}

impl<'data> UnsqueezeOp<'data> {
    pub fn new(input: NodeRef<'data>, dim: isize) -> Self {
        Self { input, dim }
    }
}

impl<'data> Op for UnsqueezeOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let shape = self.input.shape()?;
        Ok(shape.unsqueeze(self.dim))
    }

    fn dtype(&self) -> DtypeEnum {
        self.input.dtype()
    }
}

impl<'data> From<UnsqueezeOp<'data>> for NodeRef<'data> {
    fn from(op: UnsqueezeOp<'data>) -> Self {
        NodeOp::Unsqueeze(op).into()
    }
}
