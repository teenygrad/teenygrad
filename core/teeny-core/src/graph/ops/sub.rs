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
use crate::graph::ops::Op;
use crate::graph::{NodeOp, NodeRef};
use crate::tensor::shape::{DynamicShape, Shape};

#[derive(Debug, Clone)]
pub struct SubOp<'data> {
    pub lhs: NodeRef<'data>,
    pub rhs: NodeRef<'data>,
}

impl<'data> SubOp<'data> {
    pub fn new(lhs: NodeRef<'data>, rhs: NodeRef<'data>) -> Self {
        Self { lhs, rhs }
    }
}

impl<'data> Op for SubOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(self.lhs.shape()?.broadcast(&self.rhs.shape()?))
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<SubOp<'data>> for NodeRef<'data> {
    fn from(op: SubOp<'data>) -> Self {
        NodeOp::Sub(op).into()
    }
}
