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
use crate::tensor::shape::DynamicShape;

#[derive(Debug, Clone)]
pub struct IndexOp<'data> {
    pub input: NodeRef<'data>,
    pub indices: Vec<usize>,
}

impl<'data> IndexOp<'data> {
    pub fn new(input: NodeRef<'data>, indices: Vec<usize>) -> Self {
        Self { input, indices }
    }
}

impl<'data> Op for IndexOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<IndexOp<'data>> for NodeRef<'data> {
    fn from(op: IndexOp<'data>) -> Self {
        NodeOp::Index(op).into()
    }
}
