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
use crate::graph::shape::DynamicShape;
use crate::graph::{NodeOp, NodeRef, ops::Op};

#[derive(Debug, Clone)]
pub struct VMapOp<'data> {
    pub function: NodeRef<'data>,
    pub in_dims: Vec<Option<usize>>,
    pub out_dims: usize,
}

impl<'data> VMapOp<'data> {
    pub fn new(function: NodeRef<'data>, in_dims: &[Option<usize>], out_dims: usize) -> Self {
        Self {
            function,
            in_dims: in_dims.to_vec(),
            out_dims,
        }
    }
}

impl<'data> Op for VMapOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        self.function.dtype()
    }
}

impl<'data> From<VMapOp<'data>> for NodeRef<'data> {
    fn from(op: VMapOp<'data>) -> Self {
        NodeOp::VMap(op).into()
    }
}
