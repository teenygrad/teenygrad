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
pub struct NotEqOp<'data> {
    pub lhs: NodeRef<'data>,
    pub rhs: NodeRef<'data>,
}

impl<'data> NotEqOp<'data> {
    pub fn new(lhs: NodeRef<'data>, rhs: NodeRef<'data>) -> Self {
        Self { lhs, rhs }
    }
}

impl<'data> Op for NotEqOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        self.lhs.shape()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<NotEqOp<'data>> for NodeRef<'data> {
    fn from(op: NotEqOp<'data>) -> Self {
        NodeOp::NotEq(op).into()
    }
}
