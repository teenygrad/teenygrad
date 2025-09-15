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

use crate::error::Result;

use crate::{
    dtype::DtypeEnum,
    graph::shape::DynamicShape,
    graph::{NodeOp, NodeRef, ops::Op},
};

#[derive(Debug, Clone)]
pub struct WhereOp<'data> {
    pub condition: NodeRef<'data>,
    pub if_true: NodeRef<'data>,
    pub if_false: NodeRef<'data>,
}

impl<'data> WhereOp<'data> {
    pub fn new(
        condition: NodeRef<'data>,
        if_true: NodeRef<'data>,
        if_false: NodeRef<'data>,
    ) -> Result<Self> {
        assert_eq!(if_true.shape()?, if_false.shape()?);
        assert_eq!(if_true.dtype(), if_false.dtype());

        Ok(Self {
            condition,
            if_true,
            if_false,
        })
    }
}

impl<'data> Op for WhereOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        self.if_true.shape()
    }

    fn dtype(&self) -> DtypeEnum {
        self.if_true.dtype()
    }
}

impl<'data> From<WhereOp<'data>> for NodeRef<'data> {
    fn from(op: WhereOp<'data>) -> Self {
        NodeOp::Where(op).into()
    }
}
