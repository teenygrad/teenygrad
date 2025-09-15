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

use crate::{
    graph::shape::DynamicShape,
    graph::{NodeOp, NodeRef, ops::Op},
};

#[derive(Debug, Clone)]
pub struct RandnOp {
    pub shape: DynamicShape,
    pub dtype: DtypeEnum,
}

impl RandnOp {
    pub fn new(shape: DynamicShape, dtype: DtypeEnum) -> Self {
        Self { shape, dtype }
    }
}

impl Op for RandnOp {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(self.shape.clone())
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<RandnOp> for NodeRef<'data> {
    fn from(op: RandnOp) -> Self {
        NodeOp::Randn(op).into()
    }
}
