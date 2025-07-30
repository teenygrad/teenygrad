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
    graph::{NodeOp, NodeRef, ops::Op},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct OnesOp {
    pub shape: DynamicShape,
    pub dtype: DtypeEnum,
}

impl OnesOp {
    pub fn new(shape: DynamicShape, dtype: DtypeEnum) -> Self {
        Self { shape, dtype }
    }
}

impl Op for OnesOp {
    fn shape(&self) -> Result<DynamicShape> {
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<OnesOp> for NodeRef<'data> {
    fn from(op: OnesOp) -> Self {
        NodeOp::Ones(op).into()
    }
}
