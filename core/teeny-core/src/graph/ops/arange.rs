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
use crate::value::Value;
use crate::{
    graph::{NodeOp, NodeRef, ops::Op},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct ArangeOp {
    pub start: Value,
    pub end: Value,
    pub step: Value,
}

impl ArangeOp {
    pub fn new<N: Into<Value>>(start: N, end: N, step: N) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            step: step.into(),
        }
    }
}

impl Op for ArangeOp {
    fn shape(&self) -> Result<DynamicShape> {
        let len = 0; // = (self.end - self.start) / self.step;
        Ok(DynamicShape::new(&[len]))
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<ArangeOp> for NodeRef<'data> {
    fn from(op: ArangeOp) -> Self {
        NodeOp::Arange(op).into()
    }
}
