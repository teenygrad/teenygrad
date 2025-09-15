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
use crate::safetensors::TensorView;
use crate::{
    graph::shape::DynamicShape,
    graph::{NodeOp, NodeRef, ops::Op},
};

#[derive(Debug, Clone)]
pub struct SafeTensorOp<'data> {
    pub input: TensorView<'data>,
}

impl<'data> SafeTensorOp<'data> {
    pub fn new(input: TensorView<'data>) -> Self {
        Self { input }
    }
}

impl<'data> Op for SafeTensorOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::from(self.input.0.shape()))
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<SafeTensorOp<'data>> for NodeRef<'data> {
    fn from(op: SafeTensorOp<'data>) -> Self {
        NodeOp::SafeTensor(op).into()
    }
}
