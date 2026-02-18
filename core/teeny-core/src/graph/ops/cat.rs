/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::graph::shape::{DynamicShape, Shape};
use crate::graph::{NodeOp, NodeRef, ops::Op};

#[derive(Debug, Clone)]
pub struct CatOp<'data> {
    pub inputs: Vec<NodeRef<'data>>,
    pub dim: isize,
}

impl<'data> CatOp<'data> {
    pub fn new(inputs: &[NodeRef<'data>], dim: isize) -> Self {
        Self {
            inputs: inputs.to_vec(),
            dim,
        }
    }
}

impl<'data> Op for CatOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let mut shape = self.inputs[0].shape()?;
        for input in &self.inputs[1..] {
            shape = shape.broadcast(&input.shape()?);
        }
        Ok(shape)
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<CatOp<'data>> for NodeRef<'data> {
    fn from(op: CatOp<'data>) -> Self {
        NodeOp::Cat(op).into()
    }
}
