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
use crate::graph::ops::Op;
use crate::graph::shape::{DynamicShape, Shape};
use crate::graph::{NodeOp, NodeRef};

#[derive(Debug, Clone)]
pub struct MultOp<'data> {
    pub lhs: NodeRef<'data>,
    pub rhs: NodeRef<'data>,
}

impl<'data> MultOp<'data> {
    pub fn new(lhs: NodeRef<'data>, rhs: NodeRef<'data>) -> Self {
        Self { lhs, rhs }
    }
}

impl<'data> Op for MultOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(self.lhs.shape()?.broadcast(&self.rhs.shape()?))
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<MultOp<'data>> for NodeRef<'data> {
    fn from(op: MultOp<'data>) -> Self {
        NodeOp::Mult(op).into()
    }
}
