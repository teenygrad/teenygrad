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
pub struct ExpandOp<'data> {
    pub input: NodeRef<'data>,
    pub dims: Vec<isize>,
}

impl<'data> ExpandOp<'data> {
    pub fn new(input: NodeRef<'data>, dims: &[isize]) -> Self {
        Self {
            input,
            dims: dims.to_vec(),
        }
    }
}

impl<'data> Op for ExpandOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let mut shape = self.input.shape()?;
        for dim in &self.dims {
            shape = shape.unsqueeze(*dim);
        }
        Ok(shape)
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<ExpandOp<'data>> for NodeRef<'data> {
    fn from(op: ExpandOp<'data>) -> Self {
        NodeOp::Expand(op).into()
    }
}
