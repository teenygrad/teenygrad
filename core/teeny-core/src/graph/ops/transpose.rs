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
use crate::graph::shape::DynamicShape;
use crate::{
    graph::shape::Shape,
    graph::{NodeOp, NodeRef, ops::Op},
};

#[derive(Debug, Clone)]
pub struct TransposeOp<'data> {
    pub input: NodeRef<'data>,
    pub dims: Vec<isize>,
}

impl<'data> TransposeOp<'data> {
    pub fn new(input: NodeRef<'data>, dims: &[isize]) -> Self {
        Self {
            input,
            dims: dims.to_vec(),
        }
    }
}

impl<'data> Op for TransposeOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        let shape = self.input.shape()?;
        let new_shape = shape.permute(&self.dims);
        Ok(new_shape)
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<TransposeOp<'data>> for NodeRef<'data> {
    fn from(op: TransposeOp<'data>) -> Self {
        NodeOp::Transpose(op).into()
    }
}
