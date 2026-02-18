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
use crate::graph::{NodeOp, NodeRef, ops::Op};

#[derive(Debug, Clone)]
pub struct VMapOp<'data> {
    pub function: NodeRef<'data>,
    pub in_dims: Vec<Option<usize>>,
    pub out_dims: usize,
}

impl<'data> VMapOp<'data> {
    pub fn new(function: NodeRef<'data>, in_dims: &[Option<usize>], out_dims: usize) -> Self {
        Self {
            function,
            in_dims: in_dims.to_vec(),
            out_dims,
        }
    }
}

impl<'data> Op for VMapOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        self.function.dtype()
    }
}

impl<'data> From<VMapOp<'data>> for NodeRef<'data> {
    fn from(op: VMapOp<'data>) -> Self {
        NodeOp::VMap(op).into()
    }
}
