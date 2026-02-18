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
