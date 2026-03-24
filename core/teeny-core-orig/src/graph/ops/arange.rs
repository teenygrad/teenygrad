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

use crate::dtype::{self, DtypeEnum};
use crate::error::Result;
use crate::value::Value;
use crate::{
    graph::shape::DynamicShape,
    graph::{NodeOp, NodeRef, ops::Op},
};

#[derive(Debug, Clone, PartialEq)]
pub struct ArangeOp {
    pub start: Value,
    pub end: Value,
    pub step: Value,
}

impl ArangeOp {
    pub fn new<N: dtype::Dtype + Into<Value>>(start: N, end: N, step: N) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            step: step.into(),
        }
    }
}

impl Op for ArangeOp {
    fn shape(&self) -> Result<DynamicShape> {
        let len = &(&self.end - &self.start) / &self.step;

        match len {
            Value::Usize(len) => Ok(DynamicShape::new(&[len])),
            Value::F32(len) => Ok(DynamicShape::new(&[len as usize])),
            _ => todo!(),
        }
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
