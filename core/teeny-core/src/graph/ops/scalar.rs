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
use crate::value::Value;
use crate::{
    graph::shape::DynamicShape,
    graph::{NodeOp, NodeRef, ops::Op},
};

#[derive(Debug, Clone)]
pub struct ScalarOp {
    pub scalar: Value,
}

impl ScalarOp {
    pub fn new(scalar: Value) -> Self {
        Self { scalar }
    }
}

impl Op for ScalarOp {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::new(&[]))
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<ScalarOp> for NodeRef<'data> {
    fn from(op: ScalarOp) -> Self {
        NodeOp::Scalar(op).into()
    }
}
