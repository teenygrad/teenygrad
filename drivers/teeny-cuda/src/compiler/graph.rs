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

use anyhow::anyhow;

use crate::{errors::Result, model::CudaModel};
use teeny_core::{
    compiler::Target,
    graph::{Graph, compiler::GraphCompiler},
    model::{Lowering, Model},
};

#[derive(Debug, Default)]
pub struct CudaGraphCompiler {}

impl CudaGraphCompiler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GraphCompiler for CudaGraphCompiler {
    fn compile<'a, L: Lowering<'a>, T: Target>(
        &self,
        graph: &Graph,
        lowering: &L,
        _target: &T,
    ) -> Result<impl Model<'a>> {
        let dag = lowering.lower(graph)?;
        let model = CudaModel::<'a>::new(dag)?;

        Ok(model)
    }
}
