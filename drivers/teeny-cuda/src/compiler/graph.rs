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

use crate::errors::Result;
use teeny_core::{
    compiler::Target,
    graph::{Graph, compiler::GraphCompiler},
    model::{Lowering, Model},
};

use crate::compiler::PtxCompiler;

pub struct CudaGraphCompiler {
    pub ptx_compiler: PtxCompiler,
}

impl CudaGraphCompiler {
    pub fn new(compiler: PtxCompiler) -> Self {
        Self {
            ptx_compiler: compiler,
        }
    }
}

impl GraphCompiler for CudaGraphCompiler {
    fn compile<'a, L: Lowering<'a>, T: Target, M: Model<'a>>(
        &self,
        _graph: &Graph,
        _lowering: &L,
        _target: &T,
    ) -> Result<M> {
        todo!()
    }
}
