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

use std::marker::PhantomData;

use crate::{
    errors::Result,
    target::{Capability, CudaTarget},
};
use teeny_core::{
    compiler::Compiler,
    context::program::{Kernel, Program},
};
pub struct CudaProgram<'a, K: Kernel> {
    _unused: PhantomData<&'a ()>,
    _kernel: PhantomData<K>,
}

impl<'a, K: Kernel> CudaProgram<'a, K> {
    pub fn try_new(
        _compiler: &impl Compiler,
        _kernel: &impl Kernel,
        _target: &CudaTarget,
        _capability: Capability,
    ) -> Result<Self> {
        Ok(Self {
            _unused: PhantomData,
            _kernel: PhantomData,
        })
    }
}

impl<'a, K: Kernel> Program<'a, K> for CudaProgram<'a, K> {}
