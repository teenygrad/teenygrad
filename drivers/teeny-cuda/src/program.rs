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

use crate::errors::Result;
use teeny_core::context::program::{Kernel, LaunchConfig, Program};

pub struct CudaLaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: [u32; 3],
}

impl CudaLaunchConfig {
    pub fn new(grid: [u32; 3], block: [u32; 3], cluster: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            cluster,
        }
    }
}

impl LaunchConfig for CudaLaunchConfig {}

pub struct CudaProgram<'a, K: Kernel> {
    _unused: PhantomData<&'a ()>,
    _kernel: PhantomData<K>,
}

impl<'a, K: Kernel> Program<'a, K> for CudaProgram<'a, K> {
    type LaunchConfig = CudaLaunchConfig;
    type Result = ();

    fn launch<'b>(
        &self,
        _cfg: &Self::LaunchConfig,
        _args: <K as Kernel>::Args<'b>,
    ) -> teeny_core::errors::Result<Self::Result>
    where
        'a: 'b,
    {
        todo!()
    }
}
