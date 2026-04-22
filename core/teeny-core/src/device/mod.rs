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

use crate::{
    device::{
        buffer::Buffer,
        program::{Kernel, Program},
    },
    dtype::Num,
    errors::Result,
};

pub mod buffer;
pub mod context;
pub mod program;

pub trait LaunchConfig: Sized {}

pub trait Device<'a>: Sized {
    type Buffer<N: Num>: Buffer<'a, N>;
    type Program<K: Kernel>: Program<'a, K>;
    type LaunchConfig: LaunchConfig;

    fn buffer<N: Num>(&self, count: usize) -> Result<Self::Buffer<N>>;

    fn launch<K: Kernel>(
        &self,
        program: &Self::Program<K>,
        cfg: &Self::LaunchConfig,
        args: K::Args<'a>,
    ) -> Result<()>;
}
