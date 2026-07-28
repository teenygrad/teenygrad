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

/// Device-side memory buffers.
pub mod buffer;
/// Device/context management.
pub mod context;
/// Compiled kernel programs.
pub mod program;

/// A device-specific kernel launch configuration (grid/block dimensions, etc).
pub trait LaunchConfig: Sized {}

/// A device capable of allocating buffers and launching kernels.
pub trait Device<'a>: Sized {
    /// This device's buffer type for elements of dtype `N`.
    type Buffer<N: Num>: Buffer<'a, N>;
    /// This device's compiled-program type for kernel `K`.
    type Program<K: Kernel>: Program<'a, K>;
    /// This device's launch configuration type.
    type LaunchConfig: LaunchConfig;

    /// Allocates a buffer for `count` elements of `N`.
    fn buffer<N: Num>(&self, count: usize) -> Result<Self::Buffer<N>>;

    /// Launches `program` with the given launch `cfg` and `args`.
    fn launch<K: Kernel>(
        &self,
        program: &Self::Program<K>,
        cfg: &Self::LaunchConfig,
        args: K::Args<'a>,
    ) -> Result<()>;
}
