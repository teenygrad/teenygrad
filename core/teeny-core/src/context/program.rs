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

use crate::errors::Error;

/// Grid/workgroup launch configuration for a kernel program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: [u32; 3],
}

impl LaunchConfig {
    #[must_use]
    pub const fn new(grid: [u32; 3], block: [u32; 3], cluster: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            cluster,
        }
    }
}

pub trait Kernel {
    type Args<'a>;
}

pub trait Program<'a, K: Kernel>: Sized {
    fn launch<'b>(&self, cfg: LaunchConfig, args: K::Args<'b>) -> Result<(), Error>
    where
        'a: 'b;
}
