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

use std::collections::HashMap;

use crate::errors::{Error, Result};
use teeny_cuda::compiler::target::Capability;

pub struct Target {
    pub capability: Capability,
}

impl Target {
    pub fn new(capability: Capability) -> Self {
        Self { capability }
    }
}

impl teeny_core::compiler::Target for Target {}

impl TryFrom<(i32, i32)> for Target {
    type Error = anyhow::Error;

    fn try_from((major, minor): (i32, i32)) -> Result<Self> {
        let capabilities: HashMap<i32, Capability> = vec![
            (60, Capability::Sm60),
            (61, Capability::Sm61),
            (70, Capability::Sm70),
            (72, Capability::Sm72),
            (75, Capability::Sm75),
            (80, Capability::Sm80),
            (86, Capability::Sm86),
            (89, Capability::Sm89),
            (90, Capability::Sm90),
            (100, Capability::Sm100),
            (120, Capability::Sm120a),
        ]
        .into_iter()
        .collect();
        let capability = capabilities
            .get(&(major * 10 + minor))
            .cloned()
            .ok_or_else(|| {
                Error::UnknownCapability(format!("Capability not found: {major}.{minor}"))
            })?;

        Ok(Self { capability })
    }
}
