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

use derive_more::Display;

use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Capability {
    #[display("sm_60")]
    Sm60,
    #[display("sm_61")]
    Sm61,
    #[display("sm_70")]
    Sm70,
    #[display("sm_72")]
    Sm72,
    #[display("sm_75")]
    Sm75,
    #[display("sm_80")]
    Sm80,
    #[display("sm_86")]
    Sm86,
    #[display("sm_89")]
    Sm89,
    #[display("sm_90")]
    Sm90,
}

#[derive(Debug, Clone)]
pub struct CudaTarget {
    pub capability: Capability,
}

impl std::fmt::Display for CudaTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda-{}", self.capability)
    }
}

impl CudaTarget {
    pub fn new(capability: Capability) -> Self {
        Self { capability }
    }
}

impl TryFrom<(i32, i32)> for CudaTarget {
    type Error = Error;

    fn try_from((major, minor): (i32, i32)) -> std::result::Result<Self, Self::Error> {
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
        ]
        .into_iter()
        .collect();
        let capability = capabilities.get(&(major * 10 + minor)).cloned();

        if let Some(capability) = capability {
            return Ok(Self { capability });
        }

        Err(Error::UnknownCapability(format!(
            "Capability not found: {major}.{minor}"
        )))
    }
}
