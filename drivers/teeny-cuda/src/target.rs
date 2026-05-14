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

use derive_more::Display;

use crate::device::CudaDeviceInfo;
use crate::errors::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Capability {
    #[display("sm_75")]
    Sm75,  // Turing:          RTX 20xx, GTX 16xx, T4
    #[display("sm_80")]
    Sm80,  // Ampere DC:       A100, A30
    #[display("sm_86")]
    Sm86,  // Ampere:          RTX 30xx, A40, A10, A16
    #[display("sm_87")]
    Sm87,  // Ampere embedded: Jetson Orin (AGX/NX/Nano)
    #[display("sm_89")]
    Sm89,  // Ada Lovelace:    RTX 40xx, L4, L40S
    #[display("sm_90")]
    Sm90,  // Hopper:          H100, H200
    #[display("sm_100")]
    Sm100, // Blackwell DC:    B100, B200, GB200
    #[display("sm_120")]
    Sm120, // Blackwell:       RTX 50xx (GB10x)
}

impl Capability {
    pub fn from_device_info(info: &CudaDeviceInfo) -> Result<Self> {
        match (info.major, info.minor) {
            (7, 5) => Ok(Self::Sm75),
            (8, 0) => Ok(Self::Sm80),
            (8, 6) => Ok(Self::Sm86),
            (8, 7) => Ok(Self::Sm87),
            (8, 9) => Ok(Self::Sm89),
            (9, 0) => Ok(Self::Sm90),
            (10, 0) => Ok(Self::Sm100),
            (12, 0) => Ok(Self::Sm120),
            (major, minor) => Err(Error::UnknownCapability(format!("sm_{major}{minor}")).into()),
        }
    }
}
