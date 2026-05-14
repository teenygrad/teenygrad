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

use alloc::string::String;

use crate::device::program::Kernel;
use crate::errors::Result;

pub trait Target: Sized {
    fn target_cpu(&self) -> Option<String> {
        None
    }
}

pub trait Compiler {
    fn compile(&self, kernel: &impl Kernel, target: &impl Target, force: bool) -> Result<String>;
}

/// GPU compute capability (CUDA SM version).
///
/// Minimum supported: sm_75 (Turing). Triton's MMA acceleration requires sm_75+;
/// sm_70/sm_72 only have a deprecated FMA fallback path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    Sm75,  // Turing:          RTX 20xx, GTX 16xx, T4
    Sm80,  // Ampere DC:       A100, A30
    Sm86,  // Ampere:          RTX 30xx, A40, A10, A16
    Sm87,  // Ampere embedded: Jetson Orin (AGX/NX/Nano)
    Sm89,  // Ada Lovelace:    RTX 40xx, L4, L40S
    Sm90,  // Hopper:          H100, H200
    Sm100, // Blackwell DC:    B100, B200, GB200
    Sm120, // Blackwell:       RTX 50xx (GB10x)
}

impl Capability {
    pub fn from_major_minor(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (7, 5) => Some(Self::Sm75),
            (8, 0) => Some(Self::Sm80),
            (8, 6) => Some(Self::Sm86),
            (8, 7) => Some(Self::Sm87),
            (8, 9) => Some(Self::Sm89),
            (9, 0) => Some(Self::Sm90),
            (10, 0) => Some(Self::Sm100),
            (12, 0) => Some(Self::Sm120),
            _ => None,
        }
    }
}

impl core::fmt::Display for Capability {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm87 => "sm_87",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
            Self::Sm100 => "sm_100",
            Self::Sm120 => "sm_120",
        };
        f.write_str(s)
    }
}
