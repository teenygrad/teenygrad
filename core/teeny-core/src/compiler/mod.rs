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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    Sm60,
    Sm61,
    Sm70,
    Sm72,
    Sm75,
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm100,
    Sm120a,
}

impl Capability {
    pub fn from_major_minor(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (6, 0) => Some(Self::Sm60),
            (6, 1) => Some(Self::Sm61),
            (7, 0) => Some(Self::Sm70),
            (7, 2) => Some(Self::Sm72),
            (7, 5) => Some(Self::Sm75),
            (8, 0) => Some(Self::Sm80),
            (8, 6) => Some(Self::Sm86),
            (8, 9) => Some(Self::Sm89),
            (9, 0) => Some(Self::Sm90),
            (10, 0) => Some(Self::Sm100),
            (12, 0) => Some(Self::Sm120a),
            _ => None,
        }
    }
}

impl core::fmt::Display for Capability {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Sm60 => "sm_60",
            Self::Sm61 => "sm_61",
            Self::Sm70 => "sm_70",
            Self::Sm72 => "sm_72",
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
            Self::Sm100 => "sm_100",
            Self::Sm120a => "sm_120a",
        };
        f.write_str(s)
    }
}
