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

use crate::error::Error;

#[derive(Debug, Clone, Display)]
pub enum Arch {
    #[display("x86_64")]
    X86_64,
}

#[derive(Debug, Clone, Display)]
pub enum Vendor {
    #[display("unknown")]
    Unknown,
}

#[derive(Debug, Clone, Display)]
pub enum Os {
    #[display("linux")]
    Linux,
}

#[derive(Debug, Clone, Display)]
pub enum Abi {
    #[display("gnu")]
    Gnu,
}

#[derive(Debug, Clone)]
pub struct CpuTarget {
    pub arch: Arch,
    pub vendor: Vendor,
    pub os: Os,
    pub abi: Abi,
}

impl std::fmt::Display for CpuTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-{}-{}", self.arch, self.vendor, self.os, self.abi)
    }
}

impl TryFrom<&str> for CpuTarget {
    type Error = Error;

    fn try_from(target: &str) -> std::result::Result<Self, Self::Error> {
        let target = target.split('-').collect::<Vec<&str>>();

        if target.len() != 4 {
            return Err(Error::InvalidTarget(format!("Invalid target: {target:?}")));
        }

        let arch = match target[0] {
            "x86_64" => Arch::X86_64,
            _ => return Err(Error::InvalidTarget(format!("Invalid target: {target:?}"))),
        };

        let vendor = match target[1] {
            "unknown" => Vendor::Unknown,
            _ => return Err(Error::InvalidTarget(format!("Invalid target: {target:?}"))),
        };

        let os = match target[2] {
            "linux" => Os::Linux,
            _ => return Err(Error::InvalidTarget(format!("Invalid target: {target:?}"))),
        };

        let abi = match target[3] {
            "gnu" => Abi::Gnu,
            _ => return Err(Error::InvalidTarget(format!("Invalid target: {target:?}"))),
        };

        Ok(CpuTarget {
            arch,
            vendor,
            os,
            abi,
        })
    }
}
