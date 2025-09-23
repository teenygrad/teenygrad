/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
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
