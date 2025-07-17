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

#[derive(Debug, Clone, Display)]
pub enum Arch {
    #[display(fmt = "x86_64")]
    X86_64,
}

#[derive(Debug, Clone, Display)]
pub enum Vendor {
    #[display(fmt = "pc")]
    Pc,
}

#[derive(Debug, Clone, Display)]
pub enum Os {
    #[display(fmt = "linux")]
    Linux,
}

#[derive(Debug, Clone, Display)]
pub enum Abi {
    #[display(fmt = "gnu")]
    Gnu,
}

#[derive(Debug, Clone)]
pub struct Target {
    pub arch: Arch,
    pub vendor: Vendor,
    pub os: Os,
    pub abi: Abi,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-{}-{}", self.arch, self.vendor, self.os, self.abi)
    }
}
