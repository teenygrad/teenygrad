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

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Capability {
    #[display("sm_60")]
    PascalSm60,
    #[display("sm_61")]
    PascalSm61,
    #[display("sm_70")]
    VoltaSm70,
    #[display("sm_72")]
    VoltaSm72,
    #[display("sm_75")]
    TuringSm75,
    #[display("sm_80")]
    AmpereSm80,
    #[display("sm_86")]
    AmpereSm86,
    #[display("sm_89")]
    HopperSm89,
    #[display("sm_90")]
    AdaLovelaceSm90,
}

#[derive(Debug, Clone)]
pub struct Target {
    pub capability: Capability,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda-{}", self.capability)
    }
}

impl Target {
    pub fn new(capability: Capability) -> Self {
        Self { capability }
    }
}

impl TryFrom<(i32, i32)> for Target {
    type Error = Error;

    fn try_from((major, minor): (i32, i32)) -> Result<Self> {
        let capability = match (major, minor) {
            (6, 0) => Capability::PascalSm60,
            (6, 1) => Capability::PascalSm61,
            (7, 0) => Capability::VoltaSm70,
            (7, 2) => Capability::VoltaSm72,
            (7, 5) => Capability::TuringSm75,
            (8, 0) => Capability::AmpereSm80,
            (8, 6) => Capability::AmpereSm86,
            (8, 9) => Capability::HopperSm89,
            (9, 0) => Capability::AdaLovelaceSm90,
            _ => {
                return Err(Error::UnknownCapability(format!(
                    "Capability not found: {major}.{minor}"
                )));
            }
        };

        Ok(Self { capability })
    }
}
