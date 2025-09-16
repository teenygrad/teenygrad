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
