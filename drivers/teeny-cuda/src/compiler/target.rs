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

use crate::device::CudaDeviceInfo;
use crate::errors::{Error, Result};

/// GPU compute capability (CUDA SM version).
///
/// Minimum supported: sm_75 (Turing). Triton's MMA acceleration requires sm_75+;
/// sm_70/sm_72 only have a deprecated FMA fallback path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Turing: RTX 20xx, GTX 16xx, T4.
    Sm75,
    /// Ampere datacenter: A100, A30.
    Sm80,
    /// Ampere: RTX 30xx, A40, A10, A16.
    Sm86,
    /// Ampere embedded: Jetson Orin (AGX/NX/Nano).
    Sm87,
    /// Ada Lovelace: RTX 40xx, L4, L40S.
    Sm89,
    /// Hopper: H100, H200.
    Sm90,
    /// Blackwell datacenter: B100, B200, GB200.
    Sm100,
    /// Blackwell: RTX 50xx (GB10x).
    Sm120,
}

impl Capability {
    /// Looks up the `Capability` matching a CUDA `(major, minor)` compute-capability version.
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

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::str::FromStr for Capability {
    type Err = String;

    /// Accepts the canonical `sm_90` form as well as `sm-90`, `sm90` and bare
    /// `90` (case-insensitive) — the latter forms are convenient for CLI args
    /// like `--options capability=sm_90`.
    fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
        let normalized = s.trim().to_ascii_lowercase();
        let digits = normalized
            .strip_prefix("sm_")
            .or_else(|| normalized.strip_prefix("sm-"))
            .or_else(|| normalized.strip_prefix("sm"))
            .unwrap_or(normalized.as_str());

        match digits {
            "75" => Ok(Self::Sm75),
            "80" => Ok(Self::Sm80),
            "86" => Ok(Self::Sm86),
            "87" => Ok(Self::Sm87),
            "89" => Ok(Self::Sm89),
            "90" => Ok(Self::Sm90),
            "100" => Ok(Self::Sm100),
            "120" => Ok(Self::Sm120),
            _ => Err(format!(
                "unknown capability '{s}'; expected one of sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, sm_100, sm_120"
            )),
        }
    }
}

/// A CUDA compilation target: a single GPU compute capability.
pub struct Target {
    /// The target GPU's compute capability.
    pub capability: Capability,
}

impl Target {
    /// Creates a target for the given compute `capability`.
    pub fn new(capability: Capability) -> Self {
        Self { capability }
    }
}

impl teeny_core::compiler::Target for Target {
    fn target_cpu(&self) -> Option<std::string::String> {
        Some(self.capability.to_string())
    }
}

impl TryFrom<(i32, i32)> for Target {
    type Error = anyhow::Error;

    fn try_from((major, minor): (i32, i32)) -> Result<Self> {
        let capability = Capability::from_major_minor(major, minor).ok_or_else(|| {
            Error::UnknownCapability(format!("Capability not found: {major}.{minor}"))
        })?;
        Ok(Self { capability })
    }
}

/// Derives a compute [`Capability`] from a device's major/minor version, erroring if it doesn't
/// match a known architecture.
pub fn capability_from_device_info(info: &CudaDeviceInfo) -> Result<Capability> {
    Capability::from_major_minor(info.major, info.minor)
        .ok_or_else(|| Error::UnknownCapability(format!("sm_{}{}", info.major, info.minor)).into())
}

#[cfg(test)]
mod capability_from_str_tests {
    use super::Capability;

    #[test]
    fn accepts_canonical_and_tolerant_forms() {
        assert_eq!("sm_90".parse::<Capability>().unwrap(), Capability::Sm90);
        assert_eq!("sm-90".parse::<Capability>().unwrap(), Capability::Sm90);
        assert_eq!("SM90".parse::<Capability>().unwrap(), Capability::Sm90);
        assert_eq!("90".parse::<Capability>().unwrap(), Capability::Sm90);
        assert_eq!("sm_120".parse::<Capability>().unwrap(), Capability::Sm120);
    }

    #[test]
    fn rejects_unknown_capability() {
        assert!("sm_61".parse::<Capability>().is_err());
    }
}
