/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! Best-effort [`HardwareProfile`]s for AOT compilation, keyed by GPU compute
//! capability (`--options gpu-name=sm_XX`), for [`teeny_kernels::graph::Anduin`]'s
//! memory-level search when `aot_compile` has no open device to query one
//! from (see [`teeny_core::device::context::DeviceInfo::hardware_profile`]
//! for that queried path).
//!
//! Profiles live in `hardware_profiles.json` (embedded at compile time) so
//! entries can be added or corrected without touching this loader. Each
//! entry is *one representative chip* for its capability class, not a
//! queried fact about the actual target device — a capability groups chips
//! with wildly different SM counts and VRAM (e.g. `sm_86` spans an RTX 3060
//! 12GB through an A40 48GB). `compute_units` can be overridden per-run via
//! `--options sm-count=N`; there is no override yet for memory
//! capacity/bandwidth.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use teeny_core::device::hardware::{HardwareProfile, MemoryLevel, MemoryLevelKind};
use teeny_cuda::compiler::target::Capability;

const HARDWARE_PROFILES_JSON: &str = include_str!("hardware_profiles.json");

#[derive(Deserialize)]
struct ProfileTable {
    profiles: Vec<ProfileEntry>,
}

#[derive(Deserialize)]
struct ProfileEntry {
    capability: String,
    name: String,
    compute_units: u32,
    shared_memory_bytes: u64,
    device_memory_bytes: u64,
    device_memory_bandwidth_bytes_per_sec: Option<f64>,
    /// Provenance/confidence notes for this entry. Not consumed by the
    /// loader — kept in the data file for humans reading/editing it.
    #[allow(dead_code)]
    notes: String,
}

/// Looks up the packaged default [`HardwareProfile`] for `capability`,
/// overriding `compute_units` with `sm_count_override` when given (from
/// `--options sm-count=N`).
///
/// Errors if `capability` has no entry in `hardware_profiles.json` — every
/// [`Capability`] variant is expected to have one; a missing entry means the
/// data file is out of sync with that enum.
pub fn hardware_profile_for(
    capability: Capability,
    sm_count_override: Option<u32>,
) -> Result<HardwareProfile> {
    let table: ProfileTable = serde_json::from_str(HARDWARE_PROFILES_JSON)
        .context("failed to parse the packaged hardware_profiles.json")?;

    let by_capability: HashMap<String, ProfileEntry> = table
        .profiles
        .into_iter()
        .map(|entry| (entry.capability.clone(), entry))
        .collect();

    let key = capability.to_string();
    let entry = by_capability.get(&key).ok_or_else(|| {
        anyhow!("hardware_profiles.json has no entry for capability '{key}'")
    })?;

    Ok(HardwareProfile {
        name: entry.name.clone(),
        compute_units: sm_count_override.unwrap_or(entry.compute_units),
        memory_levels: vec![
            MemoryLevel {
                kind: MemoryLevelKind::SharedMemory,
                capacity: entry.shared_memory_bytes,
                bandwidth: None,
                latency: None,
            },
            MemoryLevel {
                kind: MemoryLevelKind::DeviceMemory,
                capacity: entry.device_memory_bytes,
                bandwidth: entry.device_memory_bandwidth_bytes_per_sec,
                latency: None,
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_capability_has_a_profile() {
        for capability in [
            Capability::Sm75,
            Capability::Sm80,
            Capability::Sm86,
            Capability::Sm87,
            Capability::Sm89,
            Capability::Sm90,
            Capability::Sm100,
            Capability::Sm120,
        ] {
            hardware_profile_for(capability, None)
                .unwrap_or_else(|e| panic!("missing hardware profile for {capability}: {e}"));
        }
    }

    #[test]
    fn sm_count_override_wins_over_the_packaged_default() {
        let profile = hardware_profile_for(Capability::Sm87, Some(16))
            .expect("sm_87 has a packaged profile");
        assert_eq!(profile.compute_units, 16);
    }
}
