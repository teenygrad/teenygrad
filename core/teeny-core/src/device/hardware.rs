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

//! Backend-agnostic hardware facts for tile-shape / scheduling cost models
//! (e.g. `teeny-triton`'s Welder-style cost model, teenygrad-3w0).
//!
//! [`HardwareProfile`] is plain data: it is freely constructible independent
//! of any open device (a hand-calibrated profile for tests/CI that run
//! without hardware, for example), while [`super::context::DeviceInfo`]
//! implementations additionally expose a *real*, queried one via
//! [`super::context::DeviceInfo::hardware_profile`].

use alloc::{string::String, vec::Vec};

/// Coarse category of one level in a device's memory hierarchy.
///
/// Purely descriptive today -- nothing in this crate branches on a specific
/// variant. New variants (e.g. for CPU NUMA nodes, or Vulkan's host-visible
/// vs. device-local heaps) are expected as new backends land, so this is
/// intentionally `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLevelKind {
    /// Per-thread registers.
    Register,
    /// Software-managed on-chip scratch memory shared within a block/
    /// workgroup (CUDA shared memory, Vulkan `shared` storage, ...).
    SharedMemory,
    /// Hardware-managed level-1 cache.
    L1Cache,
    /// Hardware-managed level-2 cache.
    L2Cache,
    /// Hardware-managed level-3 cache.
    L3Cache,
    /// Device-local main memory (GPU VRAM, ...).
    DeviceMemory,
    /// Host-resident main memory (system RAM).
    HostMemory,
}

/// One level of a device's memory hierarchy: how big it is, and how
/// expensive it is to move data through it.
///
/// `bandwidth_bytes_per_sec`/`latency_ns` are `None` where a backend can't
/// source them from real queried data (e.g. `cudaGetDeviceProperties` has no
/// shared-memory/L2 bandwidth field) -- left unset rather than guessed, so a
/// cost model can tell "unknown" apart from "zero".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryLevel {
    /// What kind of memory this level is.
    pub kind: MemoryLevelKind,
    /// Capacity of this level, in bytes (e.g. per-SM shared-memory budget,
    /// total L2 size, total device memory).
    pub capacity_bytes: u64,
    /// Peak achievable bandwidth through this level, in bytes/second, if known.
    pub bandwidth_bytes_per_sec: Option<f64>,
    /// Approximate access latency of this level, in nanoseconds, if known.
    pub latency_ns: Option<f64>,
}

/// A device's static hardware facts relevant to tile-shape / scheduling cost
/// models.
///
/// Ordinary data -- construct one by hand (e.g. a calibrated profile for
/// tests/CI without hardware) or obtain a real one from an open device via
/// [`super::context::DeviceInfo::hardware_profile`].
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareProfile {
    /// Human-readable device name (e.g. `"NVIDIA GeForce RTX 5070"`), for
    /// diagnostics/logging only.
    pub name: String,
    /// Number of independent compute units capable of concurrent execution
    /// (CUDA streaming multiprocessors, CPU cores, ...).
    pub compute_units: u32,
    /// This device's memory hierarchy, ordered from closest/fastest/smallest
    /// to farthest/slowest/largest.
    pub memory_levels: Vec<MemoryLevel>,
}

impl HardwareProfile {
    /// The first [`MemoryLevel`] of `kind` in [`Self::memory_levels`], if any.
    pub fn level(&self, kind: MemoryLevelKind) -> Option<&MemoryLevel> {
        self.memory_levels.iter().find(|level| level.kind == kind)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn profile() -> HardwareProfile {
        HardwareProfile {
            name: String::from("test-device"),
            compute_units: 48,
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::SharedMemory,
                    capacity_bytes: 49_152,
                    bandwidth_bytes_per_sec: None,
                    latency_ns: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity_bytes: 12 * 1024 * 1024 * 1024,
                    bandwidth_bytes_per_sec: Some(672.0e9),
                    latency_ns: None,
                },
            ],
        }
    }

    #[test]
    fn constructible_without_a_device() {
        let profile = profile();
        assert_eq!(profile.compute_units, 48);
        assert_eq!(profile.memory_levels.len(), 2);
    }

    #[test]
    fn level_finds_the_matching_kind() {
        let profile = profile();
        let shared = profile
            .level(MemoryLevelKind::SharedMemory)
            .expect("profile declares a SharedMemory level");
        assert_eq!(shared.capacity_bytes, 49_152);
    }

    #[test]
    fn level_returns_none_for_an_absent_kind() {
        let profile = profile();
        assert!(profile.level(MemoryLevelKind::L1Cache).is_none());
    }
}
