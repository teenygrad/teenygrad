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
/// Variants are declared in, and ordered (via the derived [`PartialOrd`]/
/// [`Ord`]) by, closest/fastest/smallest to farthest/slowest/largest --
/// `Register < SharedMemory < ... < HostMemory` -- so scheduling code can
/// compare levels directly (e.g. "is this edge's level above shared memory")
/// without going through a specific device's [`HardwareProfile`]. A new
/// variant must be inserted at its correct position in the hierarchy, not
/// just appended, or this ordering silently breaks.
///
/// New variants (e.g. for CPU NUMA nodes, or Vulkan's host-visible vs.
/// device-local heaps) are expected as new backends land, so this is
/// intentionally `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub capacity: u64,
    /// Peak achievable bandwidth through this level, in bytes/second, if known.
    pub bandwidth: Option<f64>,
    /// Approximate access latency of this level, in nanoseconds, if known.
    pub latency: Option<f64>,
}

/// Portable SIMT/SIMD execution-granularity facts: how many lanes execute
/// in lockstep, and the hardware/API-imposed bounds on a thread block
/// (work-group)'s size and a grid's shape. Used to compute a launch grid
/// for fused kernels (teenygrad-1nr.17) — e.g. Welder's own rule of thumb,
/// "the greatest common divisor among the tile numbers of all operators as
/// the common thread-block size, if it is larger than the hardware
/// parallelism (e.g., 128) and less than the maximum limitation (e.g.,
/// 1024)".
///
/// Deliberately carries no stored *minimum* thread-block size (Welder's
/// "128" above): that number is `4 * simt_width` on CUDA (warp 32) but
/// would be `4 * simt_width` = 256 on an AMD wave64 device -- it's a
/// scheduling-policy multiplier applied to [`Self::simt_width`], not a
/// hardware fact this struct should hardcode. Consumers compute it as
/// `occupancy_multiplier * simt_width` themselves. This mirrors
/// [`MemoryLevel`]'s own "real data or nothing, never a guess" policy: only
/// [`Self::max_threads_per_group`] (a hard launch-failure limit) belongs
/// here, not a tunable floor.
///
/// Known portability rough edges, not solved by this struct:
/// - **AMD RDNA** is dual-mode -- wave32 or wave64 is a per-kernel compile
///   choice, not a fixed device constant the way CUDA's warp size is.
///   [`Self::simt_width`] as a single queried device fact will need a
///   caveat (or a compiler-level override) once a real AMD backend lands.
/// - **RISC-V RVV** has no warp/wavefront concept at all: vector length
///   (`VLEN`) is runtime-queried via `vsetvli`, and the *effective* element
///   count varies with dtype (`SEW`)/`LMUL`. [`Self::simt_width`] won't be
///   a single fixed number the way it is for GPUs -- it'll likely need to
///   be computed per-dtype at codegen time rather than read once from a
///   profile, once a real RISC-V backend exists to design that against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionProfile {
    /// Number of lanes that execute in lockstep as one hardware-scheduled
    /// unit: 32 on every NVIDIA architecture ("warp"), 32 or 64 on AMD
    /// ("wavefront" -- see this struct's doc comment on RDNA's dual-mode
    /// caveat), or a CPU's SIMD width in elements. `1` for a plain scalar
    /// core with no such grouping.
    pub simt_width: u32,
    /// Hardware/API-imposed maximum threads (work-items) per thread block
    /// (work-group) -- e.g. 1024 on essentially every current CUDA/ROCm
    /// device. Not a tuning knob: exceeding it is a launch failure, unlike
    /// the occupancy-driven minimum described in this struct's own doc
    /// comment.
    pub max_threads_per_group: u32,
    /// Maximum number of thread blocks (work-groups) resident on one
    /// compute unit at once, if the backend can query it (CUDA:
    /// `cudaDeviceProp::maxBlocksPerMultiProcessor`) -- relevant for
    /// occupancy-driven grid sizing. `None` where unknown, rather than
    /// guessed.
    pub max_groups_per_compute_unit: Option<u32>,
    /// Hardware/API-imposed maximum grid size, per dimension (CUDA:
    /// `cudaDeviceProp::maxGridSize`, effectively unbounded in the `x`
    /// dimension and 65535 in `y`/`z` on every current architecture).
    pub max_grid_dims: [u32; 3],
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
    /// This device's SIMT/SIMD execution-granularity facts (warp/wavefront
    /// size, thread-block/grid bounds), if known. `None` for a profile with
    /// no such hierarchy (e.g. a plain scalar CPU core), or where a backend
    /// hasn't wired this up yet.
    pub execution: Option<ExecutionProfile>,
}

impl HardwareProfile {
    /// The first [`MemoryLevel`] of `kind` in [`Self::memory_levels`], if any.
    pub fn level(&self, kind: MemoryLevelKind) -> Option<&MemoryLevel> {
        self.memory_levels.iter().find(|level| level.kind == kind)
    }

    /// The next-slowest [`MemoryLevelKind`] this profile declares above
    /// `level`, if any. `level` is `None` for "nothing decided yet" --
    /// the starting point before any `GraphConnecting`/`SetConnect`
    /// decision has assigned an edge to a real level -- in which case
    /// this returns the profile's own lowest declared level. `Some(level)`
    /// returns the next-slowest declared level above it, or `None` if
    /// `level` is already the top of this profile's hierarchy.
    ///
    /// Deliberately takes `Option<MemoryLevelKind>`, not a hardcoded
    /// [`MemoryLevelKind`] variant standing in for "nothing yet" (e.g.
    /// `Register`, which is ordered first today but isn't a *declared*
    /// level on most real profiles -- Triton exposes no register-level
    /// scheduling control): the type makes "no level decided" and "a
    /// real, named level" impossible to conflate, rather than relying on
    /// `Register` always staying the minimum of the enum's ordering.
    pub fn next_memory_level(&self, level: Option<MemoryLevelKind>) -> Option<MemoryLevelKind> {
        self.memory_levels
            .iter()
            .map(|memory_level| memory_level.kind)
            .filter(|&kind| level.is_none_or(|level| kind > level))
            .min()
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
                    capacity: 49_152,
                    bandwidth: None,
                    latency: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity: 12 * 1024 * 1024 * 1024,
                    bandwidth: Some(672.0e9),
                    latency: None,
                },
            ],
            execution: Some(ExecutionProfile {
                simt_width: 32,
                max_threads_per_group: 1024,
                max_groups_per_compute_unit: Some(16),
                max_grid_dims: [u32::MAX, 65_535, 65_535],
            }),
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
        assert_eq!(shared.capacity, 49_152);
    }

    #[test]
    fn level_returns_none_for_an_absent_kind() {
        let profile = profile();
        assert!(profile.level(MemoryLevelKind::L1Cache).is_none());
    }

    #[test]
    fn execution_profile_is_constructible_and_optional() {
        let profile = profile();
        let execution = profile
            .execution
            .expect("test profile declares an execution profile");
        assert_eq!(execution.simt_width, 32);
        assert_eq!(execution.max_threads_per_group, 1024);

        let scalar_cpu_profile = HardwareProfile {
            name: String::from("test-cpu"),
            compute_units: 4,
            memory_levels: vec![],
            execution: None,
        };
        assert!(scalar_cpu_profile.execution.is_none());
    }

    #[test]
    fn memory_level_kind_orders_fast_to_slow() {
        use MemoryLevelKind::*;
        assert!(Register < SharedMemory);
        assert!(SharedMemory < L1Cache);
        assert!(L1Cache < L2Cache);
        assert!(L2Cache < L3Cache);
        assert!(L3Cache < DeviceMemory);
        assert!(DeviceMemory < HostMemory);
    }

    #[test]
    fn memory_level_kind_sorts_into_hierarchy_order() {
        let mut kinds = alloc::vec![
            MemoryLevelKind::HostMemory,
            MemoryLevelKind::Register,
            MemoryLevelKind::L2Cache,
            MemoryLevelKind::DeviceMemory,
            MemoryLevelKind::SharedMemory,
        ];
        kinds.sort();
        assert_eq!(
            kinds,
            alloc::vec![
                MemoryLevelKind::Register,
                MemoryLevelKind::SharedMemory,
                MemoryLevelKind::L2Cache,
                MemoryLevelKind::DeviceMemory,
                MemoryLevelKind::HostMemory,
            ]
        );
    }
}
