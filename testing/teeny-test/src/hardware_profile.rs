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

//! Hand-calibrated [`HardwareProfile`]s for scheduler/cost-model tests that need a profile
//! without an open device (the queried path is
//! [`teeny_core::device::context::DeviceInfo::hardware_profile`]).
//!
//! The execution-profile fields come from the CUDA C Programming Guide's
//! per-compute-capability table rather than any single chip, and anything not actually
//! verified is left `None` rather than guessed -- the same "real hardware data or nothing"
//! rule the `teeny-cuda` driver follows.

use teeny_core::device::hardware::{
    ExecutionProfile, HardwareProfile, MemoryLevel, MemoryLevelKind,
};

/// A hand-calibrated [`HardwareProfile`] for an Ampere-class 6 GB GPU (e.g.
/// NVIDIA RTX A2000), with two memory levels: per-SM shared memory and
/// device (global) memory. For scheduler/cost-model tests that need a
/// profile without an open device.
pub fn orin_nano() -> HardwareProfile {
    HardwareProfile {
        name: "NVIDIA Jetson Orin Nano (Ampere, 8 GB)".to_string(),
        compute_units: 8,
        memory_levels: vec![
            MemoryLevel {
                kind: MemoryLevelKind::SharedMemory,
                // Conservative usable shared memory per kernel is ~96 KiB/SM on Orin Nano (some reserved by CUDA/L1)
                capacity: 96 * 1024,
                bandwidth: None,
                latency: None,
            },
            MemoryLevel {
                kind: MemoryLevelKind::DeviceMemory,
                capacity: 8 * 1024 * 1024 * 1024,
                // LPDDR5, real-world sustained bandwidth is lower, but theoretical up to ~102.4 GB/s
                // https://developer.nvidia.com/embedded/jetson-orin-nano-datasheet says LPDDR5 (LP-DDR5 128-bit 68.3 GB/s for 8GB model), but will use 68.3 GB/s
                bandwidth: Some(68_300_000_000.0),
                latency: None,
            },
        ],
        // Ampere (compute capability 8.x): `simt_width`/
        // `max_threads_per_group`/`max_grid_dims` are the CUDA C
        // Programming Guide's per-compute-capability technical
        // specifications table, unchanged across 8.0/8.6/8.7/8.9, not
        // Orin-specific -- same values as `hardware_profile_for`'s packaged
        // sm_80/sm_86/sm_87/sm_89 entries. `max_groups_per_compute_unit`
        // (max resident blocks/SM) does vary within 8.x and isn't verified
        // here, so it's left `None` rather than guessed.
        execution: Some(ExecutionProfile {
            simt_width: 32,
            max_threads_per_group: 1024,
            max_groups_per_compute_unit: None,
            max_grid_dims: [u32::MAX, 65_535, 65_535],
        }),
    }
}

/// Hand-calibrated [`HardwareProfile`] for an RTX 5070-class GPU (Blackwell,
/// sm_120), matching the values documented in
/// `teeny-kernels`'s `tests/test_fused_pointwise.rs` (`sharedMemPerBlock` =
/// 49152, `regs_per_block * 4` = 262144). Four declared memory levels like
/// `teeny_cuda::device::CudaDeviceInfo::hardware_profile`, so
/// `Anduin::schedule` exercises the same hierarchy the integration test
/// queries off a real device.
pub fn nvidia_rtx5070() -> HardwareProfile {
    HardwareProfile {
        name: "NVIDIA GeForce RTX 5070 (Blackwell)".to_string(),
        compute_units: 48,
        memory_levels: vec![
            MemoryLevel {
                kind: MemoryLevelKind::Register,
                capacity: 262_144,
                bandwidth: None,
                latency: None,
            },
            MemoryLevel {
                kind: MemoryLevelKind::SharedMemory,
                capacity: 49_152,
                bandwidth: None,
                latency: None,
            },
            MemoryLevel {
                kind: MemoryLevelKind::L2Cache,
                capacity: 64 * 1024 * 1024,
                bandwidth: None,
                latency: None,
            },
            MemoryLevel {
                kind: MemoryLevelKind::DeviceMemory,
                capacity: 12 * 1024 * 1024 * 1024,
                bandwidth: None,
                latency: None,
            },
        ],
        execution: Some(ExecutionProfile {
            simt_width: 32,
            max_threads_per_group: 1024,
            max_groups_per_compute_unit: None,
            max_grid_dims: [u32::MAX, 65_535, 65_535],
        }),
    }
}
