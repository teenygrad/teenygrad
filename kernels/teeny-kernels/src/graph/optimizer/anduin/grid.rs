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

//! Welder's common thread-block-size rule (teenygrad-1nr.17, §4.1 "Decide
//! aligned computation parallelism"): the shared thread count every fused
//! op in a `VirtualNode` binds its own per-thread tile to, derived from
//! their tile sizes' GCD and clamped to real hardware bounds via
//! [`ExecutionProfile`] (teenygrad-1nr.18).
//!
//! This is deliberately narrow: it answers "how many threads does the
//! fused kernel launch with," not "what CTA grid -- whose iteration space
//! -- governs it." That's the still-open, harder half of teenygrad-1nr.17:
//! reconciling structurally different native grids (e.g. conv2d's
//! `(b, c_out, oh, ow_tile)` vs. batchnorm2d's `(c, b)`) needs either a
//! from-scratch symbolic index-remapping layer or a new MLIR-level rewrite
//! pass (Welder's own `RewriteOutputPass`/`RewriteInputPass` analog) -- see
//! that issue's own investigation comments. Not attempted here.

use teeny_core::device::hardware::ExecutionProfile;

/// Euclid's algorithm.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Welder's common thread-block-size rule: the greatest common divisor of
/// every fused op's own tile size (elements/threads it wants per block),
/// clamped to `[occupancy_multiplier * execution.simt_width,
/// execution.max_threads_per_group]` -- "the hardware parallelism (e.g.,
/// 128)... the maximum limitation (e.g., 1024)" in Welder's own words,
/// with the "128" expressed portably as a multiplier of the real device's
/// warp/wavefront width (4 warps on CUDA's 32-wide warp; 4 wavefronts
/// would be 256 on an AMD wave64 device) rather than a hardcoded constant
/// -- see [`ExecutionProfile`]'s own doc comment for why that number isn't
/// stored there.
///
/// An op whose own tile size isn't an exact multiple of the resulting
/// common size runs some threads over more than one tile (Welder's TVM
/// "virtual thread" mechanism, §4.1) -- computing that per-thread tile
/// count is the caller's job, not this function's; it only picks the
/// shared size.
///
/// Callers should keep `occupancy_multiplier * execution.simt_width <=
/// execution.max_threads_per_group` -- every real profile in this
/// codebase does (e.g. 4 * 32 = 128 well under CUDA's 1024) -- since a
/// floor above the ceiling makes the clamp's result caller-dependent
/// rather than a real answer.
///
/// # Panics
/// Panics if `tile_sizes` is empty -- a virtual node with no constituent
/// ops has no thread-block size to compute -- or if
/// `occupancy_multiplier * execution.simt_width` overflows `u32` or
/// exceeds `execution.max_threads_per_group` (see above).
pub fn common_thread_block_size(
    tile_sizes: &[u32],
    execution: &ExecutionProfile,
    occupancy_multiplier: u32,
) -> u32 {
    let common = tile_sizes
        .iter()
        .copied()
        .reduce(gcd)
        .expect("common_thread_block_size: tile_sizes must be non-empty");
    common.clamp(
        occupancy_multiplier * execution.simt_width,
        execution.max_threads_per_group,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_like_execution() -> ExecutionProfile {
        ExecutionProfile {
            simt_width: 32,
            max_threads_per_group: 1024,
            max_groups_per_compute_unit: Some(16),
            max_grid_dims: [u32::MAX, 65_535, 65_535],
        }
    }

    #[test]
    fn gcd_within_bounds_is_returned_unchanged() {
        // gcd(256, 384) = 128 -- already >= 4*32 and <= 1024.
        let execution = cuda_like_execution();
        assert_eq!(common_thread_block_size(&[256, 384], &execution, 4), 128);
    }

    #[test]
    fn welders_own_worked_example() {
        // Welder's own text: "128 (4 warps)" as the floor, "1024" as the
        // ceiling, on a 32-wide warp -- confirm the floor is exactly what
        // the paper's example says.
        let execution = cuda_like_execution();
        assert_eq!(4 * execution.simt_width, 128);
        assert_eq!(execution.max_threads_per_group, 1024);
    }

    #[test]
    fn gcd_below_the_occupancy_floor_is_clamped_up() {
        // gcd(64, 96) = 32 -- below the 128-thread floor (4 warps).
        let execution = cuda_like_execution();
        assert_eq!(common_thread_block_size(&[64, 96], &execution, 4), 128);
    }

    #[test]
    fn gcd_above_the_hardware_ceiling_is_clamped_down() {
        // gcd(2048, 4096) = 2048 -- above CUDA's real 1024-thread limit.
        let execution = cuda_like_execution();
        assert_eq!(common_thread_block_size(&[2048, 4096], &execution, 4), 1024);
    }

    #[test]
    fn a_single_tile_size_is_its_own_gcd() {
        let execution = cuda_like_execution();
        assert_eq!(common_thread_block_size(&[256], &execution, 4), 256);
    }

    #[test]
    fn an_amd_wave64_floor_is_double_cudas() {
        // Same occupancy multiplier (4), a wider SIMT group -- the floor
        // portably derives from simt_width rather than being a hardcoded
        // 128, exactly the reason ExecutionProfile stores no minimum
        // itself (see this module's own doc comment).
        let execution = ExecutionProfile {
            simt_width: 64,
            max_threads_per_group: 1024,
            max_groups_per_compute_unit: None,
            max_grid_dims: [u32::MAX, 65_535, 65_535],
        };
        assert_eq!(common_thread_block_size(&[16, 48], &execution, 4), 256);
    }

    #[test]
    #[should_panic(expected = "tile_sizes must be non-empty")]
    fn empty_tile_sizes_panics() {
        let execution = cuda_like_execution();
        common_thread_block_size(&[], &execution, 4);
    }
}
