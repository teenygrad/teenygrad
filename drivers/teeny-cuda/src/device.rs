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

use teeny_core::device::Device;

use crate::target::Target;

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub major: i32,
    pub minor: i32,
    pub multi_processor_count: i32,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multi_processor: i32,
    pub max_blocks_per_multi_processor: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub concurrent_kernels: i32,
    pub compute_mode: i32,
    pub target: Target,
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub id: String,
    pub name: String,
    pub properties: DeviceProperties,
}

impl Device for CudaDevice {}
