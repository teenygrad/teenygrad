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

use crate::target::CudaTarget;

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
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub concurrent_kernels: i32,
    pub target: CudaTarget,
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub id: String,
    pub name: String,
    pub properties: DeviceProperties,
}
