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

use crate::cuda;
use crate::device::CudaDevice;

use crate::device::DeviceProperties;
use crate::error::Error;
use crate::error::Result;
use crate::target::CudaTarget;

#[derive(Debug)]
pub struct CudaDriver;

impl CudaDriver {
    pub fn devices() -> Result<Vec<CudaDevice>> {
        let mut devices = Vec::new();
        let mut device_count = 0;
        let err = unsafe { cuda::cudaGetDeviceCount(&mut device_count) };
        if err != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::CudaError(err).into());
        }

        for i in 0..device_count {
            let mut props = cuda::cudaDeviceProp::default();
            let err = unsafe { cuda::cudaGetDeviceProperties(&mut props, i) };
            if err != cuda::cudaError_enum_CUDA_SUCCESS {
                return Err(Error::CudaError(err).into());
            }

            let name = unsafe { std::ffi::CStr::from_ptr(props.name.as_ptr()) };
            let name = name.to_string_lossy().to_string();

            let device = CudaDevice {
                id: format!("cuda:{i}"),
                name,
                properties: DeviceProperties {
                    major: props.major,
                    minor: props.minor,
                    multi_processor_count: props.multiProcessorCount,
                    total_global_mem: props.totalGlobalMem,
                    shared_mem_per_block: props.sharedMemPerBlock,
                    regs_per_block: props.regsPerBlock,
                    warp_size: props.warpSize,
                    max_threads_per_block: props.maxThreadsPerBlock,
                    max_threads_per_multi_processor: props.maxThreadsPerMultiProcessor,
                    max_blocks_per_multi_processor: props.maxBlocksPerMultiProcessor,
                    max_threads_dim: props.maxThreadsDim,
                    max_grid_size: props.maxGridSize,
                    memory_bus_width: props.memoryBusWidth,
                    l2_cache_size: props.l2CacheSize,
                    concurrent_kernels: props.concurrentKernels,
                    target: CudaTarget::try_from((props.major, props.minor))?,
                },
            };
            devices.push(device);
        }

        Ok(devices)
    }
}
