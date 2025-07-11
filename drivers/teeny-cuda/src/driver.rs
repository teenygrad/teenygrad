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

use std::sync::{Arc, Mutex};

use ctor::ctor;
use teeny_driver::device::{Device, DeviceType, NvidiaGpu};
use teeny_driver::driver::{CUDA_DRIVER_ID, Driver};
use teeny_driver::driver_manager::DriverManager;
use teeny_driver::error::{DriverError, Result};

use crate::cuda;
use crate::device::CudaDevice;

#[derive(Default)]
pub struct CudaDriver {}

impl CudaDriver {
    pub const fn new() -> Self {
        Self {}
    }
}

impl Driver for CudaDriver {
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn deinit(&mut self) -> Result<()> {
        // no-op
        Ok(())
    }

    fn id(&self) -> &str {
        CUDA_DRIVER_ID
    }

    fn name(&self) -> &str {
        "Teenygrad CUDA Driver v0.1.0"
    }

    fn devices(&self) -> Result<Vec<Arc<Mutex<dyn Device>>>> {
        let mut devices = Vec::new();
        let mut device_count = 0;
        let err = unsafe { cuda::cudaGetDeviceCount(&mut device_count) };
        if err != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(DriverError::CudaError(err));
        }

        for i in 0..device_count {
            let mut props = cuda::cudaDeviceProp::default();
            let err = unsafe { cuda::cudaGetDeviceProperties_v2(&mut props, i) };
            if err != cuda::cudaError_enum_CUDA_SUCCESS {
                return Err(DriverError::CudaError(err));
            }

            let name = unsafe { std::ffi::CStr::from_ptr(props.name.as_ptr()) };
            let name = name.to_string_lossy().to_string();

            let device = CudaDevice {
                id: format!("cuda:{}", i),
                name,
                device_type: DeviceType::NvidiaGpu(NvidiaGpu {
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
                    clock_rate: props.clockRate,
                    memory_clock_rate: props.memoryClockRate,
                    memory_bus_width: props.memoryBusWidth,
                    l2_cache_size: props.l2CacheSize,
                    concurrent_kernels: props.concurrentKernels,
                    compute_mode: props.computeMode,
                }),
            };
            devices.push(Arc::new(Mutex::new(device)) as Arc<Mutex<dyn Device>>);
        }

        println!("Found {} CUDA devices", devices.len());
        for device in devices.iter() {
            println!("Device: {:?}", device.lock().unwrap().id());
        }

        Ok(devices)
    }
}

#[ctor]
fn register_cuda() {
    DriverManager::register(Arc::new(Mutex::new(CudaDriver::new())));
}
