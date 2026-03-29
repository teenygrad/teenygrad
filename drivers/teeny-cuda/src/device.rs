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

use std::marker::PhantomData;

use teeny_core::{
    context::{
        DeviceInfo,
        device::{Device, LaunchConfig},
        program::Kernel,
    },
    dtype::Dtype,
};

use crate::{
    buffer::CudaBuffer,
    cuda,
    errors::{Error, Result},
    program::CudaProgram,
};

pub struct CudaLaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: [u32; 3],
}

impl LaunchConfig for CudaLaunchConfig {}

#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub id: i32,
    pub name: String,
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
}

impl DeviceInfo for CudaDeviceInfo {
    type Id = i32;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct CudaDevice<'a> {
    device: cuda::CUdevice,
    context: cuda::CUcontext,
    _unused: PhantomData<&'a ()>,
}

impl<'a> CudaDevice<'a> {
    pub fn try_new(id: i32) -> Result<Self> {
        let device_id = id;
        let mut device = cuda::CUdevice::default();
        let status = unsafe { cuda::cuDeviceGet(&mut device, device_id) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::CudaError(status).into());
        }

        let mut context = cuda::CUcontext::default();
        let mut params = cuda::CUctxCreateParams::default();
        let flags = 0;
        let status = unsafe { cuda::cuCtxCreate_v4(&mut context, &mut params, flags, device) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::CudaError(status).into());
        }

        Ok(Self {
            device,
            context,
            _unused: PhantomData,
        })
    }
}

impl<'a> Drop for CudaDevice<'a> {
    fn drop(&mut self) {
        let result = unsafe { cuda::cuCtxDestroy_v2(self.context) };
        if result != cuda::cudaError_enum_CUDA_SUCCESS {
            // just log, we can't do anything about it
            eprintln!("Failed to destroy CUDA context: {}", result);
        }
    }
}

impl<'a> Device<'a> for CudaDevice<'a> {
    type Buffer<D: Dtype> = CudaBuffer<'a, D>;
    type Program<K: teeny_core::context::program::Kernel> = CudaProgram<'a, K>;
    type LaunchConfig = CudaLaunchConfig;

    fn buffer<D: Dtype>(&self, _size: &[usize]) -> teeny_core::errors::Result<Self::Buffer<D>> {
        todo!()
    }

    fn launch<K: Kernel>(
        &self,
        _program: &Self::Program<K>,
        _cfg: &Self::LaunchConfig,
        _args: K::Args<'a>,
    ) -> teeny_core::errors::Result<()> {
        todo!()
    }
}
