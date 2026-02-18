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

use teeny_core::{
    dtype,
    nn::{Module, module::CompiledModule},
};
#[cfg(feature = "cpu")]
use teeny_cpu::device::CpuDevice;

#[cfg(feature = "cuda")]
use teeny_cuda::device::CudaDevice;

use crate::error::Result;

#[derive(Debug, Clone)]
pub enum Device {
    #[cfg(feature = "cpu")]
    Cpu(CpuDevice),

    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
}

impl Device {
    pub fn id(&self) -> String {
        match self {
            #[cfg(feature = "cpu")]
            Device::Cpu(device) => device.id.clone(),

            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.id.clone(),
        }
    }

    pub fn name(&self) -> String {
        match self {
            #[cfg(feature = "cpu")]
            Device::Cpu(device) => device.name.clone(),

            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.name.clone(),
        }
    }

    pub fn compile<N: dtype::Dtype, T, U>(
        &self,
        _module: Box<dyn Module<T, U>>,
    ) -> Result<Box<dyn CompiledModule<T, U>>> {
        todo!()
    }
}

/*-------------------------------- CPU --------------------------------*/

#[cfg(feature = "cpu")]
mod cpu;

#[cfg(feature = "cpu")]
pub use cpu::find_cpu_devices;

#[cfg(not(feature = "cpu"))]
pub fn find_cpu_devices() -> Result<Vec<Device>> {
    Ok(vec![])
}

/*-------------------------------- CUDA --------------------------------*/

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::find_cuda_devices;

#[cfg(not(feature = "cuda"))]
pub fn find_cuda_devices() -> Result<Vec<Device>> {
    Ok(vec![])
}
