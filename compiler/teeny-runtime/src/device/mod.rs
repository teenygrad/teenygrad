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

use teeny_core::{
    dtype,
    nn::{Module, module::CompiledModule},
    tensor::Tensor,
};
#[cfg(feature = "cpu")]
use teeny_cpu::device::CpuDevice;

#[cfg(feature = "cuda")]
use teeny_cuda::device::CudaDevice;

use crate::error::{Error, Result};

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

    pub fn compile<N: dtype::Dtype, T: Tensor<N>, U: Tensor<N>>(
        &self,
        _module: Box<dyn Module<N, T, U, Err = Error>>,
    ) -> Result<Box<dyn CompiledModule<N, T, U, Err = Error>>> {
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
