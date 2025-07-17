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

#[cfg(feature = "cpu")]
use teeny_cpu::device::CpuDevice;

#[cfg(feature = "cuda")]
use teeny_cuda::device::CudaDevice;

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
}
