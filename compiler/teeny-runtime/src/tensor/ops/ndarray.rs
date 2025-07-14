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

use ndarray::{Array, IxDyn};
use teeny_core::tensor::num;

#[cfg(feature = "cpu")]
use teeny_cpu::tensor::CpuTensor;

#[cfg(feature = "cuda")]
use teeny_cuda::tensor::CudaTensor;

use crate::{current_device, device::Device, error::RuntimeError, tensor::Tensor};

impl<T: num::Num> TryFrom<Array<T, IxDyn>> for Tensor<T> {
    type Error = RuntimeError;

    fn try_from(array: Array<T, IxDyn>) -> Result<Self, Self::Error> {
        let device = current_device()?;
        let device = device.ok_or(RuntimeError::NoDevicesAvailable)?;

        match *device {
            #[cfg(feature = "cpu")]
            Device::Cpu(_) => Ok(Tensor::Cpu(CpuTensor::from(array))),

            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Ok(Tensor::Cuda(CudaTensor::from(array))),
        }
    }
}
