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

use teeny_core::{dtype, tensor::shape};

use crate::current_device;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

#[cfg(feature = "cpu")]
mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

pub fn zeros<N: dtype::Dtype, S: shape::Shape>(shape: S) -> Result<Tensor<N>> {
    let device = current_device()?.ok_or(Error::NoDevicesAvailable)?;

    match device.as_ref() {
        Device::Cpu(_) => cpu::zeros(shape).map(|t| Tensor::Cpu(t)),
        Device::Cuda(_) => cuda::zeros(shape).map(|t| Tensor::Cuda(t)),
    }
}
