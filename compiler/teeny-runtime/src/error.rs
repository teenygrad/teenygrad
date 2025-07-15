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

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Lock error: {0}")]
    TryLockError(String),

    #[error("No devices available")]
    NoDevicesAvailable,

    #[error("Core error: {0}")]
    CoreError(teeny_core::error::Error),

    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    CudaError(teeny_cuda::error::Error),

    #[cfg(feature = "cpu")]
    #[error("CPU driver error: {0}")]
    CpuError(teeny_cpu::error::Error),

    #[cfg(feature = "cpu")]
    #[error("CPU device error: {0}")]
    CpuDeviceError(teeny_cpu::error::Error),

    #[cfg(feature = "cuda")]
    #[error("CUDA device error: {0}")]
    CudaDeviceError(teeny_cuda::error::Error),
}
