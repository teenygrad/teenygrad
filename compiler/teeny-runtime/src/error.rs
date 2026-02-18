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

pub type Result<T> = anyhow::Result<T>;

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
