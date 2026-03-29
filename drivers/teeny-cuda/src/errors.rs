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

pub type Result<T> = anyhow::Result<T>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("CUDA not available")]
    CudaNotAvailable,

    #[error("CUDA error: {0}")]
    CudaError(cuda::cudaError_enum),

    #[error("Unknown capability: {0}")]
    UnknownCapability(String),

    #[error("CString error: {0}")]
    CStringError(std::ffi::NulError),

    #[error("NVPTX Compile error: {0}")]
    NvptxCompileError(cuda::nvPTXCompileResult),
}
