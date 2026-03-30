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

    #[error("CUDA error: {code} ({message})")]
    CudaError {
        code: cuda::cudaError_enum,
        message: String,
    },

    #[error("Unknown capability: {0}")]
    UnknownCapability(String),

    #[error("CString error: {0}")]
    CStringError(std::ffi::NulError),

    #[error("NVPTX Compile error {code}: {log}")]
    NvptxCompileError {
        code: cuda::nvPTXCompileResult,
        log: String,
    },

    #[error("buffer overflow: source has {src} elements but buffer holds {buf}")]
    BufferOverflow { src: usize, buf: usize },
}

impl Error {
    pub fn from_cuda_error(code: cuda::cudaError_enum) -> Self {
        // SAFETY: cudaGetErrorString returns a valid C string for any cudaError_enum value.
        let err_str = unsafe {
            let ptr = cuda::cudaGetErrorString(code);
            if ptr.is_null() {
                "<unknown CUDA error>"
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("<invalid utf8 CUDA error>")
            }
        }
        .to_owned();

        Error::CudaError {
            code,
            message: err_str,
        }
    }
}
