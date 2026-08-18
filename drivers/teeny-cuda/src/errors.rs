/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

/// `teeny-cuda`'s result alias.
pub type Result<T> = anyhow::Result<T>;

/// Errors produced by `teeny-cuda`.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// No CUDA-capable device/driver was found.
    #[error("CUDA not available")]
    CudaNotAvailable,

    /// A CUDA runtime/driver API call failed.
    #[error("CUDA error: {code} ({message})")]
    CudaError {
        /// The CUDA error code.
        code: cuda::cudaError_enum,
        /// The human-readable message from `cudaGetErrorString`.
        message: String,
    },

    /// A capability string didn't match any known GPU architecture.
    #[error("Unknown capability: {0}")]
    UnknownCapability(String),

    /// A Rust string contained an interior NUL byte and couldn't convert to a C string.
    #[error("CString error: {0}")]
    CStringError(std::ffi::NulError),

    /// `nvptxcompiler` failed to compile PTX.
    #[error("NVPTX Compile error {code}: {log}")]
    NvptxCompileError {
        /// The `nvptxcompiler` result code.
        code: cuda::nvPTXCompileResult,
        /// The compiler's error log.
        log: String,
    },

    /// A source buffer had more elements than the destination buffer could hold.
    #[error("buffer overflow: source has {src} elements but buffer holds {buf}")]
    BufferOverflow {
        /// Number of elements in the source.
        src: usize,
        /// Capacity of the destination buffer.
        buf: usize,
    },

    /// A `--options`-style compiler options string ([`crate::compiler::options::Options::parse`])
    /// was malformed.
    #[error("invalid compiler options '{input}': {reason}")]
    InvalidOptions {
        /// The original, unparsed options string.
        input: String,
        /// Why parsing failed.
        reason: String,
    },
}

impl Error {
    /// Builds a [`Error::CudaError`] from a raw CUDA error code, looking up its message via
    /// `cudaGetErrorString`.
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
