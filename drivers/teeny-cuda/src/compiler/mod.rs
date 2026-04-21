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

use std::ffi::{c_char, c_void};
use std::ptr;

use teeny_core::context::program::Kernel;

use crate::compiler::options::Options;
use crate::cuda;
use crate::device::program::CudaProgram;
use crate::errors::{Error, Result};

pub mod options;
pub mod target;

pub struct PtxCompiler {
    compiler: cuda::nvPTXCompilerHandle,
}

impl PtxCompiler {
    /// `ptx` is the raw PTX source bytes (ASCII; the C API takes a byte pointer
    /// and length, so no UTF-8 validation or null-termination is required).
    pub fn try_new(ptx: &[u8]) -> Result<Self> {
        let mut compiler = cuda::nvPTXCompilerHandle::default();
        let result = unsafe {
            cuda::nvPTXCompilerCreate(&mut compiler, ptx.len(), ptx.as_ptr().cast::<c_char>())
        };

        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS {
            return Err(Error::NvptxCompileError {
                code: result,
                log: String::new(),
            }
            .into());
        }
        Ok(PtxCompiler { compiler })
    }

    pub fn compile(&mut self, options: &Options) -> Result<Vec<u8>> {
        let compile_options = options.to_compile_options();
        let num_options = compile_options.len() as i32;
        eprintln!("[nvPTX] compile options: {compile_options:?}");

        // Convert Vec<String> to Vec<*const c_char>
        let cstrs: Vec<std::ffi::CString> = compile_options
            .iter()
            .map(|s| std::ffi::CString::new(s.as_str()).map_err(|e| Error::CStringError(e).into()))
            .collect::<Result<Vec<std::ffi::CString>>>()?;
        let cptrs: Vec<*const c_char> = cstrs.iter().map(|cs| cs.as_ptr()).collect();

        let result =
            unsafe { cuda::nvPTXCompilerCompile(self.compiler, num_options, cptrs.as_ptr()) };

        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS {
            // Retrieve the compiler error log for a human-readable message.
            let log = self.error_log();
            return Err(Error::NvptxCompileError { code: result, log }.into());
        }

        let mut binary_size = 0usize;
        let result =
            unsafe { cuda::nvPTXCompilerGetCompiledProgramSize(self.compiler, &mut binary_size) };
        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS {
            return Err(Error::NvptxCompileError {
                code: result,
                log: String::new(),
            }
            .into());
        }

        let mut binary = vec![0u8; binary_size];
        let result = unsafe {
            cuda::nvPTXCompilerGetCompiledProgram(
                self.compiler,
                binary.as_mut_ptr().cast::<c_void>(),
            )
        };
        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS {
            return Err(Error::NvptxCompileError {
                code: result,
                log: String::new(),
            }
            .into());
        }

        Ok(binary)
    }

    fn error_log(&self) -> String {
        let mut log_size = 0usize;
        let result = unsafe { cuda::nvPTXCompilerGetErrorLogSize(self.compiler, &mut log_size) };
        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS || log_size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; log_size];
        let result = unsafe {
            cuda::nvPTXCompilerGetErrorLog(self.compiler, buf.as_mut_ptr().cast::<c_char>())
        };
        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS {
            return String::new();
        }
        String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string()
    }

    /// Compile the PTX to a cubin and load it into the current CUDA context,
    /// returning a ready-to-launch `CudaProgram`. The entry-point symbol is
    /// always `"entry_point"` (the name emitted by the `#[kernel]` proc macro).
    pub fn compile_program<K: Kernel>(
        &mut self,
        options: &Options,
    ) -> Result<CudaProgram<'static, K>> {
        let cubin = self.compile(options)?;
        CudaProgram::try_new(&cubin, options.entry.as_str())
    }
}

impl Drop for PtxCompiler {
    fn drop(&mut self) {
        let result = unsafe { cuda::nvPTXCompilerDestroy(ptr::addr_of_mut!(self.compiler)) };
        if result != cuda::nvPTXCompileResult_NVPTXCOMPILE_SUCCESS {
            eprintln!("Failed to destroy NVPTX compiler: {}", result);
        }
    }
}
