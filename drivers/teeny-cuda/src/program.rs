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

use std::ffi::CString;
use std::marker::PhantomData;

use teeny_core::context::program::{Kernel, Program};

use crate::cuda;
use crate::errors::{Error, Result};

/// Strip DWARF debug sections from PTX source.
///
/// The Rust NVPTX backend emits PTX object files that include DWARF debug
/// sections with relocation references (e.g. `.b32 .debug_abbrev`). These
/// relocations are resolved by a PTX linker but are invalid when passing PTX
/// directly to `cuModuleLoadData` for driver JIT compilation.
///
/// This function truncates the PTX at the first `.file` or `.section .debug`
/// directive, which always appears after the kernel body.
fn strip_debug_sections(ptx: &[u8]) -> &[u8] {
    let mut pos = 0;
    while pos < ptx.len() {
        let line_end = ptx[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|i| pos + i)
            .unwrap_or(ptx.len());

        let line = &ptx[pos..line_end];
        let trim_start = line
            .iter()
            .position(|&b| b != b' ' && b != b'\t')
            .unwrap_or(line.len());
        let trimmed = &line[trim_start..];

        // .file directives and .section .debug_* mark the start of DWARF content
        if trimmed.starts_with(b".file")
            || (trimmed.starts_with(b".section") && line.windows(7).any(|w| w == b".debug_"))
        {
            return &ptx[..pos];
        }

        pos = line_end + 1;
    }
    ptx
}

/// A loaded CUDA program: the cubin is loaded into a `CUmodule` and the
/// entry-point function is resolved to a `CUfunction` ready to launch.
pub struct CudaProgram<'a, K: Kernel> {
    pub(crate) module: cuda::CUmodule,
    pub(crate) function: cuda::CUfunction,
    _unused: PhantomData<&'a ()>,
    _kernel: PhantomData<K>,
}

impl<'a, K: Kernel> CudaProgram<'a, K> {
    pub fn module_ptr(&self) -> usize {
        self.module as usize
    }
    pub fn function_ptr(&self) -> usize {
        self.function as usize
    }

    /// Load a cubin image into the current CUDA context and resolve `entry_point`.
    pub fn try_new(cubin: &[u8], entry_point: &str) -> Result<Self> {
        let mut module = cuda::CUmodule::default();
        let status = unsafe { cuda::cuModuleLoadData(&mut module, cubin.as_ptr().cast()) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        Self::resolve_function(module, entry_point)
    }

    /// JIT-compile PTX source via the CUDA driver and resolve `entry_point`.
    ///
    /// This bypasses the nvPTX compiler library and lets the driver JIT the PTX
    /// directly, which is more robust across CUDA toolkit versions.
    /// `ptx` must be ASCII PTX text; a null terminator is appended automatically.
    pub fn try_from_ptx(ptx: &[u8], entry_point: &str) -> Result<Self> {
        // Strip DWARF debug sections — they contain relocations (.b32 .debug_abbrev)
        // that are only valid in linked PTX object files, not for driver JIT.
        let ptx = strip_debug_sections(ptx);

        // cuModuleLoadData expects a null-terminated string for PTX input.
        let mut ptx_ntstr = ptx.to_vec();
        ptx_ntstr.push(0);

        let mut module = cuda::CUmodule::default();
        let status = unsafe { cuda::cuModuleLoadData(&mut module, ptx_ntstr.as_ptr().cast()) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        Self::resolve_function(module, entry_point)
    }

    fn resolve_function(module: cuda::CUmodule, entry_point: &str) -> Result<Self> {
        let name = CString::new(entry_point).map_err(Error::CStringError)?;
        let mut function = cuda::CUfunction::default();
        let status = unsafe { cuda::cuModuleGetFunction(&mut function, module, name.as_ptr()) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            unsafe { cuda::cuModuleUnload(module) };
            return Err(Error::from_cuda_error(status).into());
        }

        Ok(Self {
            module,
            function,
            _unused: PhantomData,
            _kernel: PhantomData,
        })
    }
}

impl<'a, K: Kernel> Drop for CudaProgram<'a, K> {
    fn drop(&mut self) {
        let status = unsafe { cuda::cuModuleUnload(self.module) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            eprintln!("Failed to unload CUDA module: {}", status);
        }
    }
}

impl<'a, K: Kernel> Program<'a, K> for CudaProgram<'a, K> {}
