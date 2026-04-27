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

use teeny_core::device::program::{Kernel, Program};

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
    /// Dynamic shared memory bytes required by the kernel (from TRITON_SHARED_MEM_BYTES).
    pub(crate) shared_mem_bytes: u32,
    /// Per-CTA global scratch memory required for TMA descriptors (TRITON_GLOBAL_SCRATCH_BYTES_PER_CTA).
    pub(crate) global_scratch_bytes_per_cta: u64,
    /// Alignment for global scratch buffer (TRITON_GLOBAL_SCRATCH_ALIGN).
    pub(crate) global_scratch_align: u64,
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

        Self::resolve_function(module, entry_point, 0, 0, 1)
    }

    /// JIT-compile PTX source via the CUDA driver and resolve `entry_point`.
    ///
    /// This bypasses the nvPTX compiler library and lets the driver JIT the PTX
    /// directly, which is more robust across CUDA toolkit versions.
    /// `ptx` must be ASCII PTX text; a null terminator is appended automatically.
    pub fn try_from_ptx(ptx: &[u8], entry_point: &str) -> Result<Self> {
        // Extract dynamic shared memory size from TRITON_SHARED_MEM_BYTES comment,
        // which is prepended by our Triton C++ backend to the PTX text.
        fn parse_ptx_meta(ptx: &[u8], key: &[u8]) -> u64 {
            ptx.windows(key.len())
                .position(|w| w == key)
                .and_then(|pos| {
                    let rest = &ptx[pos + key.len()..];
                    let end = rest.iter().position(|&b| b == b'\n').unwrap_or(rest.len());
                    std::str::from_utf8(&rest[..end])
                        .ok()
                        .and_then(|s| s.trim().parse::<u64>().ok())
                })
                .unwrap_or(0)
        }

        let shared_mem_bytes = parse_ptx_meta(ptx, b"// TRITON_SHARED_MEM_BYTES: ") as u32;
        let global_scratch_bytes_per_cta =
            parse_ptx_meta(ptx, b"// TRITON_GLOBAL_SCRATCH_BYTES_PER_CTA: ");
        let global_scratch_align = parse_ptx_meta(ptx, b"// TRITON_GLOBAL_SCRATCH_ALIGN: ").max(1);
        eprintln!(
            "[CUDA-JIT] dynamic shared mem = {} bytes, global scratch = {} bytes/CTA (align {})",
            shared_mem_bytes, global_scratch_bytes_per_cta, global_scratch_align
        );

        // Strip DWARF debug sections — they contain relocations (.b32 .debug_abbrev)
        // that are only valid in linked PTX object files, not for driver JIT.
        let ptx = strip_debug_sections(ptx);

        // cuModuleLoadData expects a null-terminated string for PTX input.
        let mut ptx_ntstr = ptx.to_vec();
        ptx_ntstr.push(0);

        // Use cuModuleLoadDataEx with error logging only. Do NOT override CU_JIT_TARGET:
        // our PTX targets sm_90a (Hopper with TMA), and Blackwell runs sm_90a code in
        // forward-compatible mode. Forcing CU_JIT_TARGET=120 causes ptxas to attempt a
        // cross-architecture compile that it refuses ("cannot be compiled to future
        // architectures"). Without a target override, the CUDA driver uses the PTX's
        // declared target and JITs it for the current device automatically.
        const LOG_SIZE: usize = 65536;
        let mut error_log = vec![0u8; LOG_SIZE];
        let mut info_log = vec![0u8; LOG_SIZE];

        // CUDA cuModuleLoadDataEx option values:
        // - Buffer pointers: pass as pointer (void*)
        // - Size/enum values: pass as the VALUE itself cast to void* (not a pointer to value)
        let mut options = [
            cuda::CUjit_option_enum_CU_JIT_ERROR_LOG_BUFFER,
            cuda::CUjit_option_enum_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            cuda::CUjit_option_enum_CU_JIT_INFO_LOG_BUFFER,
            cuda::CUjit_option_enum_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        ];
        #[allow(clippy::cast_ptr_alignment)]
        let mut option_values: [*mut std::ffi::c_void; 4] = [
            error_log.as_mut_ptr().cast(),
            LOG_SIZE as *mut std::ffi::c_void, // size value, not a pointer
            info_log.as_mut_ptr().cast(),
            LOG_SIZE as *mut std::ffi::c_void, // size value, not a pointer
        ];

        let mut module = cuda::CUmodule::default();
        let status = unsafe {
            cuda::cuModuleLoadDataEx(
                &mut module,
                ptx_ntstr.as_ptr().cast(),
                4,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            )
        };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            let err_len = error_log.iter().position(|&b| b == 0).unwrap_or(LOG_SIZE);
            let error_str = std::str::from_utf8(&error_log[..err_len]).unwrap_or("<invalid utf8>");
            eprintln!("[CUDA-JIT] error log: {}", error_str);
            return Err(Error::from_cuda_error(status).into());
        }

        let info_len = info_log.iter().position(|&b| b == 0).unwrap_or(0);
        if info_len > 0 {
            let info_str = std::str::from_utf8(&info_log[..info_len]).unwrap_or("<invalid utf8>");
            eprintln!("[CUDA-JIT] info: {}", info_str);
        }

        Self::resolve_function(
            module,
            entry_point,
            shared_mem_bytes,
            global_scratch_bytes_per_cta,
            global_scratch_align,
        )
    }

    fn resolve_function(
        module: cuda::CUmodule,
        entry_point: &str,
        shared_mem_bytes: u32,
        global_scratch_bytes_per_cta: u64,
        global_scratch_align: u64,
    ) -> Result<Self> {
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
            shared_mem_bytes,
            global_scratch_bytes_per_cta,
            global_scratch_align,
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

/// A dummy `Kernel` marker used to load PTX without a concrete kernel type.
/// Enables `CudaProgram<ErasedKernel>` for type-erased model execution.
pub struct ErasedKernel;

impl Kernel for ErasedKernel {
    type Args<'a> = ();
    fn name(&self) -> &str { "" }
    fn source(&self) -> &str { "" }
    fn kernel_source(&self) -> &str { "" }
    fn entry_point(&self) -> &str { "" }
}
