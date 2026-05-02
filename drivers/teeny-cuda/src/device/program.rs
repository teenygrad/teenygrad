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

/// Kernel resource metadata parsed from `// meta:key=value` PTX comments
/// appended by the Triton CUDA backend during compilation.
#[derive(Debug, Default, Clone)]
pub(crate) struct KernelMetadata {
    pub(crate) name: String,
    pub(crate) num_warps: u32,
    pub(crate) num_ctas: u32,
    pub(crate) shared: u32,
    pub(crate) tmem_size: u32,
    pub(crate) global_scratch_size: u64,
    pub(crate) global_scratch_align: u64,
    pub(crate) profile_scratch_size: u32,
    pub(crate) profile_scratch_align: u32,
}

impl KernelMetadata {
    fn parse(ptx: &str) -> Self {
        let mut m = KernelMetadata { num_ctas: 1, global_scratch_align: 1, profile_scratch_align: 1, ..Default::default() };
        let mut reqntid: Option<u32> = None;

        for line in ptx.lines() {
            let trimmed = line.trim();

            // Parse Triton metadata block: `// meta:key=value`
            if let Some(rest) = trimmed.strip_prefix("// meta:") {
                if let Some((key, val)) = rest.split_once('=') {
                    match key {
                        "name"                  => m.name                  = val.to_owned(),
                        "num_warps"             => m.num_warps             = val.parse().unwrap_or(0),
                        "num_ctas"              => m.num_ctas              = val.parse().unwrap_or(1),
                        "shared"               => m.shared               = val.parse().unwrap_or(0),
                        "tmem_size"             => m.tmem_size             = val.parse().unwrap_or(0),
                        "global_scratch_size"   => m.global_scratch_size   = val.parse().unwrap_or(0),
                        "global_scratch_align"  => m.global_scratch_align  = val.parse().unwrap_or(1),
                        "profile_scratch_size"  => m.profile_scratch_size  = val.parse().unwrap_or(0),
                        "profile_scratch_align" => m.profile_scratch_align = val.parse().unwrap_or(1),
                        _ => {}
                    }
                }
                continue;
            }

            // Parse `.reqntid X` — PTX thread-count directive emitted by NVPTX backend.
            // Used as fallback when Triton metadata is absent (e.g. NVPTX-compiled kernels).
            if let Some(rest) = trimmed.strip_prefix(".reqntid ") {
                let x_str = rest.split(',').next().unwrap_or("").trim();
                if let Ok(x) = x_str.parse::<u32>() {
                    reqntid = Some(x);
                }
            }

            // Parse legacy Triton comment format emitted by older PTX cached before the
            // `// meta:` block was introduced. These act as low-priority fallbacks: a
            // later `// meta:` line for the same key will have already set the value, so
            // we only write when the field still holds its zero/unit default.
            if let Some(v) = trimmed.strip_prefix("// TRITON_SHARED_MEM_BYTES: ") {
                if m.shared == 0 {
                    m.shared = v.trim().parse().unwrap_or(0);
                }
            } else if let Some(v) = trimmed.strip_prefix("// TRITON_GLOBAL_SCRATCH_BYTES_PER_CTA: ") {
                if m.global_scratch_size == 0 {
                    m.global_scratch_size = v.trim().parse().unwrap_or(0);
                }
            } else if let Some(v) = trimmed.strip_prefix("// TRITON_GLOBAL_SCRATCH_ALIGN: ") {
                if m.global_scratch_align == 1 {
                    m.global_scratch_align = v.trim().parse::<u64>().unwrap_or(1).max(1);
                }
            }
        }

        // Fill name and num_warps from fallbacks for NVPTX-compiled kernels.
        if m.name.is_empty() {
            m.name = "entry_point".to_owned();
        }
        if m.num_warps == 0 {
            // Derive num_warps from .reqntid (round up to full warps).
            m.num_warps = reqntid.unwrap_or(128).div_ceil(32);
        }

        m
    }

    /// Threads per block, derived from num_warps (CUDA warp size is always 32).
    pub(crate) fn threads_per_block(&self) -> u32 {
        self.num_warps * 32
    }
}

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
    /// Kernel resource metadata parsed from `// meta:key=value` PTX comments.
    pub(crate) metadata: KernelMetadata,
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
    ///
    /// Metadata is not available from a cubin; default values are used.
    pub fn try_new(cubin: &[u8], entry_point: &str) -> Result<Self> {
        let mut module = cuda::CUmodule::default();
        let status = unsafe { cuda::cuModuleLoadData(&mut module, cubin.as_ptr().cast()) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        Self::resolve_function(module, entry_point, KernelMetadata::default())
    }

    /// JIT-compile PTX source via the CUDA driver.
    ///
    /// The entry-point function name and all resource metadata are read from
    /// the `// meta:key=value` block that the Triton CUDA backend appends to
    /// every PTX output. `ptx` must be ASCII PTX text; a null terminator is
    /// appended automatically.
    pub fn try_from_ptx(ptx: &[u8]) -> Result<Self> {
        let ptx_str = std::str::from_utf8(ptx).unwrap_or("");
        let metadata = KernelMetadata::parse(ptx_str);

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

        Self::resolve_function(module, &metadata.name.clone(), metadata)
    }

    fn resolve_function(
        module: cuda::CUmodule,
        entry_point: &str,
        metadata: KernelMetadata,
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
            metadata,
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
