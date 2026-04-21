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

use std::marker::PhantomData;

use teeny_core::{
    context::{
        DeviceInfo,
        device::{Device, LaunchConfig},
        program::{ArgVisitor, Kernel, KernelArgs},
    },
    dtype::Num,
};

use crate::{
    cuda,
    device::buffer::CudaBuffer,
    device::program::CudaProgram,
    errors::{Error, Result},
};

pub mod buffer;
pub mod context;
pub mod mem;
pub mod program;

/// Packs kernel arguments into the `void**` array expected by `cuLaunchKernel`.
///
/// Each argument's value is stored as raw bytes in `values`. After visiting all
/// args, `as_ptrs()` returns a mutable slice of `*mut c_void` pointing into
/// those buffers — the slice lifetime is tied to `self`.
struct CudaArgPacker {
    values: Vec<Vec<u8>>,
}

impl CudaArgPacker {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        self.values.push(bytes.to_vec());
    }

    /// Returns a `Vec` of `*mut c_void` pointers, one per argument, each
    /// pointing at the argument's value buffer. The caller must not outlive
    /// `self`.
    fn as_ptrs(&mut self) -> Vec<*mut core::ffi::c_void> {
        self.values
            .iter_mut()
            .map(|v| v.as_mut_ptr().cast())
            .collect()
    }
}

impl ArgVisitor for CudaArgPacker {
    fn visit_ptr(&mut self, ptr: *mut core::ffi::c_void) {
        self.push_bytes(&(ptr as usize).to_ne_bytes());
    }

    fn visit_i32(&mut self, val: i32) {
        self.push_bytes(&val.to_ne_bytes());
    }

    fn visit_u32(&mut self, val: u32) {
        self.push_bytes(&val.to_ne_bytes());
    }

    fn visit_f32(&mut self, val: f32) {
        self.push_bytes(&val.to_ne_bytes());
    }
}

pub struct CudaLaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: [u32; 3],
}

impl LaunchConfig for CudaLaunchConfig {}

#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub id: i32,
    pub name: String,
    pub major: i32,
    pub minor: i32,
    pub multi_processor_count: i32,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multi_processor: i32,
    pub max_blocks_per_multi_processor: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub concurrent_kernels: i32,
}

impl CudaDeviceInfo {
    pub fn new(id: i32, props: cuda::cudaDeviceProp) -> Self {
        let name = unsafe { std::ffi::CStr::from_ptr(props.name.as_ptr()) };
        let name = name.to_string_lossy().to_string();

        CudaDeviceInfo {
            id,
            name,
            major: props.major,
            minor: props.minor,
            multi_processor_count: props.multiProcessorCount,
            total_global_mem: props.totalGlobalMem,
            shared_mem_per_block: props.sharedMemPerBlock,
            regs_per_block: props.regsPerBlock,
            warp_size: props.warpSize,
            max_threads_per_block: props.maxThreadsPerBlock,
            max_threads_per_multi_processor: props.maxThreadsPerMultiProcessor,
            max_blocks_per_multi_processor: props.maxBlocksPerMultiProcessor,
            max_threads_dim: props.maxThreadsDim,
            max_grid_size: props.maxGridSize,
            memory_bus_width: props.memoryBusWidth,
            l2_cache_size: props.l2CacheSize,
            concurrent_kernels: props.concurrentKernels,
        }
    }
}
impl DeviceInfo for CudaDeviceInfo {
    type Id = i32;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct CudaDevice<'a> {
    pub info: CudaDeviceInfo,
    device: cuda::CUdevice,
    context: cuda::CUcontext,
    _unused: PhantomData<&'a ()>,
}

impl<'a> CudaDevice<'a> {
    pub fn try_new(id: i32) -> Result<Self> {
        let device_id = id;
        let mut device = cuda::CUdevice::default();
        let status = unsafe { cuda::cuDeviceGet(&mut device, device_id) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        let mut props = cuda::cudaDeviceProp::default();
        let status = unsafe { cuda::cudaGetDeviceProperties(&mut props, device_id) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        let info = CudaDeviceInfo::new(device_id, props);

        let mut context = cuda::CUcontext::default();
        let mut params = cuda::CUctxCreateParams::default();
        let flags = 0;
        let status = unsafe { cuda::cuCtxCreate_v4(&mut context, &mut params, flags, device) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        Ok(Self {
            device,
            context,
            info,
            _unused: PhantomData,
        })
    }

    pub fn info(&self) -> &CudaDeviceInfo {
        &self.info
    }
}

impl<'a> Drop for CudaDevice<'a> {
    fn drop(&mut self) {
        let result = unsafe { cuda::cuCtxDestroy_v2(self.context) };
        if result != cuda::cudaError_enum_CUDA_SUCCESS {
            // just log, we can't do anything about it
            eprintln!("Failed to destroy CUDA context: {}", result);
        }
    }
}

impl<'a> Device<'a> for CudaDevice<'a> {
    type Buffer<N: Num> = CudaBuffer<'a, N>;
    type Program<K: teeny_core::context::program::Kernel> = CudaProgram<'a, K>;
    type LaunchConfig = CudaLaunchConfig;

    fn buffer<N: Num>(&self, count: usize) -> teeny_core::errors::Result<Self::Buffer<N>> {
        Ok(CudaBuffer::try_new(count)?)
    }

    fn launch<K: Kernel>(
        &self,
        program: &Self::Program<K>,
        cfg: &Self::LaunchConfig,
        args: K::Args<'a>,
    ) -> teeny_core::errors::Result<()> {
        // Allocate global scratch memory for TMA descriptors if the kernel requires it.
        // The kernel uses per-CTA scratch = global_scratch_bytes_per_cta * num_ctas bytes.
        let num_ctas = (cfg.grid[0] * cfg.grid[1] * cfg.grid[2]) as u64;
        let scratch_total = program.global_scratch_bytes_per_cta * num_ctas;
        let mut scratch_ptr: cuda::CUdeviceptr = 0;
        if scratch_total > 0 {
            // cuMemAlloc_v2 guarantees 256-byte alignment, which satisfies Triton's
            // scratch alignment requirement (typically 128 bytes).
            let alloc_status =
                unsafe { cuda::cuMemAlloc_v2(&mut scratch_ptr, scratch_total as usize) };
            if alloc_status != cuda::cudaError_enum_CUDA_SUCCESS {
                return Err(Error::from_cuda_error(alloc_status).into());
            }
            // Zero-initialize the scratch pad so TMA descriptors start in a clean state.
            unsafe { cuda::cuMemsetD8_v2(scratch_ptr, 0, scratch_total as usize) };
            eprintln!("[LAUNCH] allocated {} bytes global scratch", scratch_total);
        }

        let mut packer = CudaArgPacker::new();
        args.visit_args(&mut packer);
        // Trailing Triton kernel parameters: global scratch pad + profile scratch pad.
        packer.visit_ptr(scratch_ptr as *mut std::ffi::c_void); // global scratch pad
        packer.visit_ptr(std::ptr::null_mut()); // profile scratch pad (unused)
        // Build the pointer array while `packer` is still alive — both must
        // remain live for the entire duration of `cuLaunchKernel`.
        let mut ptrs = packer.as_ptrs();

        let status = unsafe {
            cuda::cuLaunchKernel(
                program.function,
                cfg.grid[0],
                cfg.grid[1],
                cfg.grid[2],
                cfg.block[0],
                cfg.block[1],
                cfg.block[2],
                program.shared_mem_bytes, // dynamic shared memory required by Triton kernel
                std::ptr::null_mut(),     // hStream (default/null stream)
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(), // extra
            )
        };

        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        // `cuLaunchKernel` returns immediately; the kernel runs asynchronously.
        // Synchronize here so that any GPU-side fault (bad pointer, out-of-bounds
        // access) surfaces as a CUDA error code rather than a later SIGSEGV.
        let sync_status = unsafe { cuda::cuCtxSynchronize() };

        println!("sync_status: {:?}", sync_status);

        if sync_status != cuda::cudaError_enum_CUDA_SUCCESS {
            if scratch_ptr != 0 {
                unsafe { cuda::cuMemFree_v2(scratch_ptr) };
            }
            return Err(Error::from_cuda_error(sync_status).into());
        }

        if scratch_ptr != 0 {
            unsafe { cuda::cuMemFree_v2(scratch_ptr) };
        }

        println!("Launch completed successfully");

        Ok(())
    }
}
