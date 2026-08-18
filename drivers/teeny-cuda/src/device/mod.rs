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

use std::marker::PhantomData;

use teeny_core::{
    device::{
        context::DeviceInfo,
        hardware::{HardwareProfile, MemoryLevel, MemoryLevelKind},
        program::{ArgVisitor, Kernel, KernelArgs},
        {Device, LaunchConfig},
    },
    dtype::Num,
};

use crate::{
    cuda,
    device::buffer::CudaBuffer,
    device::program::CudaProgram,
    errors::{Error, Result},
};

/// Device memory buffers.
pub mod buffer;
/// Device/context management.
pub mod context;
/// Memory-related helpers.
pub mod mem;
/// Compiled kernel programs.
pub mod program;

/// Packs kernel arguments into the `void**` array expected by `cuLaunchKernel`.
///
/// Each argument's value is stored as raw bytes in `values`. After visiting all
/// args, `as_ptrs()` returns a mutable slice of `*mut c_void` pointing into
/// those buffers — the slice lifetime is tied to `self`.
pub struct CudaArgPacker {
    values: Vec<Vec<u8>>,
}

impl Default for CudaArgPacker {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaArgPacker {
    /// Creates an empty argument packer.
    pub fn new() -> Self {
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
    fn visit_bool(&mut self, val: bool) {
        self.push_bytes(&[val as u8]);
    }
    fn visit_i8(&mut self, val: i8) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_i16(&mut self, val: i16) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_i32(&mut self, val: i32) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_i64(&mut self, val: i64) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_u8(&mut self, val: u8) {
        self.push_bytes(&[val]);
    }
    fn visit_u16(&mut self, val: u16) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_u32(&mut self, val: u32) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_u64(&mut self, val: u64) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_f32(&mut self, val: f32) {
        self.push_bytes(&val.to_ne_bytes());
    }
    fn visit_f64(&mut self, val: f64) {
        self.push_bytes(&val.to_ne_bytes());
    }
}

/// A kernel launch's grid/block/cluster dimensions (in `(x, y, z)` order).
pub struct CudaLaunchConfig {
    /// Grid dimensions (number of blocks per dimension).
    pub grid: [u32; 3],
    /// Block dimensions (number of threads per block, per dimension).
    pub block: [u32; 3],
    /// Thread-block-cluster dimensions.
    pub cluster: [u32; 3],
}

impl LaunchConfig for CudaLaunchConfig {}

/// A CUDA device's static properties, read from `cudaGetDeviceProperties` at
/// [`CudaDevice::try_new`] time.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// The device's ordinal ID.
    pub id: i32,
    /// The device's name (e.g. `"NVIDIA GeForce RTX 5070"`).
    pub name: String,
    /// Compute capability major version.
    pub major: i32,
    /// Compute capability minor version.
    pub minor: i32,
    /// Number of streaming multiprocessors.
    pub multi_processor_count: i32,
    /// Total global memory, in bytes.
    pub total_global_mem: usize,
    /// Shared memory available per block, in bytes.
    pub shared_mem_per_block: usize,
    /// Number of 32-bit registers available per block.
    pub regs_per_block: i32,
    /// Warp size in threads.
    pub warp_size: i32,
    /// Maximum threads per block.
    pub max_threads_per_block: i32,
    /// Maximum resident threads per multiprocessor.
    pub max_threads_per_multi_processor: i32,
    /// Maximum resident blocks per multiprocessor.
    pub max_blocks_per_multi_processor: i32,
    /// Maximum block size, per dimension.
    pub max_threads_dim: [i32; 3],
    /// Maximum grid size, per dimension.
    pub max_grid_size: [i32; 3],
    /// Global memory bus width, in bits.
    pub memory_bus_width: i32,
    /// L2 cache size, in bytes.
    pub l2_cache_size: i32,
    /// Whether the device supports executing multiple kernels concurrently.
    pub concurrent_kernels: i32,
}

impl CudaDeviceInfo {
    /// Builds a `CudaDeviceInfo` from a raw `cudaDeviceProp`.
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

    /// Builds a [`HardwareProfile`] from this device's queried
    /// `cudaDeviceProp` fields. Memory-level bandwidth/latency are left
    /// `None`: `cudaGetDeviceProperties` doesn't expose them, and this
    /// crate's policy (see `teeny-triton`'s `CostModel` docs) is real
    /// hardware data or nothing, never a guess.
    fn hardware_profile(&self) -> HardwareProfile {
        HardwareProfile {
            name: self.name.clone(),
            compute_units: u32::try_from(self.multi_processor_count).unwrap_or(0),
            memory_levels: vec![
                MemoryLevel {
                    kind: MemoryLevelKind::Register,
                    capacity_bytes: u64::from(u32::try_from(self.regs_per_block).unwrap_or(0)) * 4,
                    bandwidth_bytes_per_sec: None,
                    latency_ns: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::SharedMemory,
                    capacity_bytes: self.shared_mem_per_block as u64,
                    bandwidth_bytes_per_sec: None,
                    latency_ns: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::L2Cache,
                    capacity_bytes: u64::from(u32::try_from(self.l2_cache_size).unwrap_or(0)),
                    bandwidth_bytes_per_sec: None,
                    latency_ns: None,
                },
                MemoryLevel {
                    kind: MemoryLevelKind::DeviceMemory,
                    capacity_bytes: self.total_global_mem as u64,
                    bandwidth_bytes_per_sec: None,
                    latency_ns: None,
                },
            ],
        }
    }
}

/// An open CUDA device and its context. Destroys the context on drop.
#[derive(Debug, Clone)]
pub struct CudaDevice<'a> {
    /// The device's static properties.
    pub info: CudaDeviceInfo,
    // Retained for future device-property queries; only `context` is currently used for CUDA
    // API calls.
    #[allow(dead_code)]
    device: cuda::CUdevice,
    context: cuda::CUcontext,
    _unused: PhantomData<&'a ()>,
}

impl<'a> CudaDevice<'a> {
    /// Opens device `id`, creating a new CUDA context for it.
    pub fn try_new(id: i32) -> Result<Self> {
        let device_id = id;
        let mut device = cuda::CUdevice::default();
        let status = unsafe { cuda::cuDeviceGet(&mut device, device_id) };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }

        let mut props = cuda::cudaDeviceProp::default();
        #[cfg(cuda_props_v2)]
        let status = unsafe { cuda::cudaGetDeviceProperties_v2(&mut props, device_id) };
        #[cfg(not(cuda_props_v2))]
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

    /// This device's static properties.
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

impl CudaDevice<'_> {
    /// Raise the per-function dynamic shared-memory ceiling when Triton asks for
    /// more than the default 48 KiB carveout.
    ///
    /// Without this, `cuLaunchKernel` returns `CUDA_ERROR_INVALID_ARGUMENT` for
    /// kernels whose `meta:shared` is in (48 KiB, opt-in max] — e.g.
    /// `conv2d_bn_silu_gemm` at 128×128 tiles (~64 KiB) on sm_120. Tiles that
    /// still exceed the device opt-in max (~99 KiB on RTX 5070) must be capped
    /// at compile/dispatch time instead.
    pub(crate) fn ensure_dynamic_shared(
        function: cuda::CUfunction,
        shared_bytes: u32,
    ) -> Result<()> {
        if shared_bytes == 0 {
            return Ok(());
        }
        let status = unsafe {
            cuda::cuFuncSetAttribute(
                function,
                cuda::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )
        };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(status).into());
        }
        Ok(())
    }
}

impl<'a> Device<'a> for CudaDevice<'a> {
    type Buffer<N: Num> = CudaBuffer<'a, N>;
    type Program<K: teeny_core::device::program::Kernel> = CudaProgram<'a, K>;
    type LaunchConfig = CudaLaunchConfig;

    fn buffer<N: Num>(&self, count: usize) -> teeny_core::errors::Result<Self::Buffer<N>> {
        CudaBuffer::try_new(count)
    }

    fn launch<K: Kernel>(
        &self,
        program: &Self::Program<K>,
        cfg: &Self::LaunchConfig,
        args: K::Args<'a>,
    ) -> teeny_core::errors::Result<()> {
        // Allocate global scratch memory for TMA descriptors if the kernel requires it.
        // Total scratch = per-CTA scratch * number of CTAs in the launch grid.
        let num_ctas = (cfg.grid[0] * cfg.grid[1] * cfg.grid[2]) as u64;
        let scratch_total = program.metadata.global_scratch_size * num_ctas;
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
        }

        let mut packer = CudaArgPacker::new();
        args.visit_args(&mut packer);
        // Trailing Triton kernel parameters: global scratch pad + profile scratch pad.
        packer.visit_ptr(scratch_ptr as *mut std::ffi::c_void); // global scratch pad
        packer.visit_ptr(std::ptr::null_mut()); // profile scratch pad (unused)
        // Build the pointer array while `packer` is still alive — both must
        // remain live for the entire duration of `cuLaunchKernel`.
        let mut ptrs = packer.as_ptrs();

        Self::ensure_dynamic_shared(program.function, program.metadata.shared)?;

        let status = unsafe {
            cuda::cuLaunchKernel(
                program.function,
                cfg.grid[0],
                cfg.grid[1],
                cfg.grid[2],
                cfg.block[0],
                cfg.block[1],
                cfg.block[2],
                program.metadata.shared, // dynamic shared memory required by Triton kernel
                std::ptr::null_mut(),    // hStream (default/null stream)
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

        if sync_status != cuda::cudaError_enum_CUDA_SUCCESS {
            if scratch_ptr != 0 {
                unsafe { cuda::cuMemFree_v2(scratch_ptr) };
            }
            return Err(Error::from_cuda_error(sync_status).into());
        }

        if scratch_ptr != 0 {
            unsafe { cuda::cuMemFree_v2(scratch_ptr) };
        }

        Ok(())
    }
}

impl<'a> CudaDevice<'a> {
    /// Launch a kernel on the given stream without allocating scratch or
    /// synchronising. Used during CUDA graph capture: the caller must have
    /// already appended the global-scratch-pad and profile-scratch-pad pointers
    /// to `packer` before calling this.
    pub(crate) fn launch_on_stream<K: Kernel>(
        &self,
        program: &CudaProgram<'_, K>,
        cfg: &CudaLaunchConfig,
        packer: &mut CudaArgPacker,
        stream: cuda::CUstream,
    ) -> Result<()> {
        let mut ptrs = packer.as_ptrs();
        Self::ensure_dynamic_shared(program.function, program.metadata.shared)?;
        let status = unsafe {
            cuda::cuLaunchKernel(
                program.function,
                cfg.grid[0],
                cfg.grid[1],
                cfg.grid[2],
                cfg.block[0],
                cfg.block[1],
                cfg.block[2],
                program.metadata.shared,
                stream,
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            Err(Error::from_cuda_error(status).into())
        } else {
            Ok(())
        }
    }

    /// Launch a pre-loaded kernel using a dynamically-built arg list.
    ///
    /// Use this instead of `launch` when the kernel type is erased (e.g. in
    /// `LoadedModel::forward`) and arguments are packed by `CudaArgPacker`
    /// via a `RuntimeOp` rather than a static `Kernel::Args` tuple.
    pub fn launch_with_packer<K: Kernel>(
        &self,
        program: &CudaProgram<'_, K>,
        cfg: &CudaLaunchConfig,
        packer: &mut CudaArgPacker,
    ) -> Result<()> {
        let num_ctas = (cfg.grid[0] * cfg.grid[1] * cfg.grid[2]) as u64;
        let scratch_total = program.metadata.global_scratch_size * num_ctas;
        let mut scratch_ptr: cuda::CUdeviceptr = 0;
        if scratch_total > 0 {
            let alloc_status =
                unsafe { cuda::cuMemAlloc_v2(&mut scratch_ptr, scratch_total as usize) };
            if alloc_status != cuda::cudaError_enum_CUDA_SUCCESS {
                return Err(Error::from_cuda_error(alloc_status).into());
            }
            unsafe { cuda::cuMemsetD8_v2(scratch_ptr, 0, scratch_total as usize) };
        }

        packer.visit_ptr(scratch_ptr as *mut std::ffi::c_void);
        packer.visit_ptr(std::ptr::null_mut());
        let mut ptrs = packer.as_ptrs();

        Self::ensure_dynamic_shared(program.function, program.metadata.shared)?;

        let status = unsafe {
            cuda::cuLaunchKernel(
                program.function,
                cfg.grid[0],
                cfg.grid[1],
                cfg.grid[2],
                cfg.block[0],
                cfg.block[1],
                cfg.block[2],
                program.metadata.shared,
                std::ptr::null_mut(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if status != cuda::cudaError_enum_CUDA_SUCCESS {
            if scratch_ptr != 0 {
                unsafe { cuda::cuMemFree_v2(scratch_ptr) };
            }
            return Err(Error::from_cuda_error(status).into());
        }

        let sync_status = unsafe { cuda::cuCtxSynchronize() };
        if scratch_ptr != 0 {
            unsafe { cuda::cuMemFree_v2(scratch_ptr) };
        }

        if sync_status != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(sync_status).into());
        }
        Ok(())
    }
}
