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

//! Thin safe wrappers around the CUDA driver memory API.

use crate::cuda;
use crate::errors::{Error, Result};

/// Opaque CUDA device pointer (a virtual address in device memory).
pub type DevicePtr = cuda::CUdeviceptr;

/// Allocate `byte_size` bytes of device memory in the current CUDA context.
pub fn alloc(byte_size: usize) -> Result<DevicePtr> {
    let mut ptr: DevicePtr = 0;
    let status = unsafe { cuda::cuMemAlloc_v2(&mut ptr, byte_size) };
    if status != cuda::cudaError_enum_CUDA_SUCCESS {
        return Err(Error::from_cuda_error(status).into());
    }
    Ok(ptr)
}

/// Free a device allocation previously returned by [`alloc`].
pub fn free(ptr: DevicePtr) -> Result<()> {
    let status = unsafe { cuda::cuMemFree_v2(ptr) };
    if status != cuda::cudaError_enum_CUDA_SUCCESS {
        return Err(Error::from_cuda_error(status).into());
    }
    Ok(())
}

/// Copy `count` elements of type `T` from host memory to device memory.
///
/// # Safety
/// `src` must be valid for `count` reads. `dst` must be a valid device
/// allocation of at least `count * size_of::<T>()` bytes.
pub unsafe fn copy_h_to_d<T>(dst: DevicePtr, src: *const T, count: usize) -> Result<()> {
    let status =
        unsafe { cuda::cuMemcpyHtoD_v2(dst, src.cast(), count * std::mem::size_of::<T>()) };
    if status != cuda::cudaError_enum_CUDA_SUCCESS {
        return Err(Error::from_cuda_error(status).into());
    }
    Ok(())
}

/// Copy `count` elements of type `T` from device memory to host memory.
///
/// # Safety
/// `dst` must be valid for `count` writes. `src` must be a valid device
/// allocation of at least `count * size_of::<T>()` bytes.
pub unsafe fn copy_d_to_h<T>(dst: *mut T, src: DevicePtr, count: usize) -> Result<()> {
    let status =
        unsafe { cuda::cuMemcpyDtoH_v2(dst.cast(), src, count * std::mem::size_of::<T>()) };
    if status != cuda::cudaError_enum_CUDA_SUCCESS {
        return Err(Error::from_cuda_error(status).into());
    }
    Ok(())
}
