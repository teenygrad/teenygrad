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

use teeny_core::{context::buffer::Buffer, dtype::Num};

use crate::errors::{Error, Result};
use crate::mem::{self, DevicePtr};

/// A device-side buffer holding `count` elements of type `N`.
///
/// The allocation size is derived from `N::BITS`: `count * N::BITS / 8` bytes.
/// Memory is freed automatically on drop.
pub struct CudaBuffer<'a, N: Num> {
    ptr: DevicePtr,
    count: usize,
    _unused: PhantomData<&'a ()>,
    _num: PhantomData<N>,
}

impl<'a, N: Num> CudaBuffer<'a, N> {
    pub fn try_new(count: usize) -> Result<Self> {
        let byte_size = count * N::BITS as usize / 8;
        let ptr = mem::alloc(byte_size)?;
        Ok(Self {
            ptr,
            count,
            _unused: PhantomData,
            _num: PhantomData,
        })
    }

    pub fn as_device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

impl<'a, N: Num> Drop for CudaBuffer<'a, N> {
    fn drop(&mut self) {
        if let Err(e) = mem::free(self.ptr) {
            eprintln!("Failed to free CUDA buffer: {e}");
        }
    }
}

impl<'a, N: Num> Buffer<'a, N> for CudaBuffer<'a, N> {
    fn to_device(&mut self, src: &[N]) -> teeny_core::errors::Result<()> {
        if src.len() > self.count {
            return Err(Error::BufferOverflow {
                src: src.len(),
                buf: self.count,
            }
            .into());
        }
        // SAFETY: src slice is valid for src.len() reads by construction.
        unsafe { mem::copy_h_to_d(self.ptr, src.as_ptr(), src.len()) }
    }

    fn to_host(&self, dst: &mut [N]) -> teeny_core::errors::Result<()> {
        // SAFETY: dst slice is valid for dst.len() writes by construction.
        unsafe { mem::copy_d_to_h(dst.as_mut_ptr(), self.ptr, dst.len()) }
    }
}
