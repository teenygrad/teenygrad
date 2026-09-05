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

use teeny_core::device::buffer::Buffer;
use teeny_core::dtype::Num;

use crate::errors::{Error, Result};

/// A host-memory buffer holding `count` elements of type `N`.
///
/// Unlike `teeny-cuda`'s `CudaBuffer`, there is no separate device address space to allocate in
/// or copy across -- RISC-V kernels run against host-owned memory directly, so this is just a
/// `Vec<N>`.
pub struct RiscvBuffer<'a, N: Num> {
    data: Vec<N>,
    _unused: PhantomData<&'a ()>,
}

impl<'a, N: Num> RiscvBuffer<'a, N> {
    /// Allocates a zero-initialized host buffer for `count` elements of `N`.
    pub fn try_new(count: usize) -> Result<Self> {
        // Safety: every `Num` impl in this codebase is a plain numeric type (integers, floats)
        // whose all-zero bit pattern is a valid value -- matches `CudaBuffer`'s uninitialized
        // `cuMemAlloc`, except zeroed rather than left uninitialized.
        let data = vec![unsafe { std::mem::zeroed::<N>() }; count];
        Ok(Self {
            data,
            _unused: PhantomData,
        })
    }

    /// The number of elements this buffer holds.
    pub fn count(&self) -> usize {
        self.data.len()
    }

    /// A raw pointer to this buffer's host-owned storage.
    ///
    /// Named to match `CudaBuffer::as_device_ptr` so generic call sites (e.g. building a
    /// [`teeny_core::device::program::Kernel::Args`] tuple, which is a fixed pointer-argument
    /// shape regardless of backend) compile the same way against either buffer type. RISC-V
    /// kernel launches don't dereference this yet -- see
    /// [`crate::errors::Error::ArgumentPassingNotSupported`].
    pub fn as_device_ptr(&self) -> *mut N {
        self.data.as_ptr() as *mut N
    }
}

impl<'a, N: Num> Buffer<'a, N> for RiscvBuffer<'a, N> {
    fn to_device(&mut self, src: &[N]) -> Result<()> {
        if src.len() > self.data.len() {
            return Err(Error::BufferOverflow {
                src: src.len(),
                buf: self.data.len(),
            }
            .into());
        }
        self.data[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn to_host(&self, dst: &mut [N]) -> Result<()> {
        let len = dst.len().min(self.data.len());
        dst[..len].copy_from_slice(&self.data[..len]);
        Ok(())
    }
}
