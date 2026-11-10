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

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use teeny_core::device::buffer::Buffer;
use teeny_core::dtype::Num;

use crate::errors::{Error, Result};

/// The address range of one live buffer allocated by a [`crate::device::RiscvDevice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "qemu"), allow(dead_code))]
pub(crate) struct Region {
    /// Address of the buffer's first byte.
    pub(crate) addr: usize,
    /// The buffer's size in bytes.
    pub(crate) bytes: usize,
}

/// The buffers a device has allocated and not yet dropped, so `launch` can map a kernel's pointer
/// arguments back to the memory they address.
#[derive(Debug, Default)]
pub(crate) struct BufferRegistry {
    regions: Mutex<Vec<Region>>,
}

impl BufferRegistry {
    /// Locks the registry. While the guard is held no registered buffer can be freed, since a
    /// buffer unregisters itself, under this lock, before its memory is released.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Vec<Region>> {
        // A panic while holding the lock can't leave the list inconsistent: every update is a
        // single push or swap_remove.
        self.regions.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The registered region containing `addr`, if any.
    #[cfg(test)]
    pub(crate) fn find(&self, addr: usize) -> Option<Region> {
        Self::find_in(&self.lock(), addr)
    }

    /// The region in `regions` containing `addr`, if any.
    #[cfg_attr(not(any(feature = "qemu", test)), allow(dead_code))]
    pub(crate) fn find_in(regions: &[Region], addr: usize) -> Option<Region> {
        regions
            .iter()
            .copied()
            .find(|r| addr >= r.addr && addr - r.addr < r.bytes)
    }

    fn register(&self, region: Region) {
        self.lock().push(region);
    }

    fn unregister(&self, region: Region) {
        let mut regions = self.lock();
        if let Some(index) = regions.iter().position(|r| *r == region) {
            regions.swap_remove(index);
        }
    }
}

/// A host-memory buffer holding `count` elements of type `N`.
///
/// Unlike `teeny-cuda`'s `CudaBuffer`, there is no separate device address space to allocate in
/// or copy across. A launched kernel reads and writes this memory through the pointer from
/// [`Self::as_device_ptr`], so each element sits in an `UnsafeCell`.
pub struct RiscvBuffer<'a, N: Num> {
    data: Vec<UnsafeCell<N>>,
    /// The allocating device's registry, which this buffer stays registered in until dropped.
    registry: Option<Arc<BufferRegistry>>,
    _unused: PhantomData<&'a ()>,
}

impl<'a, N: Num> RiscvBuffer<'a, N> {
    /// Allocates a zero-initialized host buffer for `count` elements of `N`.
    ///
    /// No device knows about a buffer created this way, so it can't be passed to a kernel launch;
    /// allocate one with [`teeny_core::device::Device::buffer`] for that.
    pub fn try_new(count: usize) -> Result<Self> {
        Ok(Self::allocate(count, None))
    }

    /// Allocates a zero-initialized buffer that stays registered in `registry` until dropped.
    pub(crate) fn with_registry(count: usize, registry: Arc<BufferRegistry>) -> Self {
        Self::allocate(count, Some(registry))
    }

    fn allocate(count: usize, registry: Option<Arc<BufferRegistry>>) -> Self {
        // Safety: every `Num` impl in this codebase is a plain numeric type (integers, floats)
        // whose all-zero bit pattern is a valid value.
        let data = (0..count)
            .map(|_| UnsafeCell::new(unsafe { std::mem::zeroed::<N>() }))
            .collect();
        let buffer = Self {
            data,
            registry,
            _unused: PhantomData,
        };
        if let (Some(registry), Some(region)) = (&buffer.registry, buffer.region()) {
            registry.register(region);
        }
        buffer
    }

    /// This buffer's address range, or `None` if it holds no bytes.
    fn region(&self) -> Option<Region> {
        let bytes = std::mem::size_of_val(self.data.as_slice());
        (bytes > 0).then_some(Region {
            addr: self.data.as_ptr() as usize,
            bytes,
        })
    }

    /// The number of elements this buffer holds.
    pub fn count(&self) -> usize {
        self.data.len()
    }

    /// A raw pointer to this buffer's storage, for building a kernel's argument tuple.
    ///
    /// Named to match `CudaBuffer::as_device_ptr` so generic call sites (e.g. building a
    /// [`teeny_core::device::program::Kernel::Args`] tuple) compile the same way against either
    /// buffer type. A launched kernel may write through it.
    pub fn as_device_ptr(&self) -> *mut N {
        UnsafeCell::raw_get(self.data.as_ptr())
    }
}

impl<N: Num> Drop for RiscvBuffer<'_, N> {
    fn drop(&mut self) {
        if let (Some(registry), Some(region)) = (&self.registry, self.region()) {
            registry.unregister(region);
        }
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
        for (cell, &value) in self.data.iter_mut().zip(src) {
            *cell.get_mut() = value;
        }
        Ok(())
    }

    fn to_host(&self, dst: &mut [N]) -> Result<()> {
        for (value, cell) in dst.iter_mut().zip(&self.data) {
            // Safety: nothing writes to the buffer concurrently -- `launch` copies a kernel's
            // writes back before it returns, and it can't run while this shared borrow is used to
            // read.
            *value = unsafe { *cell.get() };
        }
        Ok(())
    }
}
