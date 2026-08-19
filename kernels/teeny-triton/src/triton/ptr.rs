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

// Kernel-argument markers that tag a value as an input, output, or both.
//
// These wrap a device pointer, a tile value, or any other Copy handle at the
// #[kernel] / #[tiled_kernel] parameter boundary so fusion / lowering can
// classify ops by I/O arity. Not pointer-specific: the same markers tag
// #[kernel]'s Triton::Pointer args and #[tiled_kernel]'s Tile args alike.
// Bodies Deref to the inner handle (`x.add_offsets(...)` on the host).
//
// The #[kernel] macro strips markers from the device-side source string —
// MLIR only lowers bare pointers/tensors. Keep this file free of derive,
// //! docs, and host-only APIs (see KernelIo in crate::kernel_io).

use core::ops::{Deref, DerefMut};

/// Read-only kernel argument.
#[repr(transparent)]
pub struct In<P>(pub P);

impl<P: Clone> Clone for In<P> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<P: Copy> Copy for In<P> {}

impl<P> Deref for In<P> {
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Write-only kernel argument.
#[repr(transparent)]
pub struct Out<P>(pub P);

impl<P: Clone> Clone for Out<P> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<P: Copy> Copy for Out<P> {}

impl<P> Deref for Out<P> {
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P> DerefMut for Out<P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Read-write / in-place kernel argument.
#[repr(transparent)]
pub struct InOut<P>(pub P);

impl<P: Clone> Clone for InOut<P> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<P: Copy> Copy for InOut<P> {}

impl<P> Deref for InOut<P> {
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P> DerefMut for InOut<P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
