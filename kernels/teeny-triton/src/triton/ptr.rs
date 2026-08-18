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

// Kernel-argument markers that tag device pointers as inputs, outputs, or both.
//
// These wrap Triton::Pointer (or any Copy pointer handle) at the #[kernel]
// parameter boundary so fusion / lowering can classify ops by I/O arity.
// Bodies Deref to the inner handle (`x.add_offsets(...)` on the host).
//
// The #[kernel] macro strips markers from the device-side source string —
// MLIR only lowers bare pointers. Keep this file free of derive, //! docs,
// and host-only APIs (see KernelIo in crate::kernel_io).

use core::ops::{Deref, DerefMut};

/// Read-only device pointer argument.
#[repr(transparent)]
pub struct InPtr<P>(pub P);

impl<P: Clone> Clone for InPtr<P> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<P: Copy> Copy for InPtr<P> {}

impl<P> Deref for InPtr<P> {
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Write-only device pointer argument.
#[repr(transparent)]
pub struct OutPtr<P>(pub P);

impl<P: Clone> Clone for OutPtr<P> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<P: Copy> Copy for OutPtr<P> {}

impl<P> Deref for OutPtr<P> {
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P> DerefMut for OutPtr<P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Read-write / in-place device pointer argument.
#[repr(transparent)]
pub struct InOutPtr<P>(pub P);

impl<P: Clone> Clone for InOutPtr<P> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<P: Copy> Copy for InOutPtr<P> {}

impl<P> Deref for InOutPtr<P> {
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P> DerefMut for InOutPtr<P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
