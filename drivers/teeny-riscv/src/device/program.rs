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
use std::path::Path;

use teeny_core::device::program::{Kernel, Program};

use crate::errors::Result;
use crate::runtime::KernelLibrary;

/// A kernel `K` compiled for RISC-V and `dlopen`'d, ready to (eventually) launch.
///
/// `try_new` genuinely loads `path` via [`KernelLibrary::load`] -- this only succeeds when
/// actually running on RISC-V (native, or under `qemu-riscv64`); on any other host it fails
/// immediately with an architecture-mismatch error, the same way `KernelLibrary::load` always
/// has. That's a real, correct error, not something this type papers over.
pub struct RiscvProgram<'a, K: Kernel> {
    library: KernelLibrary,
    _unused: PhantomData<&'a ()>,
    _kernel: PhantomData<K>,
}

impl<'a, K: Kernel> RiscvProgram<'a, K> {
    /// Loads the kernel shared library at `path`.
    pub fn try_new(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let library = KernelLibrary::load(path)?;
        Ok(Self {
            library,
            _unused: PhantomData,
            _kernel: PhantomData,
        })
    }

    /// Path this program's shared library was loaded from.
    pub fn path(&self) -> &Path {
        self.library.path()
    }

    /// The underlying loaded library, for callers that need
    /// [`KernelLibrary::call_void_kernel`] directly (real per-kernel argument passing isn't
    /// supported by [`teeny_core::device::Device::launch`] yet -- see
    /// [`crate::errors::Error::ArgumentPassingNotSupported`]).
    pub fn library(&self) -> &KernelLibrary {
        &self.library
    }
}

impl<'a, K: Kernel> Program<'a, K> for RiscvProgram<'a, K> {}
