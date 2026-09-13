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
use std::path::{Path, PathBuf};

use teeny_core::device::program::{Kernel, Program};

use crate::elf;
use crate::errors::Result;

/// A kernel `K` compiled for RISC-V: the shared library `compile_kernel` produced and the entry
/// point it exports.
///
/// Nothing is loaded into this process -- the library is a RISC-V ELF, which can't be `dlopen`ed
/// on another architecture. [`crate::device::RiscvDevice`]'s `launch` runs it instead.
pub struct RiscvProgram<'a, K: Kernel> {
    path: PathBuf,
    entry_point: String,
    _unused: PhantomData<&'a ()>,
    _kernel: PhantomData<K>,
}

impl<'a, K: Kernel> RiscvProgram<'a, K> {
    /// Opens the kernel shared library at `path` and finds its entry point, the one exported
    /// function named `*_entry_point`.
    pub fn try_new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let entry_point = elf::find_entry_point(&path)?;
        Ok(Self::from_parts(path, entry_point))
    }

    pub(crate) fn from_parts(path: impl Into<PathBuf>, entry_point: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            entry_point: entry_point.into(),
            _unused: PhantomData,
            _kernel: PhantomData,
        }
    }

    /// Path to this program's shared library.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The name of the kernel's entry point in the shared library.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

impl<'a, K: Kernel> Program<'a, K> for RiscvProgram<'a, K> {}
