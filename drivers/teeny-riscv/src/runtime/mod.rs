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

use std::path::{Path, PathBuf};

use libloading::{Library, Symbol};

use crate::errors::{Error, Result};

/// A dlopen'd kernel shared library, as produced by `LlvmCompiler::compile` targeting
/// [`crate::compiler::TARGET_TRIPLE`] (see `RiscvBackend::makeBIN` in the `teeny` compiler fork,
/// which links the compiled kernel into a real ELF shared library via `ld.lld`).
///
/// This only makes sense to actually load on RISC-V (native, or under user-mode emulation like
/// `qemu-riscv64`) -- `dlopen`-ing a RISC-V `.so` from an x86_64 process fails immediately with
/// an architecture mismatch. See the crate README for how this was verified during development.
pub struct KernelLibrary {
    path: PathBuf,
    lib: Library,
}

impl KernelLibrary {
    /// Loads the kernel shared library at `path`.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        // Safety: loading an arbitrary shared library is inherently unsafe -- its
        // initializers run immediately, and any symbol resolved from it must be
        // called with a signature matching what the compiler actually generated.
        // The caller is responsible for path being a kernel library this backend
        // produced, not arbitrary/untrusted input.
        let lib = unsafe { Library::new(&path) }.map_err(|source| Error::LoadLibrary {
            path: path.clone(),
            source,
        })?;
        Ok(Self { path, lib })
    }

    /// Path this library was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Resolves `symbol` as a no-argument, no-return kernel entry point (the shape
    /// `RiscvBackend::makeLLVMIR` currently always generates -- see the crate README) and calls
    /// it.
    ///
    /// # Safety
    /// The caller must ensure `symbol` actually refers to a function with the `extern "C" fn()`
    /// signature. Calling with a mismatched signature is undefined behavior.
    pub unsafe fn call_void_kernel(&self, symbol: &str) -> Result<()> {
        // Safety: forwarded to the caller via this function's own safety contract.
        let func: Symbol<unsafe extern "C" fn()> = unsafe { self.lib.get(symbol.as_bytes()) }
            .map_err(|source| Error::ResolveSymbol {
                path: self.path.clone(),
                symbol: symbol.to_string(),
                source,
            })?;
        unsafe { func() };
        Ok(())
    }
}
