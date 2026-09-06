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

use std::path::PathBuf;

/// `teeny-riscv`'s result alias.
pub type Result<T> = anyhow::Result<T>;

/// Errors produced by `teeny-riscv`.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// A capability string didn't match any known RISC-V chip identifier.
    #[error("Unknown RISC-V capability: {0}")]
    UnknownCapability(String),

    /// Loading a compiled kernel's shared library failed (see [`libloading::Error`]).
    #[error("failed to load kernel library {path:?}: {source}")]
    LoadLibrary {
        /// Path to the shared library that failed to load.
        path: PathBuf,
        /// The underlying `libloading` error.
        #[source]
        source: libloading::Error,
    },

    /// Resolving a symbol within a loaded kernel library failed.
    #[error("failed to resolve symbol '{symbol}' in {path:?}: {source}")]
    ResolveSymbol {
        /// Path to the shared library the symbol was looked up in.
        path: PathBuf,
        /// The symbol name that failed to resolve.
        symbol: String,
        /// The underlying `libloading` error.
        #[source]
        source: libloading::Error,
    },

    /// A source buffer had more elements than the destination buffer could hold.
    #[error("buffer overflow: source has {src} elements but buffer holds {buf}")]
    BufferOverflow {
        /// Number of elements in the source.
        src: usize,
        /// Capacity of the destination buffer.
        buf: usize,
    },

    /// [`crate::device::Device::launch`] was called, but the RISC-V compiler backend
    /// (`RiscvBackend`) always emits the same no-argument placeholder function regardless of a
    /// kernel's actual body -- there is no real per-kernel argument ABI to marshal `args` into
    /// yet. Tracked on the `teenygrad-1zd` epic; see `teeny-riscv`'s README for current status.
    #[error(
        "RISC-V kernel argument passing is not supported yet -- the compiler backend only \
         produces a fixed no-argument placeholder kernel (see teenygrad-1zd)"
    )]
    ArgumentPassingNotSupported,
}
