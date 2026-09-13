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

use alloc::string::String;

use crate::device::program::Kernel;
use crate::errors::Result;

/// A compilation target (e.g. a GPU architecture).
pub trait Target: Sized {
    /// The `-Ctarget-cpu`-style string identifying this target, if applicable.
    fn target_cpu(&self) -> Option<String> {
        None
    }
}

/// Which artifacts a [`Compiler`] should produce, and whether cached ones may be reused.
///
/// [`Default`] asks for both the object and the assembly without forcing a rebuild, which is
/// what every call site relied on before these options existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerOptions {
    /// Recompile even if a cached artifact already exists.
    pub force: bool,
    /// Emit the generated assembly (`.s`) alongside the object. Snapshot tests read this (see
    /// `teeny_test::read_compiled_asm`); on RISC-V it is the only readable form of the kernel,
    /// since the object there is a linked shared library.
    pub emit_asm: bool,
    /// Emit the object file (`.o`) -- the artifact that actually gets loaded and launched.
    pub emit_bin: bool,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            force: false,
            emit_asm: true,
            emit_bin: true,
        }
    }
}

impl CompilerOptions {
    /// Requests only the object file, skipping assembly generation.
    pub fn bin_only() -> Self {
        Self {
            emit_asm: false,
            ..Self::default()
        }
    }

    /// Requests only the generated assembly, skipping the object file.
    pub fn asm_only() -> Self {
        Self {
            emit_bin: false,
            ..Self::default()
        }
    }

    /// Returns `self` with [`force`](Self::force) set to `force`.
    pub fn with_force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }
}

/// Something capable of compiling a [`Kernel`] for a [`Target`].
pub trait Compiler {
    /// Compiles `kernel` for `target` as directed by `options`, returning the path to the
    /// primary artifact: the object file when [`CompilerOptions::emit_bin`] is set, and the
    /// generated assembly otherwise. Both share a stem, so either path locates the other by
    /// extension -- which is how `teeny_test::read_compiled_asm` finds the `.s`.
    fn compile(
        &self,
        kernel: &impl Kernel,
        target: &impl Target,
        options: &CompilerOptions,
    ) -> Result<String>;
}
