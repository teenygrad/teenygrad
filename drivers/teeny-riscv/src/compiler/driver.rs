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

use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::compiler::{Compiler, Target as _};
use teeny_core::device::program::Kernel;

use crate::compiler::TARGET_TRIPLE;
use crate::compiler::target::Target;
use crate::errors::Result;

/// Compiles `kernel` for `target` via the LLVM backend's RISC-V path (`riscv64-generic`), using
/// the `teenyc` binary resolved by [`teeny_compiler::compiler::find_teenyc`] and the default
/// cache directory. Mirrors `teeny_cuda::compiler::compile_kernel`'s shape so call sites read
/// the same way across backends.
///
/// `force`: recompile even if a cached artifact exists.
///
/// The underlying backend (`RiscvBackend`) is still a stub -- every kernel compiles to the same
/// placeholder no-argument `void @<name>()` function regardless of `kernel`'s actual body (see
/// the crate README) -- so this only proves the compile pipeline (real LLVM RISC-V codegen,
/// linked via `ld.lld`) produces a genuine RISC-V shared object, not that it runs `kernel`'s
/// logic.
pub fn compile_kernel(kernel: &impl Kernel, target: &Target, force: bool) -> Result<String> {
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir = teeny_compiler::compiler::default_cache_dir();
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?
        .with_target_triple(TARGET_TRIPLE)
        .with_target_cpu(
            target
                .target_cpu()
                .expect("Target::target_cpu is always Some"),
        );

    compiler.compile(kernel, target, force)
}
