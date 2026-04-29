/*
 * Copyright (c) 2026 Teenygrad.
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

use teeny_core::compiler::{Capability, Compiler};
use teeny_core::device::program::Kernel;

use crate::compiler::backend::llvm::compiler::LlvmCompiler;
use crate::compiler::target::cuda::Target;
use crate::errors::Result;

/// Highest SM version our Triton MLIR codegen has validated support for.
///
/// LLVM's NVPTX backend maps `sm_90` to `.target sm_90a` in PTX, and the `a`
/// suffix means architecture-specific — not forward-compatible with newer GPUs.
/// `sm_89` (Ada Lovelace) has no `a` variant and generates plain `.target sm_89`
/// PTX, which the CUDA driver JITs forward-compatibly on any newer architecture.
const MAX_CODEGEN_CAPABILITY: Capability = Capability::Sm89;

pub fn compile_kernel(kernel: &impl Kernel, target: &Target, force: bool) -> Result<String> {
    let rustc_path = std::env::var("TEENY_RUSTC_PATH")?;
    let cache_dir =
        std::env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());

    let effective_cpu = clamp_capability(target.capability).to_string();
    let compiler = LlvmCompiler::new(rustc_path, cache_dir)?.with_target_cpu(effective_cpu);
    compiler.compile(kernel, target, force)
}

/// Clamp `cap` down to `MAX_CODEGEN_CAPABILITY` when the requested capability
/// is newer than what the backend supports, so PTX remains forward-compatible.
fn clamp_capability(cap: Capability) -> Capability {
    match cap {
        Capability::Sm90 | Capability::Sm100 | Capability::Sm120a => MAX_CODEGEN_CAPABILITY,
        other => other,
    }
}
