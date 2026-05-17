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
/// LLVM's NVPTX backend emits `.target sm_NNNa` for each architecture:
///   sm_90  → sm_90a  (Hopper:   TMA, wgmma)
///   sm_100 → sm_100a (Blackwell DC: B100/B200/GB200)
///   sm_120 → sm_120a (Blackwell consumer: RTX 50xx)
///
/// Architecture-specific PTX (the `a` suffix) runs only on its own
/// architecture and forward via native execution — NOT via driver JIT
/// cross-architecture.  Each GPU therefore needs code compiled for its own SM
/// version to avoid `ptxas fatal: ... cannot be compiled to future architecture`.
///
/// The teenyc backend (LLVM 20+) validates all SM versions up to sm_120.
/// If a future architecture is released before the backend adds support,
/// extend this constant and the match arm below.
const MAX_CODEGEN_CAPABILITY: Capability = Capability::Sm120;

pub fn compile_kernel(kernel: &impl Kernel, target: &Target, force: bool) -> Result<String> {
    let teenyc_path = std::env::var("TEENYC_PATH").unwrap_or_else(|_| "teenyc".to_string());
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());

    let effective_cpu = clamp_capability(target.capability).to_string();
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?.with_target_cpu(effective_cpu);
    compiler.compile(kernel, target, force)
}

/// Clamp `cap` to `MAX_CODEGEN_CAPABILITY` for any architecture newer than
/// what the backend supports.  All SM versions up to sm_120 are natively
/// supported, so no clamping is needed for current hardware.  If a new SM
/// version is added to `Capability` before teenyc validates it, add a match
/// arm here that maps it to `MAX_CODEGEN_CAPABILITY`.
fn clamp_capability(cap: Capability) -> Capability {
    cap
}
