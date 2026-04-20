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

use teeny_core::compiler::Compiler;
use teeny_core::context::program::Kernel;

use crate::compiler::backend::llvm::compiler::LlvmCompiler;
use crate::compiler::target::cuda::Target;
use crate::errors::Result;

pub fn compile_kernel(kernel: &impl Kernel, target: &Target, force: bool) -> Result<String> {
    let rustc_path = std::env::var("TEENY_RUSTC_PATH")?;
    let cache_dir =
        std::env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());

    let compiler = LlvmCompiler::new(rustc_path, cache_dir)?
        .with_target_cpu(target.capability.to_string());
    compiler.compile(kernel, target, force)
}
