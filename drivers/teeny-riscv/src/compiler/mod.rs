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

/// `-C target-cpu` chip identifiers and the [`teeny_core::compiler::Target`] impl for RISC-V.
pub mod target;

/// `--target` triple `teenyc` uses for the RISC-V backend, set on `LlvmCompiler` via
/// [`teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler::with_target_triple`].
///
/// A dedicated triple (not one of the general-purpose `riscv64gc-*` targets) that exists purely
/// to be compiled through the `mlir` codegen backend's RISC-V path -- see
/// `rustc_target::spec::targets::riscv64_generic` in the `teeny` compiler fork.
pub const TARGET_TRIPLE: &str = "riscv64-generic";
