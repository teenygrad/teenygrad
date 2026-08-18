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

use teeny_core::dtype;

use crate::compiler::backend::llvm::{compiler::LlvmCompiler, module::MlirModule};

#[cfg(feature = "ndarray")]
use crate::compiler::backend::ndarray::{compiler::NdarrayCompiler, module::NdarrayModule};

/// The `ndarray`-backed CPU backend.
#[cfg(feature = "ndarray")]
pub mod ndarray;

/// The LLVM/MLIR backend (compiles via the `teenyc` compiler at runtime).
pub mod llvm;

/// A compiled module, tagged by which backend produced it.
#[derive(Debug, Clone)]
pub enum Module<N: dtype::Dtype> {
    /// A module compiled by the LLVM/MLIR backend.
    Mlir(MlirModule<N>),

    /// A module compiled by the `ndarray` backend.
    #[cfg(feature = "ndarray")]
    Ndarray(NdarrayModule<N>),
}

/// A backend capable of compiling kernels, tagged by which one it is.
pub enum Compiler {
    /// The LLVM/MLIR backend.
    Llvm(LlvmCompiler),

    /// The `ndarray` backend.
    #[cfg(feature = "ndarray")]
    Ndarray(NdarrayCompiler),
}
