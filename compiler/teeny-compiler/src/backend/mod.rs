/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use teeny_core::dtype;

use crate::backend::llvm::{compiler::MlirCompiler, module::MlirModule};

#[cfg(feature = "ndarray")]
use crate::backend::ndarray::{compiler::NdarrayCompiler, module::NdarrayModule};

#[cfg(feature = "ndarray")]
pub mod ndarray;

pub mod llvm;

#[derive(Debug, Clone)]
pub enum Module<N: dtype::Dtype> {
    Mlir(MlirModule<N>),

    #[cfg(feature = "ndarray")]
    Ndarray(NdarrayModule<N>),
}

pub enum Compiler<N: dtype::Dtype> {
    Mlir(MlirCompiler<N>),

    #[cfg(feature = "ndarray")]
    Ndarray(NdarrayCompiler<N>),
}
