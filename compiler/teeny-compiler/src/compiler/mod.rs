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

use crate::error::Result;
use teeny_core::nn::module::Module;

pub mod mlir;
pub mod module;

#[cfg(feature = "ndarray")]
pub mod ndarray;

pub trait Compiler {
    fn compile<M1: Module<T, U>, M2: module::Module>(
        &self,
        target: Target,
        module: M1,
        samples: &[T],
    ) -> Result<M2>;
}
