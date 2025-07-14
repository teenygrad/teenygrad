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

use crate::tensor::{Device, Tensor, num};

pub trait Module<T: num::Num, D: Device<T>, U> {
    type ParamCollection: Iterator<Item = Self::ParamTensor>;
    type ParamTensor: Tensor<T, D>;

    /// Forward pass that returns a computation graph node
    fn forward(&self) -> U;

    fn parameters(&self) -> Self::ParamCollection;
}

/// Trait for all neural network components
pub trait Module1<T: num::Num, D: Device<T>, U, V> {
    type ParamCollection: Iterator<Item = Self::ParamTensor>;
    type ParamTensor: Tensor<T, D>;

    /// Forward pass that returns a computation graph node
    fn forward(&self, input: U) -> V;

    fn parameters(&self) -> Self::ParamCollection;
}
