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

use crate::dtype;
use crate::graph::NodeRef;
use crate::tensor::Tensor;
pub trait Module<N: dtype::Dtype, T, U> {
    type Err;

    fn forward(&self, x: T) -> Result<U, Self::Err>;

    fn parameters(&self) -> Vec<NodeRef<N>>;
}

pub type NodeRefModule<N, Error> = Box<dyn Module<N, NodeRef<N>, NodeRef<N>, Err = Error>>;

pub trait CompiledModule<N: dtype::Dtype, T: Tensor<N>, U: Tensor<N>> {
    type Err;

    fn forward(&self, x: T) -> Result<U, Self::Err>;
}
