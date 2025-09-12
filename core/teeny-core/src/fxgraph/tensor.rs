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

use crate::fxgraph::{device::Device, dtype::DType, shape::Shape};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Shape,
    pub device: Device,
    pub stride: Vec<u32>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(
        dtype: DType,
        shape: Shape,
        device: Device,
        stride: Vec<u32>,
        requires_grad: bool,
    ) -> Self {
        Self {
            dtype,
            shape,
            device,
            stride,
            requires_grad,
        }
    }
}
