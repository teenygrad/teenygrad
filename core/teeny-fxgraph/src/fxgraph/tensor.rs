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
