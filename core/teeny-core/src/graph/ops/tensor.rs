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

use std::marker::PhantomData;

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;

use crate::error::Result;
use crate::{
    graph::{NodeOp, NodeRef, ops::OpShape},
    tensor::shape::DynamicShape,
};

#[derive(Debug, Clone)]
pub struct TensorOpF32 {
    #[cfg(feature = "ndarray")]
    pub input: ndarray::Array<f32, IxDyn>,
}

impl TensorOpF32 {
    #[cfg(feature = "ndarray")]
    pub fn new(input: ndarray::Array<f32, IxDyn>) -> Self {
        Self { input }
    }
}

impl OpShape for TensorOpF32 {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::new(&[1, self.input.len()]))
    }
}

impl<'data> From<TensorOpF32> for NodeRef<'data> {
    fn from(op: TensorOpF32) -> Self {
        NodeOp::Tensor(op).into()
    }
}
