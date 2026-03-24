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

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;

use crate::types::bf16::bf16;
use crate::{dtype::DtypeEnum, error::Result};
use crate::{
    graph::shape::DynamicShape,
    graph::{NodeOp, NodeRef, ops::Op},
};

/*--------------------------------- TensorUsize ---------------------------------*/

#[derive(Debug, Clone)]
pub struct TensorUsizeOp {
    #[cfg(feature = "ndarray")]
    pub input: ndarray::Array<usize, IxDyn>,
}

impl TensorUsizeOp {
    #[cfg(feature = "ndarray")]
    pub fn new(input: ndarray::Array<usize, IxDyn>) -> Self {
        Self { input }
    }
}

impl Op for TensorUsizeOp {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::from(self.input.shape()))
    }

    fn dtype(&self) -> DtypeEnum {
        DtypeEnum::Usize
    }
}

impl<'data> From<TensorUsizeOp> for NodeRef<'data> {
    fn from(op: TensorUsizeOp) -> Self {
        NodeOp::TensorUsize(op).into()
    }
}

/*--------------------------------- TensorF32 ---------------------------------*/

#[derive(Debug, Clone)]
pub struct TensorF32Op {
    #[cfg(feature = "ndarray")]
    pub input: ndarray::Array<f32, IxDyn>,
}

impl TensorF32Op {
    #[cfg(feature = "ndarray")]
    pub fn new(input: ndarray::Array<f32, IxDyn>) -> Self {
        Self { input }
    }
}

impl Op for TensorF32Op {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::from(self.input.shape()))
    }

    fn dtype(&self) -> DtypeEnum {
        DtypeEnum::F32
    }
}

impl<'data> From<TensorF32Op> for NodeRef<'data> {
    fn from(op: TensorF32Op) -> Self {
        NodeOp::TensorF32(op).into()
    }
}

/*--------------------------------- TensorBF16 ---------------------------------*/

#[derive(Debug, Clone)]
pub struct TensorBF16Op {
    #[cfg(feature = "ndarray")]
    pub input: ndarray::Array<bf16, IxDyn>,
}

impl TensorBF16Op {
    #[cfg(feature = "ndarray")]
    pub fn new(input: ndarray::Array<bf16, IxDyn>) -> Self {
        Self { input }
    }
}

impl Op for TensorBF16Op {
    fn shape(&self) -> Result<DynamicShape> {
        Ok(DynamicShape::from(self.input.shape()))
    }

    fn dtype(&self) -> DtypeEnum {
        DtypeEnum::Bf16
    }
}

impl<'data> From<TensorBF16Op> for NodeRef<'data> {
    fn from(op: TensorBF16Op) -> Self {
        NodeOp::TensorBF16(op).into()
    }
}
