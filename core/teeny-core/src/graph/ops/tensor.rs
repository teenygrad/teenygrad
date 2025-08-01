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

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;

#[cfg(feature = "ndarray")]
use crate::num::bf16::bf16;
use crate::{dtype::DtypeEnum, error::Result};
use crate::{
    graph::{NodeOp, NodeRef, ops::Op},
    tensor::shape::DynamicShape,
};

/*--------------------------------- TensorNodeRef ---------------------------------*/

#[derive(Debug, Clone)]
pub struct TensorNodeRefOp<'data> {
    #[cfg(feature = "ndarray")]
    pub input: ndarray::Array<NodeRef<'data>, IxDyn>,
}

impl<'data> TensorNodeRefOp<'data> {
    #[cfg(feature = "ndarray")]
    pub fn new(input: ndarray::Array<NodeRef<'data>, IxDyn>) -> Self {
        Self { input }
    }
}

impl<'data> Op for TensorNodeRefOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<TensorNodeRefOp<'data>> for NodeRef<'data> {
    fn from(op: TensorNodeRefOp<'data>) -> Self {
        NodeOp::TensorNodeRef(op).into()
    }
}

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
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
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
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
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
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        todo!()
    }
}

impl<'data> From<TensorBF16Op> for NodeRef<'data> {
    fn from(op: TensorBF16Op) -> Self {
        NodeOp::TensorBF16(op).into()
    }
}
