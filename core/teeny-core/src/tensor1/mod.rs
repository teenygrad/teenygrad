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

use std::{any::Any, sync::Arc};

use crate::{device::Device, tensor1::shape::DynamicShape};

pub mod num;
pub mod ops;
pub mod shape;

use ndarray::IxDyn;

pub type DTensor<E> = dyn Tensor<DType = E, Shape = DynamicShape> + Send + Sync;

pub trait Tensor: Send + Sync + std::fmt::Debug + Any {
    type DType: num::Num;
    type Shape: shape::Shape;

    fn to_device(
        &self,
        device: &Arc<dyn Device>,
    ) -> Result<Arc<DTensor<Self::DType>>, Box<dyn std::error::Error>>;

    fn dtype(&self) -> Self::DType;

    fn shape(&self) -> Self::Shape;

    // Downcast
    fn as_any(&self) -> &dyn Any;

    fn add(&self, other: &DTensor<Self::DType>) -> Arc<DTensor<Self::DType>>;
}

// Wrapper type for ergonomic operations
#[derive(Debug, Clone)]
pub struct TensorRef<T: num::Num>(pub Arc<DTensor<T>>);

impl<T: num::Num> AsRef<DTensor<T>> for TensorRef<T> {
    fn as_ref(&self) -> &DTensor<T> {
        &*self.0
    }
}

impl<T: num::Num> TensorRef<T> {
    pub fn new(tensor: Arc<DTensor<T>>) -> Self {
        Self(tensor)
    }

    pub fn to_device(&self, device: &Arc<dyn Device>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(self.0.to_device(device)?))
    }

    pub fn into_inner(self) -> Arc<DTensor<T>> {
        self.0
    }

    // Convenience method for addition
    pub fn add_tensor(&self, other: &TensorRef<T>) -> TensorRef<T> {
        TensorRef(Tensor::add(&*self.0, &*other.0))
    }

    // Convenience method for addition with Arc
    pub fn add_arc(&self, other: &Arc<DTensor<T>>) -> TensorRef<T> {
        TensorRef(Tensor::add(&*self.0, &**other))
    }
}

impl<T: num::Num> From<Arc<DTensor<T>>> for TensorRef<T> {
    fn from(tensor: Arc<DTensor<T>>) -> Self {
        Self(tensor)
    }
}

impl<T: num::Num> From<TensorRef<T>> for Arc<DTensor<T>> {
    fn from(tensor_ref: TensorRef<T>) -> Self {
        tensor_ref.into_inner()
    }
}

pub fn from_ndarray<T: num::Num>(
    _array: ndarray::ArrayBase<ndarray::OwnedRepr<T>, IxDyn>,
) -> TensorRef<f32> {
    unimplemented!()
}
