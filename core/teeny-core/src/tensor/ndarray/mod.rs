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

use std::ops::Add;

use ndarray::IxDyn;

use crate::{
    device::Device,
    dtype,
    tensor::{
        Tensor,
        shape::{self, Shape},
    },
};

#[derive(Debug)]
pub struct NdarrayTensor<D: Device, N: dtype::Dtype> {
    pub data: ndarray::Array<N, IxDyn>,
    _marker: std::marker::PhantomData<D>,
}

impl<D: Device, N: dtype::Dtype> Tensor<D, N> for NdarrayTensor<D, N> {
    fn zeros<S: shape::Shape>(shape: S) -> Self {
        Self {
            data: ndarray::Array::zeros::<IxDyn>(shape.to_ndarray_shape()),
            _marker: std::marker::PhantomData,
        }
    }

    fn randn<S: shape::Shape>(_shape: S) -> Self {
        todo!()
    }

    fn arange(_start: N, _end: N, _step: N) -> Self {
        todo!()
    }
}

impl<D: Device, N: dtype::Dtype> Add<NdarrayTensor<D, N>> for NdarrayTensor<D, N> {
    type Output = Self;

    fn add(self, _other: Self) -> Self::Output {
        NdarrayTensor {
            data: self.data + _other.data,
            _marker: std::marker::PhantomData,
        }
    }
}

pub trait ToNdarrayShape {
    fn to_ndarray_shape(&self) -> ndarray::IxDyn;
}

// Implement it for your Shape types
impl<S: Shape> ToNdarrayShape for S {
    fn to_ndarray_shape(&self) -> ndarray::IxDyn {
        ndarray::IxDyn(self.dims().as_ref())
    }
}
