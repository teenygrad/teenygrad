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

use ndarray::IxDyn;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr;

use crate::error::Error;
use crate::{
    device::Device,
    dtype,
    error::Result,
    tensor::{
        Tensor,
        shape::{self, Shape},
    },
};

pub mod add;

#[cfg(feature = "training")]
#[derive(Debug)]
pub struct AutogradContext<D: Device, N: dtype::Dtype> {
    pub activation: ndarray::Array<N, IxDyn>,
    pub grad: Option<ndarray::Array<N, IxDyn>>,
    pub requires_grad: bool,
    pub retain_grad: bool,
    _marker: std::marker::PhantomData<D>,
}

#[derive(Debug)]
pub struct NdarrayTensor<D: Device, N: dtype::Dtype> {
    pub data: ndarray::Array<N, IxDyn>,
    #[cfg(feature = "training")]
    pub autograd_context: Option<AutogradContext<D, N>>,
    _marker: std::marker::PhantomData<D>,
}

impl<D: Device, N: dtype::Dtype> NdarrayTensor<D, N> {
    pub fn new(data: ndarray::Array<N, IxDyn>) -> Self {
        Self {
            data,
            #[cfg(feature = "training")]
            autograd_context: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D: Device, N: dtype::Dtype> Tensor<D, N> for NdarrayTensor<D, N> {
    fn zeros<S: shape::Shape>(shape: S) -> Result<Self> {
        Ok(Self {
            data: ndarray::Array::zeros::<IxDyn>(shape.to_ndarray_shape()),
            #[cfg(feature = "training")]
            autograd_context: None,
            _marker: std::marker::PhantomData,
        })
    }

    fn randn<S: shape::Shape>(shape: S) -> Result<Self> {
        let distribution =
            rand_distr::Normal::new(0.0f32, 1.0).map_err(|e| Error::NdarrayError(e.to_string()))?;
        let data = ndarray::Array::random(shape.to_ndarray_shape(), distribution).into_dyn();

        Ok(Self {
            data: data.mapv(|x| N::from(x).unwrap_or_default()),
            #[cfg(feature = "training")]
            autograd_context: None,
            _marker: std::marker::PhantomData,
        })
    }

    fn arange(start: N, end: N, step: N) -> Result<Self> {
        Ok(Self {
            data: ndarray::Array::range(start, end, step).into_dyn(),
            #[cfg(feature = "training")]
            autograd_context: None,
            _marker: std::marker::PhantomData,
        })
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
