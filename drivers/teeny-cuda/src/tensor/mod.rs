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

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;
use teeny_core::{
    dtype,
    tensor::{Tensor, shape},
};

use crate::device::CudaDevice;

#[derive(Debug)]
pub struct CudaTensor<T: dtype::Dtype> {
    _marker: std::marker::PhantomData<T>,
}

impl<N: dtype::Dtype> Tensor<CudaDevice, N> for CudaTensor<N> {
    fn zeros<S: shape::Shape>(_shape: S) -> teeny_core::error::Result<Self> {
        todo!()
    }

    fn randn<S: shape::Shape>(_shape: S) -> teeny_core::error::Result<Self> {
        todo!()
    }

    fn arange(_start: N, _end: N, _step: N) -> teeny_core::error::Result<Self> {
        todo!()
    }
}

impl<N: dtype::Dtype> Add<CudaTensor<N>> for CudaTensor<N> {
    type Output = CudaTensor<N>;

    fn add(self, _other: CudaTensor<N>) -> Self::Output {
        todo!()
    }
}

#[cfg(feature = "ndarray")]
impl<T: dtype::Dtype> From<ndarray::Array<T, IxDyn>> for CudaTensor<T> {
    fn from(_data: ndarray::Array<T, IxDyn>) -> Self {
        unimplemented!()
    }
}
