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

use crate::{
    device::{Device, get_current_device},
    tensor::{DenseTensor, DynamicShape, Shape, Tensor},
    types::NumericType,
};

pub fn empty_like<T: NumericType>(
    _tensor: &DenseTensor<DynamicShape, T>,
) -> DenseTensor<DynamicShape, T> {
    unimplemented!()
}

pub fn empty<T: NumericType>(_shape: &[usize]) -> DenseTensor<DynamicShape, T> {
    unimplemented!()
}

pub fn cdiv(_a: usize, _b: usize) -> usize {
    unimplemented!()
}

pub fn exp<S: Shape, T: Tensor<S>>(_a: f32, _b: T) -> T {
    unimplemented!()
}

pub fn inv<S: Shape, T: Tensor<S>>(_a: f32, _b: T) -> T {
    unimplemented!()
}

pub fn from_ndarray<S: ndarray::Dimension, T: NumericType + 'static>(
    arr: &ndarray::ArrayBase<ndarray::ViewRepr<&T::RustType>, S>,
) -> crate::error::Result<Box<dyn Tensor<DynamicShape, Element = T> + Send + Sync>>
where
    T::RustType: Send + Sync + Clone,
{
    let device = get_current_device()?;
    match *device {
        Device::Cpu => {
            // Extract data from ndarray - convert to owned array first
            let owned_arr = arr.to_owned();
            let data = owned_arr.into_raw_vec_and_offset().0;

            // Extract shape from ndarray
            let shape = arr.shape().to_vec();
            let tensor = DenseTensor::new(data, shape);

            Ok(Box::new(tensor))
        }
    }
}
