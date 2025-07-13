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

pub mod num;
pub mod ops;
pub mod shape;

pub trait Device<T: num::Num>: Sized + std::fmt::Debug {
    type Tensor: Tensor<T, Self>;

    fn from_ndarray(ndarray: ndarray::Array<T, IxDyn>) -> Self::Tensor;
}

pub trait Tensor<T: num::Num, D: Device<T>>: Sized + std::fmt::Debug {
    type DType: num::Num;

    // fn to<ToD: Device>(self, device: &ToD) -> impl Tensor<ToD, T>;
    fn add(&self, other: &Self) -> Self;
}

pub fn from_ndarray<T: num::Num, D: Device<T>>(
    ndarray: ndarray::Array<T, IxDyn>,
) -> impl Tensor<T, D> {
    D::from_ndarray(ndarray)
}
