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

pub trait Device: Sized + std::fmt::Debug {
    type Tensor<T: num::Num>: Tensor<Self, T>;

    fn from_ndarray<T: num::Num>(ndarray: ndarray::Array<T, IxDyn>) -> Self::Tensor<T>;
}

pub trait Tensor<D: Device, T: num::Num>: Sized + std::fmt::Debug {
    type DType: num::Num;
    type Shape: shape::Shape;

    fn to<ToD: Device>(self, device: &ToD) -> impl Tensor<ToD, T>;

    fn dtype(&self) -> Self::DType;

    fn shape(&self) -> Self::Shape;

    fn add(&self, other: &impl Tensor<D, T>) -> impl Tensor<D, T>;
}

pub fn from_ndarray<D: Device, T: num::Num>(
    ndarray: ndarray::Array<T, IxDyn>,
) -> impl Tensor<D, T> {
    D::from_ndarray(ndarray)
}
