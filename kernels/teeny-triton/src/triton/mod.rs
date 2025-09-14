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

use teeny_core::dtype::Dtype;

pub struct Buffer<T: Dtype> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Dtype, D: Dtype> Add<Tensor<T>> for Buffer<D> {
    type Output = Buffer<T>;

    fn add(self, _other: Tensor<T>) -> Buffer<T> {
        todo!()
    }
}
pub struct Tensor<T> {
    _phantom: std::marker::PhantomData<T>,
}

pub fn program_id(_axis: usize) -> usize {
    todo!()
}

pub fn arange<T: Dtype>(_start: usize, _end: usize) -> Tensor<T> {
    todo!()
}

pub fn load<T: Dtype>(_ptr: &Buffer<T>, _mask: bool) {
    todo!()
}

pub fn store<T: Dtype>(_ptr: &Buffer<T>, _value: usize, _mask: bool) {
    todo!()
}
