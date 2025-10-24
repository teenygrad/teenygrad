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

pub trait Dtype: 'static {}

impl Dtype for i32 {}
impl Dtype for f32 {}

pub trait Tensor<D: Dtype> {}

impl<T: Tensor<i32>> LegacyReceiver for &T {}

pub struct TensorImpl<D: Dtype> {
    pub x: D,
}

impl<T: Dtype> Tensor<T> for TensorImpl<T> {}

impl<D: Dtype, T: Tensor<i32>> Add<&T> for &Pointer<D> {
    type Output = Pointer<D>;

    #[inline(never)]
    fn add(self, _other: &T) -> Self::Output {
        loop {}
    }
}

pub struct Pointer<D: Dtype> {
    ptr: *mut D,
}

impl<D: Dtype> Add<Pointer<D>> for Pointer<D> {
    type Output = Pointer<D>;

    #[inline(never)]
    fn add(self, other: Pointer<D>) -> Self::Output {
        other
    }
}

pub enum Mask<T: Tensor<i32>> {
    None,
    Some(T),
}

mod tl {
    pub use super::*;

    pub enum ProgramAxis {
        Axis0,
        Axis1,
        Axis2,
    }

    #[inline(never)]
    pub fn program_id(_axis: ProgramAxis) -> i32 {
        0
    }

    #[inline(never)]
    pub fn num_programs(_axis: ProgramAxis) -> i32 {
        0
    }

    #[inline(never)]
    pub fn load<D: Dtype, MT: Tensor<i32>>(_ptr: Pointer<D>, _mask: &Mask<MT>) -> Pointer<D> {
        loop {}
    }

    #[inline(never)]
    pub fn store<D: Dtype, MT: Tensor<i32>>(
        _ptr: Pointer<D>,
        _ptr1: Pointer<D>,
        _mask: &Mask<MT>,
    ) -> Pointer<D> {
        loop {}
    }

    #[inline(never)]
    pub fn arange<T: Tensor<i32>>(_start: i32, _end: i32) -> T {
        loop {}
    }
}

use crate::tl::*;
