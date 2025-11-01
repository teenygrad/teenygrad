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

pub mod types;

pub trait Dtype: Sized + Copy {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct I32(pub i32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32(pub f32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bool(pub bool);

impl Dtype for I32 {}
impl Dtype for F32 {}
impl Dtype for Bool {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramAxis {
    Axis0,
    Axis1,
    Axis2,
}

pub trait Triton {
    type Pointer<D: Dtype>: Sized;
    type Tensor<D: Dtype>: Sized;

    fn program_id(axis: ProgramAxis) -> I32;

    fn num_programs(axis: ProgramAxis) -> I32;

    fn arange(start: I32, end: I32) -> Self::Tensor<I32>;

    fn load<D: Dtype>(
        ptr: &Self::Pointer<D>,
        mask: &Option<Self::Tensor<Bool>>,
    ) -> Self::Pointer<D>;

    fn store<D: Dtype>(
        src: &Self::Pointer<D>,
        dest: &mut Self::Pointer<D>,
        mask: &Option<Self::Tensor<Bool>>,
    );
}
