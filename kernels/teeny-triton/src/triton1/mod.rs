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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramAxis {
    Axis0,
    Axis1,
    Axis2,
}

pub trait Triton {
    type AnyType: types::AnyType;
    type IntLike: types::IntLike;
    type I64Like: types::I64Like;
    type PointerLike: types::PointerLike;

    type Pointer<D: Dtype>: types::Pointer<Self::PointerLike, Self::AnyType>;
    type Tensor<D: Dtype>;

    type I32: types::I32<Self::IntLike, Self::AnyType, Self::I64>;
    type I64: types::I64<Self::IntLike, Self::AnyType>;

    type IntTensor: types::IntTensor<Self::IntLike, Self::AnyType, Self::I64, Self::I32>;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;

    fn arange<S1, S2>(start: S1, end: S2) -> Self::IntTensor
    where
        S1: Into<Self::I32>,
        S2: Into<Self::I32>;

    fn load<D: Dtype>(ptr: &Self::Pointer<D>, mask: &Option<Self::IntTensor>) -> Self::Pointer<D>;

    // fn store<D: Dtype>(
    //     src: &Self::Pointer<D>,
    //     dest: &mut Self::Pointer<D>,
    //     mask: &Option<Self::Tensor<Bool>>,
    // );
}
