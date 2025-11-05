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

pub mod dummy;
pub mod types;

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
    type BoolLike: types::BoolLike;
    type PointerLike: types::PointerLike;

    type Pointer<D: types::Dtype>: types::Pointer<
            D,
            Self::PointerLike,
            Self::IntLike,
            Self::BoolLike,
            Self::AnyType,
            Self::I64,
            Self::I32,
            Self::I1,
            Self::BoolTensor,
        >;

    type I1: types::I1<Self::BoolLike, Self::AnyType>;
    type I32: types::I32<Self::IntLike, Self::AnyType, Self::I64>;
    type I64: types::I64<Self::IntLike, Self::AnyType>;

    type Tensor<D: types::Dtype>: types::RankedTensor<Self::AnyType, D>;
    type BoolTensor: types::BoolTensor<Self::BoolLike, Self::AnyType, Self::I1>;
    type IntTensor<'a, D: types::Dtype>: types::IntTensor<
            Self::IntLike,
            Self::BoolLike,
            Self::AnyType,
            Self::I64,
            Self::I32,
            Self::I1,
            Self::BoolTensor,
        >;

    fn program_id(axis: ProgramAxis) -> Self::I32;

    fn num_programs(axis: ProgramAxis) -> Self::I32;

    fn arange<'a, S1, S2>(start: S1, end: S2) -> Self::IntTensor<'a, Self::I32>
    where
        S1: Into<Self::I32>,
        S2: Into<Self::I32>;

    fn load<'a, D: types::Dtype>(
        ptr: &Self::Pointer<D>,
        mask: &Option<Self::BoolTensor>,
    ) -> Self::Pointer<D>;

    fn store<'a, D: types::Dtype>(
        src: &Self::Pointer<D>,
        dest: &Self::Pointer<D>,
        mask: &Option<Self::BoolTensor>,
    );
}
