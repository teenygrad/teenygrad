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

use crate::triton::{
    llvm::triton::{
        BoolLike,
        num::{I1, I32, I64, Int, IntLike},
        types::{AnyType, Bool},
    },
    types::{self as ty},
};

/*--------------------------------- Tensor ---------------------------------*/

pub struct Tensor<D: ty::Dtype> {
    _phantom_1: std::marker::PhantomData<D>,
}

impl<D: ty::Dtype> ty::RankedTensor<D> for Tensor<D> {
    type AnyType = AnyType;
}

/*--------------------------------- BoolTensor ---------------------------------*/

pub struct BoolTensor<B: ty::Bool, T: ty::AnyType, BL: ty::BoolLike> {
    _phantom_1: std::marker::PhantomData<B>,
    _phantom_2: std::marker::PhantomData<T>,
    _phantom_3: std::marker::PhantomData<BL>,
}

impl ty::BoolTensor for BoolTensor<Bool, AnyType, BoolLike> {
    type Bool = Bool;
    type AnyType = AnyType;
    type BoolLike = BoolLike;
}

impl ty::RankedTensor<Bool> for BoolTensor<Bool, AnyType, BoolLike> {
    type AnyType = AnyType;
}

impl From<BoolTensor<Bool, AnyType, BoolLike>> for BoolLike {
    fn from(_: BoolTensor<Bool, AnyType, BoolLike>) -> Self {
        todo!()
    }
}

impl From<BoolTensor<Bool, AnyType, BoolLike>> for AnyType {
    fn from(_: BoolTensor<Bool, AnyType, BoolLike>) -> Self {
        todo!()
    }
}

impl ty::RankedTensor<I1> for BoolTensor<Bool, AnyType, BoolLike> {
    type AnyType = AnyType;
}

/*--------------------------------- IntTensor ---------------------------------*/
pub struct IntTensor<I: ty::Int, IL: ty::IntLike, I32: ty::I32, I64: ty::I64, BT: ty::BoolTensor> {
    _phantom_0: std::marker::PhantomData<I>,
    _phantom_1: std::marker::PhantomData<IL>,
    _phantom_2: std::marker::PhantomData<I32>,
    _phantom_3: std::marker::PhantomData<I64>,
    _phantom_4: std::marker::PhantomData<BT>,
}

impl<D: ty::Dtype> ty::IntTensor
    for IntTensor<Self::Int, Self::IntLike, Self::I32, Self::I64, Self::BoolTensor>
{
    type Int = Int;
    type IntLike = IntLike;

    type I32 = I32;
    type I64 = I64;
    type BoolTensor = BoolTensor<Bool, AnyType, BoolLike>;
}

impl<D: ty::Dtype> ty::RankedTensor<AnyType, I32>
    for IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
{
}

impl<D: ty::Dtype>
    From<IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>>
    for AnyType
{
    fn from(
        _: IntTensor<
            D,
            IntLike,
            BoolLike,
            AnyType,
            I64,
            I32,
            I1,
            BoolTensor<BoolLike, AnyType, I1>,
        >,
    ) -> Self {
        todo!()
    }
}

impl<D: ty::Dtype>
    From<IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>>
    for IntLike
{
    fn from(
        _: IntTensor<
            D,
            IntLike,
            BoolLike,
            AnyType,
            I64,
            I32,
            I1,
            BoolTensor<BoolLike, AnyType, I1>,
        >,
    ) -> Self {
        todo!()
    }
}

impl<D: ty::Dtype> Mul<I64>
    for IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
{
    type Output =
        IntTensor<I64, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>;

    fn mul(self, _other: I64) -> Self::Output {
        todo!()
    }
}

impl<D: ty::Dtype> Add<I64>
    for IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
{
    type Output =
        IntTensor<I64, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>;

    fn add(self, _other: I64) -> Self::Output {
        todo!()
    }
}

impl<D: ty::Dtype> Comparison<I64>
    for IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
{
    type Output = BoolTensor<BoolLike, AnyType, I1>;

    fn eq(&self, _other: I64) -> Self::Output {
        todo!()
    }

    fn ne(&self, _other: I64) -> Self::Output {
        todo!()
    }

    fn slt(&self, _other: I64) -> Self::Output {
        todo!()
    }

    fn sle(&self, _other: I64) -> Self::Output {
        todo!()
    }

    fn sgt(&self, _other: I64) -> Self::Output {
        todo!()
    }

    fn sge(&self, _other: I64) -> Self::Output {
        todo!()
    }
}

impl<D: ty::Dtype> Comparison<I32>
    for IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
{
    type Output = BoolTensor<BoolLike, AnyType, I1>;

    fn eq(&self, _other: I32) -> Self::Output {
        todo!()
    }

    fn ne(&self, _other: I32) -> Self::Output {
        todo!()
    }

    fn slt(&self, _other: I32) -> Self::Output {
        todo!()
    }

    fn sle(&self, _other: I32) -> Self::Output {
        todo!()
    }

    fn sgt(&self, _other: I32) -> Self::Output {
        todo!()
    }

    fn sge(&self, _other: I32) -> Self::Output {
        todo!()
    }
}
