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

use std::ops::{Add, Mul};

use crate::triton::{
    llvm::triton::{
        BoolLike, IntLike,
        num::{I1, I32, I64},
        types::AnyType,
    },
    types::{self as ty, Comparison},
};

/*--------------------------------- Tensor ---------------------------------*/

pub struct Tensor<D: ty::Dtype> {
    _phantom_1: std::marker::PhantomData<D>,
}

impl ty::RankedTensor<AnyType, I32> for Tensor<I32> {}

impl From<Tensor<I32>> for AnyType {
    fn from(_: Tensor<I32>) -> Self {
        todo!()
    }
}

impl From<Tensor<I32>> for I32 {
    fn from(_: Tensor<I32>) -> Self {
        todo!()
    }
}

/*--------------------------------- BoolTensor ---------------------------------*/

pub struct BoolTensor<S: ty::BoolLike, T: ty::AnyType, U: ty::I1<S, T>> {
    _phantom_1: std::marker::PhantomData<S>,
    _phantom_2: std::marker::PhantomData<T>,
    _phantom_3: std::marker::PhantomData<U>,
}

impl ty::BoolTensor<BoolLike, AnyType, I1> for BoolTensor<BoolLike, AnyType, I1> {}

impl ty::RankedTensor<AnyType, I1> for BoolTensor<BoolLike, AnyType, I1> {}

impl From<BoolTensor<BoolLike, AnyType, I1>> for AnyType {
    fn from(_: BoolTensor<BoolLike, AnyType, I1>) -> Self {
        todo!()
    }
}

impl From<BoolTensor<BoolLike, AnyType, I1>> for BoolLike {
    fn from(_: BoolTensor<BoolLike, AnyType, I1>) -> Self {
        todo!()
    }
}

/*--------------------------------- IntTensor ---------------------------------*/

pub struct IntTensor<
    D: ty::Dtype,
    S: ty::IntLike,
    B: ty::BoolLike,
    T: ty::AnyType,
    O: ty::I64<S, T>,
    V: ty::I32<S, T, O>,
    U: ty::I1<B, T>,
    BT: ty::BoolTensor<B, T, U>,
> {
    _phantom_0: std::marker::PhantomData<D>,
    _phantom_1: std::marker::PhantomData<S>,
    _phantom_2: std::marker::PhantomData<B>,
    _phantom_3: std::marker::PhantomData<T>,
    _phantom_4: std::marker::PhantomData<O>,
    _phantom_5: std::marker::PhantomData<V>,
    _phantom_6: std::marker::PhantomData<U>,
    _phantom_7: std::marker::PhantomData<BT>,
}

impl<D: ty::Dtype>
    ty::IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
    for IntTensor<D, IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>
{
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
