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
        BoolLike, IntLike, PointerLike,
        num::{I1, I32, I64},
        tensor::BoolTensor,
        types::AnyType,
    },
    types as ty,
};
pub struct Pointer<
    D: ty::Dtype,
    PL: ty::PointerLike,
    S: ty::IntLike,
    B: ty::BoolLike,
    T: ty::AnyType,
    O: ty::I64<S, T>,
    V: ty::I32<S, T, O>,
    U: ty::I1<B, T>,
    BT: ty::BoolTensor<B, T, U>,
> {
    _phantom_1: std::marker::PhantomData<D>,
    _phantom_2: std::marker::PhantomData<PL>,
    _phantom_3: std::marker::PhantomData<S>,
    _phantom_4: std::marker::PhantomData<B>,
    _phantom_5: std::marker::PhantomData<T>,
    _phantom_6: std::marker::PhantomData<O>,
    _phantom_7: std::marker::PhantomData<V>,
    _phantom_8: std::marker::PhantomData<U>,
    _phantom_9: std::marker::PhantomData<BT>,
}

impl<D: ty::Dtype>
    ty::Pointer<
        D,
        PointerLike,
        IntLike,
        BoolLike,
        AnyType,
        I64,
        I32,
        I1,
        BoolTensor<BoolLike, AnyType, I1>,
    >
    for Pointer<
        D,
        PointerLike,
        IntLike,
        BoolLike,
        AnyType,
        I64,
        I32,
        I1,
        BoolTensor<BoolLike, AnyType, I1>,
    >
{
    fn add(&self, _other: &Self) -> Self {
        todo!()
    }

    fn add_offsets<
        IT: ty::IntTensor<IntLike, BoolLike, AnyType, I64, I32, I1, BoolTensor<BoolLike, AnyType, I1>>,
    >(
        &self,
        _other: &IT,
    ) -> Self {
        todo!()
    }
}

impl<D: ty::Dtype>
    From<
        Pointer<
            D,
            PointerLike,
            IntLike,
            BoolLike,
            AnyType,
            I64,
            I32,
            I1,
            BoolTensor<BoolLike, AnyType, I1>,
        >,
    > for PointerLike
{
    fn from(
        _value: Pointer<
            D,
            PointerLike,
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
