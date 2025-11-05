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

use super::types as ty;

pub enum AnyType<D: ty::Dtype> {
    Tensor(RankedTensor<D>),
}

impl<D: ty::Dtype> ty::AnyType for AnyType<D> {}

pub struct RankedTensor<D: ty::Dtype> {
    _phantom_2: std::marker::PhantomData<D>,
}

impl<D: ty::Dtype> ty::RankedTensor<AnyType<D>, D> for RankedTensor<D> {}

impl<D: ty::Dtype> From<RankedTensor<D>> for AnyType<D> {
    fn from(tensor: RankedTensor<D>) -> Self {
        AnyType::Tensor(tensor)
    }
}
