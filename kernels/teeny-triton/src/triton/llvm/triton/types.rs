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

use crate::triton::types as ty;
pub enum AnyType {}

impl ty::AnyType for AnyType {}

/*--------------------------------- Bool ---------------------------------*/

pub enum BoolLike {}

impl ty::BoolLike for BoolLike {}
pub struct Bool(bool);

impl ty::Dtype for Bool {}

impl ty::Bool for Bool {
    type AnyType = AnyType;
    type BoolLike = BoolLike;
}

impl Clone for Bool {
    fn clone(&self) -> Self {
        Bool(self.0)
    }
}

impl Copy for Bool {}

impl From<Bool> for AnyType {
    fn from(_value: Bool) -> Self {
        todo!()
    }
}

impl From<Bool> for BoolLike {
    fn from(_value: Bool) -> Self {
        todo!()
    }
}
