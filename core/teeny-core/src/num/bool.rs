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

use crate::dtype::{Dtype, DtypeEnum};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Bool(pub bool);

impl num_traits::Zero for Bool {
    fn zero() -> Self {
        Bool(false)
    }

    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        Bool(value)
    }
}

impl Add for Bool {
    type Output = Self;

    fn add(self, _other: Self) -> Self {
        unimplemented!()
    }
}

impl Dtype for Bool {
    const DTYPE: DtypeEnum = DtypeEnum::Bool;

    fn from_f32(_value: f32) -> Self {
        unimplemented!()
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        unimplemented!()
    }

    fn to_f32(self) -> f32 {
        unimplemented!()
    }

    fn to_u32(self) -> u32 {
        unimplemented!()
    }
}
