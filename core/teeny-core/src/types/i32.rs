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

use crate::dtype::{Dtype, DtypeEnum};

impl Dtype for i32 {
    const DTYPE: DtypeEnum = DtypeEnum::I32;

    fn from_f32(value: f32) -> Self {
        value as i32
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }

    fn to_u32(self) -> u32 {
        self as u32
    }
}
