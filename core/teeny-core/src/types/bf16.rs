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
#[allow(non_camel_case_types)]
pub struct bf16(pub half::bf16);
impl num_traits::Zero for bf16 {
    fn zero() -> Self {
        bf16(half::bf16::ZERO)
    }

    fn is_zero(&self) -> bool {
        half::bf16::ZERO.eq(&self.0)
    }
}

impl Add for bf16 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        bf16(self.0 + other.0)
    }
}

impl Dtype for bf16 {
    const DTYPE: DtypeEnum = DtypeEnum::Bf16;

    fn from_f32(value: f32) -> Self {
        bf16(half::bf16::from_f32(value))
    }

    fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        let mut data = Vec::new();
        assert_eq!(bytes.len() % 2, 0);
        for i in 0..bytes.len() / 2 {
            let value = half::bf16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            data.push(bf16(value));
        }
        data
    }

    fn to_u32(self) -> u32 {
        self.0.to_f32() as u32
    }
}
