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

use derive_more::Display;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

use crate::num::bf16::bf16;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, Display)]
pub enum DtypeEnum {
    #[display("usize")]
    Usize,
    #[display("f32")]
    F32,
    #[display("bf16")]
    Bf16,
}

impl From<(DtypeEnum, DtypeEnum)> for DtypeEnum {
    fn from(value: (DtypeEnum, DtypeEnum)) -> Self {
        match value {
            (DtypeEnum::F32, _) | (_, DtypeEnum::F32) => DtypeEnum::F32,
            (DtypeEnum::Bf16, _) | (_, DtypeEnum::Bf16) => DtypeEnum::Bf16,
            (DtypeEnum::Usize, _) => DtypeEnum::Usize,
        }
    }
}
pub trait Dtype: 'static + Default + Clone + Copy + Zero + std::fmt::Debug {
    const DTYPE: DtypeEnum;

    fn from_bytes(bytes: &[u8]) -> Vec<Self>;

    fn from_f32(value: f32) -> Self;

    fn to_f32(self) -> f32;

    fn to_u32(self) -> u32;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Usize(usize),
    F32(f32),
    Bf16(bf16),
}

impl Dtype for usize {
    const DTYPE: DtypeEnum = DtypeEnum::Usize;

    fn from_f32(value: f32) -> Self {
        value as usize
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

impl Dtype for f32 {
    const DTYPE: DtypeEnum = DtypeEnum::F32;

    fn from_f32(value: f32) -> Self {
        value
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }

    fn to_u32(self) -> u32 {
        self as u32
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
