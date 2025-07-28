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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Display)]
pub enum DtypeEnum {
    #[display("i8")]
    I8,
    #[display("u8")]
    U8,
    #[display("i16")]
    I16,
    #[display("u16")]
    U16,
    #[display("i32")]
    I32,
    #[display("u32")]
    U32,
    #[display("i64")]
    I64,
    #[display("u64")]
    U64,
    #[display("f32")]
    F32,
    #[display("f64")]
    F64,
    #[display("bf16")]
    Bf16,
}
pub trait Dtype: 'static + Default + Clone + Copy + Zero + std::fmt::Debug {
    const DTYPE: DtypeEnum;

    fn from_bytes(bytes: &[u8]) -> Vec<Self>;

    fn from_f32(value: f32) -> Self;

    fn to_f32(self) -> f32;

    fn to_u32(self) -> u32;
}

impl Dtype for i8 {
    const DTYPE: DtypeEnum = DtypeEnum::I8;

    fn from_f32(value: f32) -> Self {
        value as i8
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn to_u32(self) -> u32 {
        self as u32
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }
}

impl Dtype for usize {
    #[cfg(target_pointer_width = "32")]
    const DTYPE: DtypeEnum = DtypeEnum::U32;

    #[cfg(target_pointer_width = "64")]
    const DTYPE: DtypeEnum = DtypeEnum::U64;

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

impl Dtype for isize {
    #[cfg(target_pointer_width = "32")]
    const DTYPE: DtypeEnum = DtypeEnum::I32;

    #[cfg(target_pointer_width = "64")]
    const DTYPE: DtypeEnum = DtypeEnum::I64;

    fn from_f32(value: f32) -> Self {
        value as isize
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn to_u32(self) -> u32 {
        self as u32
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }
}

impl Dtype for i32 {
    const DTYPE: DtypeEnum = DtypeEnum::I32;

    fn from_f32(value: f32) -> Self {
        value as i32
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn to_u32(self) -> u32 {
        self as u32
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }
}

impl Dtype for u32 {
    const DTYPE: DtypeEnum = DtypeEnum::U32;

    fn from_f32(value: f32) -> Self {
        value as u32
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn to_u32(self) -> u32 {
        self
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }
}

impl Dtype for i64 {
    const DTYPE: DtypeEnum = DtypeEnum::I64;

    fn from_f32(value: f32) -> Self {
        value as i64
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

impl Dtype for u64 {
    const DTYPE: DtypeEnum = DtypeEnum::U64;

    fn from_f32(value: f32) -> Self {
        value as u64
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

impl Dtype for f64 {
    const DTYPE: DtypeEnum = DtypeEnum::F64;

    fn from_f32(value: f32) -> Self {
        value as f64
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

impl Dtype for bf16 {
    const DTYPE: DtypeEnum = DtypeEnum::Bf16;

    fn from_f32(value: f32) -> Self {
        bf16(half::bf16::from_f32(value))
    }

    fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    fn from_bytes(_bytes: &[u8]) -> Vec<Self> {
        todo!()
    }

    fn to_u32(self) -> u32 {
        self.0.to_f32() as u32
    }
}
