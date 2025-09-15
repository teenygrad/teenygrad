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

pub use crate::types::*;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Display)]
pub enum DtypeEnum {
    #[display("usize")]
    Usize,
    #[display("i32")]
    I32,
    #[display("f32")]
    F32,
    #[display("bf16")]
    Bf16,
    #[display("bool")]
    Bool,
    #[display("default")]
    Default,
    #[display("tensor")]
    Tensor(Box<DtypeEnum>),
}

impl From<(DtypeEnum, DtypeEnum)> for DtypeEnum {
    fn from(value: (DtypeEnum, DtypeEnum)) -> Self {
        match value {
            (DtypeEnum::Bool, _) | (_, DtypeEnum::Bool) => DtypeEnum::Bool,
            (DtypeEnum::F32, _) | (_, DtypeEnum::F32) => DtypeEnum::F32,
            (DtypeEnum::Bf16, _) | (_, DtypeEnum::Bf16) => DtypeEnum::Bf16,
            (DtypeEnum::Usize, _) => DtypeEnum::Usize,
            (DtypeEnum::Default, _) => DtypeEnum::Default,
            _ => todo!(),
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
