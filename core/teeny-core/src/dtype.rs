/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
