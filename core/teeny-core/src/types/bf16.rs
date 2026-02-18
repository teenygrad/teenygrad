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
