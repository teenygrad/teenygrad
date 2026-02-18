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
