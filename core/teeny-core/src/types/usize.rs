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

use crate::dtype::{Dtype, DtypeEnum};

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
