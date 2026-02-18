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

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    F32,
    BF16,
    Bool,
}

impl DType {
    pub fn promote(&self, other: &DType) -> Result<DType, Error> {
        match (self, other) {
            (DType::F32, DType::F32) => Ok(DType::F32),
            (DType::BF16, DType::BF16) => Ok(DType::BF16),
            (DType::F32, DType::BF16) | (DType::BF16, DType::F32) => Ok(DType::F32),
            (DType::Bool, DType::Bool) => Ok(DType::Bool),
            _ => Err(Error::InvalidTypeConversion(format!(
                "Cannot promote dtype: {} and {}",
                self, other
            ))),
        }
    }
}

impl FromStr for DType {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::BF16 => write!(f, "bf16"),
            DType::Bool => write!(f, "bool"),
        }
    }
}
