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

use ordered_float::OrderedFloat;

use crate::errors::Error;

// Value types for constants
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstantValue {
    Int(i64),
    Float32(OrderedFloat<f32>),
    Bool(bool),
    IntList(Vec<i64>),
    String(String),
    // FloatList(Vec<OrderedFloat>),
}

impl FromStr for ConstantValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for ConstantValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantValue::Int(i) => write!(f, "{i}"),
            ConstantValue::Bool(b) => write!(f, "{b}"),
            ConstantValue::IntList(list) => write!(
                f,
                "[{}]",
                list.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ConstantValue::String(s) => write!(f, "{s}"),
            ConstantValue::Float32(s) => write!(f, "{s}"),
        }
    }
}
