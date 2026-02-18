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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymInt {
    Int(i64),
    Sym(String),
}

impl From<i32> for SymInt {
    fn from(value: i32) -> Self {
        SymInt::Int(value as i64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape {
    pub shape: Vec<SymInt>,
}

impl Shape {
    pub fn new(shape: &[SymInt]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl FromStr for Shape {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        format!("{:?}", self).fmt(f)
    }
}
