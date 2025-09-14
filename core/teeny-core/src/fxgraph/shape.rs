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
