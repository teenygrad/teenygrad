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
pub enum DtypeValue {
    F32,
    F64,
    I32,
    I64,
    Bool,
    Complex64,
    Complex128,
}

impl FromStr for DtypeValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for DtypeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DtypeValue::F32 => write!(f, "f32"),
            DtypeValue::F64 => write!(f, "f64"),
            DtypeValue::I32 => write!(f, "i32"),
            DtypeValue::I64 => write!(f, "i64"),
            DtypeValue::Bool => write!(f, "bool"),
            DtypeValue::Complex64 => write!(f, "c64"),
            DtypeValue::Complex128 => write!(f, "c128"),
        }
    }
}
