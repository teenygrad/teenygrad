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
pub enum ShapeValue {
    Static(Vec<i64>),
    Dynamic(Vec<Option<i64>>), // None represents dynamic dimensions
    Symbolic(Vec<String>),     // For symbolic shapes
}

impl FromStr for ShapeValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for ShapeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeValue::Static(dims) => write!(
                f,
                "[{}]",
                dims.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ShapeValue::Dynamic(dims) => write!(
                f,
                "[{}]",
                dims.iter()
                    .map(|x| match x {
                        Some(d) => d.to_string(),
                        None => "?".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ShapeValue::Symbolic(syms) => write!(f, "[{}]", syms.join(", ")),
        }
    }
}
