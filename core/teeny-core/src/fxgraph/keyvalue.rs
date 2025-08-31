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

use egg::Id;

use crate::error::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyValue {
    Kv(String, Id),
}

impl FromStr for KeyValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for KeyValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyValue::Kv(k, v) => write!(f, "{}: {}", k, v),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyValueList(Vec<KeyValue>);

impl KeyValueList {
    pub fn new(kvs: Vec<KeyValue>) -> Self {
        Self(kvs)
    }
}

impl IntoIterator for KeyValueList {
    type Item = KeyValue;
    type IntoIter = std::vec::IntoIter<KeyValue>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromStr for KeyValueList {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for KeyValueList {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(|kv| kv.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
