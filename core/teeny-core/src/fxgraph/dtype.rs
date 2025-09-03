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

use once_cell::sync::Lazy;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::error::Error;

use z3::{Sort, Symbol};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    F32,
    BF16,
    Bool,
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

pub struct DTypeSort {
    pub sort: Sort,
}

unsafe impl Send for DTypeSort {}
unsafe impl Sync for DTypeSort {}

impl DTypeSort {
    pub fn new() -> Self {
        Self {
            sort: Sort::uninterpreted(Symbol::String("DType".to_string())),
        }
    }
}

pub static DTYPE_SORT: Lazy<Arc<Mutex<DTypeSort>>> =
    Lazy::new(|| Arc::new(Mutex::new(DTypeSort::new())));
