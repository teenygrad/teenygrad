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

use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use z3::{Sort, Symbol};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeSort {
    pub sort: Sort,
}

unsafe impl Send for TypeSort {}
unsafe impl Sync for TypeSort {}

impl TypeSort {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            sort: Sort::uninterpreted(Symbol::String("Type".to_string())),
        }
    }
}

pub static TYPE_SORT: Lazy<Arc<Mutex<TypeSort>>> =
    Lazy::new(|| Arc::new(Mutex::new(TypeSort::new())));
