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

use crate::num::bf16::bf16;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Usize(usize),
    F32(f32),
    Bf16(bf16),
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Value::Usize(value)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::F32(value)
    }
}

impl From<bf16> for Value {
    fn from(value: bf16) -> Self {
        Value::Bf16(value)
    }
}
