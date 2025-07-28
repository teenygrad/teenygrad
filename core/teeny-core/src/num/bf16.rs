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

use std::ops::Add;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct bf16(pub half::bf16);
impl num_traits::Zero for bf16 {
    fn zero() -> Self {
        bf16(half::bf16::ZERO)
    }

    fn is_zero(&self) -> bool {
        half::bf16::ZERO.eq(&self.0)
    }
}

impl Add for bf16 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        bf16(self.0 + other.0)
    }
}
