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

/// Trait for numeric types that can be used in tensors
pub trait Num: 'static + std::fmt::Debug {
    type RustType: Send + Sync + Clone + Copy + 'static;
}

impl Num for f32 {
    type RustType = f32;
}

// impl Num for usize {
//     type RustType = usize;
// }

// impl Num for half::f16 {
//     type RustType = half::f16;
// }
