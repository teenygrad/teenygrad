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

use num_traits::{Float, Zero};

pub trait Dtype: 'static + Default + Clone + Float + Zero + std::fmt::Debug {
    type RustType: Send + Sync + Clone + Copy + 'static;
}

impl Dtype for f32 {
    type RustType = f32;
}

impl Dtype for f64 {
    type RustType = f64;
}

/// Converts any type that implements `Into<f32>` (such as f64, f32, etc.) to f32.
pub fn to_f32<T: Into<f32>>(value: T) -> f32 {
    value.into()
}
