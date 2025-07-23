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

use num_traits::Zero;

pub trait Dtype: 'static + Default + Clone + Copy + Zero + std::fmt::Debug {
    type RustType: Send + Sync + Zero + Clone + Copy + 'static;

    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl Dtype for f32 {
    type RustType = f32;

    fn from_f32(value: f32) -> Self {
        value
    }

    fn to_f32(self) -> f32 {
        self
    }
}

impl Dtype for usize {
    type RustType = usize;

    fn from_f32(value: f32) -> Self {
        value as usize
    }

    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Dtype for f64 {
    type RustType = f64;

    fn from_f32(value: f32) -> Self {
        value as f64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }
}
