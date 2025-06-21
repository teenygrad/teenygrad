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
/// A type to represent u8 data type
pub struct U8;

/// A type to represent usize data type
pub struct USize;

/// A type to represent int data type
pub struct Int;

/// A type to represent f32 data type
pub struct F32;

/// Trait for numeric types that can be used in tensors
pub trait NumericType {
    type RustType;
}

impl NumericType for U8 {
    type RustType = u8;
}

impl NumericType for F32 {
    type RustType = f32;
}

impl NumericType for USize {
    type RustType = usize;
}

impl NumericType for Int {
    type RustType = isize;
}

impl NumericType for usize {
    type RustType = usize;
}

impl NumericType for f32 {
    type RustType = f32;
}
