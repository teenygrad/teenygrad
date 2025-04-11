/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use alloc::vec::Vec;

pub mod memory;

pub enum ElementType {
    FP16,
}

pub trait Tensor<T>: Sized {
    fn element_type(&self) -> &ElementType;
    fn shape(&self) -> &[i64];
    fn data(&self) -> &[T];

    fn reshape(&mut self, shape: Vec<i64>) -> &mut Self;
}
