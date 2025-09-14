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

use z3::{
    Sort,
    ast::{Array, Int},
};

use crate::fxgraph::shape::SymInt;

pub fn shape_sort() -> Sort {
    Sort::array(&Sort::int(), &Sort::int())
}

pub fn create_shape_ty(dims: &[SymInt]) -> Array {
    let shape = Array::fresh_const("shape", &Sort::int(), &Sort::int());

    for (i, dim) in dims.iter().cloned().enumerate() {
        let index = Int::from_i64(i as i64);
        let value = match dim {
            SymInt::Int(value) => Int::from_i64(value),
            SymInt::Sym(value) => Int::new_const(value),
        };
        shape.store(&index, &value);
    }

    shape
}
