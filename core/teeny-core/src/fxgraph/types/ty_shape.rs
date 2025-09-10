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
    DatatypeAccessor, DatatypeBuilder, Sort,
    ast::{Array, Dynamic, Int},
};

use crate::fxgraph::{shape::SymInt, types::TypeTheory};

pub fn shape_builder(symint_sort: &Sort) -> DatatypeBuilder {
    DatatypeBuilder::new("Shape").variant(
        "value",
        vec![(
            "value",
            DatatypeAccessor::Sort(Sort::array(&Sort::int(), symint_sort)),
        )],
    )
}

pub fn create_shape_ty(th: &mut TypeTheory, dims: &[SymInt]) -> Dynamic {
    let shape = Array::fresh_const("shape", &Sort::int(), &th.symint_sort.sort);

    for (i, dim) in dims.iter().cloned().enumerate() {
        let index = Int::from_i64(i as i64);
        let value = match dim {
            SymInt::Int(value) => th.symint_sort.variants[0]
                .constructor
                .apply(&[&Int::from_i64(value)]),
            SymInt::Sym(value) => th.symint_sort.variants[1]
                .constructor
                .apply(&[&z3::ast::String::from(value)]),
        };
        shape.store(&index, &value);
    }

    let constructor = &th.shape_sort.variants[0].constructor;
    constructor.apply(&[&shape])
}
