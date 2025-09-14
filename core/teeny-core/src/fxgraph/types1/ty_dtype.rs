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

use z3::{DatatypeBuilder, ast::Dynamic};

use crate::fxgraph::{dtype::DType, types1::TypeTheory};

pub fn dtype_builder() -> DatatypeBuilder {
    DatatypeBuilder::new("DType")
        .variant("F32", vec![])
        .variant("BF16", vec![])
        .variant("Bool", vec![])
}

pub fn create_dtype_ty(th: &mut TypeTheory, dtype: &DType) -> Dynamic {
    let constructor = match dtype {
        DType::F32 => &th.dtype_sort.variants[0].constructor,
        DType::BF16 => &th.dtype_sort.variants[1].constructor,
        DType::Bool => &th.dtype_sort.variants[2].constructor,
    };

    constructor.apply(&[])
}
