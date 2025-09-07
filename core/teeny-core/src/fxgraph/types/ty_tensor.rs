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

use z3::{DatatypeAccessor, DatatypeBuilder, DatatypeSort, Sort};

use crate::{
    error::Error,
    fxgraph::types::{
        ty_device::create_device, ty_dtype::create_dtype, ty_shape::create_shape,
        ty_symint::create_symint, util::datatype_sort,
    },
};

#[derive(Debug)]
pub struct TyTensor {
    pub dtype: DatatypeBuilder,
    pub device: DatatypeBuilder,
    pub shape: DatatypeBuilder,
    pub symint_sort: DatatypeSort,
    pub tensor: DatatypeBuilder,
}

impl TyTensor {
    pub fn new() -> Result<Self, Error> {
        let dtype = create_dtype();
        let device = create_device();
        let symint_sort = create_symint();
        let shape = create_shape(&symint_sort.sort);
        let tensor = DatatypeBuilder::new("Tensor").variant(
            "value",
            vec![
                ("dtype", datatype_sort("DType")),
                ("device", datatype_sort("Device")),
                ("shape", datatype_sort("Shape")),
                ("rank", DatatypeAccessor::Sort(Sort::int())),
            ],
        );

        Ok(Self {
            dtype,
            device,
            shape,
            symint_sort,
            tensor,
        })
    }
}
