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

use egg::EGraph;
use z3::{
    DatatypeAccessor, DatatypeBuilder, DatatypeSort, Sort,
    ast::{Dynamic, Int},
};

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        tensor::Tensor,
        types::{
            ty_device::{create_device_ty, device_builder},
            ty_dtype::{create_dtype_ty, dtype_builder},
            ty_shape::{create_shape_ty, shape_builder},
            ty_symint::symint_builder,
            util::datatype_sort,
        },
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

pub fn tensor_builder() -> Result<TyTensor, Error> {
    let dtype = dtype_builder();
    let device = device_builder();
    let symint_sort = symint_builder();
    let shape = shape_builder(&symint_sort.sort);
    let tensor = DatatypeBuilder::new("Tensor").variant(
        "value",
        vec![
            ("dtype", datatype_sort("DType")),
            ("device", datatype_sort("Device")),
            ("shape", datatype_sort("Shape")),
            ("rank", DatatypeAccessor::Sort(Sort::int())),
        ],
    );

    Ok(TyTensor {
        dtype,
        device,
        shape,
        symint_sort,
        tensor,
    })
}

pub fn create_tensor_ty(
    egraph: &mut EGraph<FxGraphLang, GraphAnalysis>,
    tensor: &Tensor,
) -> Result<Dynamic, Error> {
    let next_id = egraph.analysis.next_id();
    let th = &mut egraph.analysis.type_theory;
    let device_ty = create_device_ty(th, &tensor.device);

    let dtype_ty = create_dtype_ty(th, &tensor.dtype);
    let shape_dims = &tensor.shape.shape;
    let shape_ty = create_shape_ty(th, shape_dims);
    let rank_ty = Int::from_i64(shape_dims.len() as i64);

    let tensor_ty = th
        .make_tensor_fn
        .apply(&[&dtype_ty, &device_ty, &shape_ty, &rank_ty]);
    let ty = Dynamic::new_const(format!("#{}", next_id), &th.tensor_sort.sort);
    th.solver.assert(tensor_ty.eq(&ty));

    Ok(ty)
}
