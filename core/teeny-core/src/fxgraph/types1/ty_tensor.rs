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
    DatatypeAccessor, DatatypeBuilder, Sort,
    ast::{Dynamic, Int},
};

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        tensor::Tensor,
        types1::{
            ty_device::{create_device_ty, device_builder},
            ty_dtype::{create_dtype_ty, dtype_builder},
            ty_shape::{create_shape_ty, shape_sort},
            util::datatype_sort,
        },
    },
};

#[derive(Debug)]
pub struct TyTensor {
    pub dtype: DatatypeBuilder,
    pub device: DatatypeBuilder,
    pub tensor: DatatypeBuilder,
}

pub fn tensor_builder() -> Result<TyTensor, Error> {
    let dtype = dtype_builder();
    let device = device_builder();
    let shape = shape_sort();
    let tensor = DatatypeBuilder::new("Tensor").variant(
        "value",
        vec![
            ("dtype", datatype_sort("DType")),
            ("device", datatype_sort("Device")),
            ("shape", DatatypeAccessor::Sort(shape)),
            ("rank", DatatypeAccessor::Sort(Sort::int())),
        ],
    );

    Ok(TyTensor {
        dtype,
        device,
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
    let shape_ty = create_shape_ty(shape_dims);
    let rank_ty = Int::from_i64(shape_dims.len() as i64);

    let tensor_ty = th
        .make_tensor_fn
        .apply(&[&dtype_ty, &device_ty, &shape_ty, &rank_ty]);
    let ty = Dynamic::new_const(format!("#{}", next_id), &th.tensor_sort.sort);

    th.solver.assert(tensor_ty.eq(&ty));
    th.solver
        .assert(th.tensor_dtype_fn.apply(&[&ty]).eq(&dtype_ty));
    th.solver
        .assert(th.tensor_shape_fn.apply(&[&ty]).eq(&shape_ty));
    th.solver
        .assert(th.tensor_rank_fn.apply(&[&ty]).eq(&rank_ty));

    Ok(ty)
}

#[cfg(test)]
mod tests {
    use z3::SatResult;

    use crate::fxgraph::{
        device::Device,
        dtype::DType,
        shape::{Shape, SymInt},
    };

    use super::*;

    #[test]
    fn test_create_tensor_ty() {
        let mut egraph = EGraph::new(GraphAnalysis::new().unwrap());

        let tensor = Tensor::new(
            DType::F32,
            Shape::new(&[SymInt::Int(1), SymInt::Int(2), SymInt::Sym("x".to_string())]),
            Device::Cpu("cpu:0".to_string()),
            vec![1, 2, 3],
            false,
        );

        let tensor = create_tensor_ty(&mut egraph, &tensor).unwrap();
        let th = &mut egraph.analysis.type_theory;

        let dtype_ty = create_dtype_ty(th, &DType::F32);
        let dtype_axiom = th.tensor_dtype_fn.apply(&[&tensor]).eq(&dtype_ty);

        let shape_ty =
            create_shape_ty(&[SymInt::Int(1), SymInt::Int(2), SymInt::Sym("x".to_string())]);
        let shape_axiom = th.tensor_shape_fn.apply(&[&tensor]).eq(&shape_ty);

        let device_ty = create_device_ty(th, &Device::Cpu("cpu:0".to_string()));
        let device_axiom = th.tensor_device_fn.apply(&[&tensor]).eq(&device_ty);

        let rank_ty = Int::from_i64(3);
        let rank_axiom = th.tensor_rank_fn.apply(&[&tensor]).eq(&rank_ty);

        th.solver.assert(dtype_axiom);
        th.solver.assert(shape_axiom);
        th.solver.assert(device_axiom);
        th.solver.assert(rank_axiom);

        let result = th.solver.check();
        if result == SatResult::Unknown {
            println!(
                "Solver returned Unknown. Reason: {:?}",
                th.solver.get_reason_unknown()
            );
        }
        assert_eq!(result, SatResult::Sat);
        println!("Tensor type axiom: {:?}", th.solver.get_model());
    }
}
