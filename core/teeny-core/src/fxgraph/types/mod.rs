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

use std::collections::HashMap;

use z3::{
    DatatypeBuilder, DatatypeSort, FuncDecl, Solver, Sort,
    ast::{Array, Bool, Dynamic, Int},
    datatype_builder::create_datatypes,
};

use crate::{
    error::Error,
    fxgraph::{
        device::Device,
        dtype::DType,
        shape::SymInt,
        tensor::Tensor,
        types::{ty_tensor::TyTensor, util::datatype_sort},
    },
};

mod ty_device;
mod ty_dtype;
mod ty_shape;
mod ty_symint;
mod ty_tensor;
mod util;

pub type Type = z3::ast::Dynamic;

#[derive(Debug)]
pub struct TypeTheory {
    pub solver: Solver,
    pub anytype_sort: DatatypeSort,
    pub dtype_sort: DatatypeSort,
    pub device_sort: DatatypeSort,
    pub shape_sort: DatatypeSort,
    pub symint_sort: DatatypeSort,
    pub tensor_sort: DatatypeSort,
    pub subtype_any_fn: FuncDecl,
    pub subtype_dtype_fn: FuncDecl,
    pub tensor_dtype_fn: FuncDecl,
    pub tensor_shape_fn: FuncDecl,
    pub tensor_rank_fn: FuncDecl,
    pub shape_dim_fn: FuncDecl,
    pub make_tensor_fn: FuncDecl,
    pub tensor_compatible_fn: FuncDecl,
    pub broadcast_compatible_fn: FuncDecl,
    pub devices: HashMap<String, Dynamic>,
}

impl TypeTheory {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Result<Self, Error> {
        let solver = Solver::new();
        let TyTensor {
            dtype,
            device,
            shape,
            symint_sort,
            tensor,
        } = TyTensor::new()?;

        let any_type = DatatypeBuilder::new("Any")
            .variant("Any", vec![])
            .variant("Int64", vec![])
            .variant("Float32", vec![])
            .variant("Bool", vec![])
            .variant("Tensor", vec![("value", datatype_sort("Tensor"))]);

        let datatypes = create_datatypes(vec![dtype, device, shape, tensor, any_type]);
        let [
            dtype_sort,
            device_sort,
            shape_sort,
            tensor_sort,
            anytype_sort,
        ] = <[_; 5]>::try_from(datatypes).map_err(|_| Error::Z3("Expected 5 datatypes".into()))?;

        let subtype_any_fn = FuncDecl::new(
            "subtype_any",
            &[&anytype_sort.sort, &anytype_sort.sort],
            &Sort::bool(),
        );

        let subtype_dtype_fn = FuncDecl::new(
            "subtype_dtype",
            &[&dtype_sort.sort, &dtype_sort.sort],
            &Sort::bool(),
        );

        let tensor_dtype_fn = FuncDecl::new("tensor_dtype", &[&tensor_sort.sort], &dtype_sort.sort);
        let tensor_shape_fn = FuncDecl::new("tensor_shape", &[&tensor_sort.sort], &shape_sort.sort);
        let tensor_rank_fn = FuncDecl::new("tensor_rank", &[&tensor_sort.sort], &Sort::int());
        let shape_dim_fn = FuncDecl::new("shape_dim", &[&shape_sort.sort], &Sort::int());

        let make_tensor_fn = FuncDecl::new(
            "make_tensor",
            &[
                &dtype_sort.sort,
                &device_sort.sort,
                &shape_sort.sort,
                &Sort::int(),
            ],
            &tensor_sort.sort,
        );

        let tensor_compatible_fn = FuncDecl::new(
            "tensor_compatible",
            &[&tensor_sort.sort, &tensor_sort.sort],
            &Sort::bool(),
        );

        let broadcast_compatible_fn = FuncDecl::new(
            "broadcast_compatible",
            &[&tensor_sort.sort, &tensor_sort.sort],
            &Sort::bool(),
        );

        let mut type_theory = Self {
            solver,
            anytype_sort,
            dtype_sort,
            device_sort,
            shape_sort,
            symint_sort,
            tensor_sort,
            subtype_any_fn,
            subtype_dtype_fn,
            tensor_dtype_fn,
            tensor_shape_fn,
            tensor_rank_fn,
            shape_dim_fn,
            make_tensor_fn,
            tensor_compatible_fn,
            broadcast_compatible_fn,
            devices: HashMap::new(),
        };

        type_theory.setup_subtyping_axioms()?;
        type_theory.setup_tensor_axioms();
        type_theory.setup_shape_axioms()?;

        Ok(type_theory)
    }

    pub fn create_tensor_type(
        &mut self,
        tensor: &Tensor, // dtype: &Dynamic,
                         // device: &Dynamic,
                         // shape_dims: &[SymInt],
    ) -> Result<Dynamic, Error> {
        let device_desc = format!("{}", tensor.device);
        let device = self.devices.get(&device_desc);
        if device.is_none() {
            let device = self.create_device(&tensor.device);
            self.devices.insert(device_desc.clone(), device);
        }

        let device = self
            .devices
            .get(&device_desc)
            .ok_or(Error::DeviceTypeNotFound(device_desc))?;
        let dtype = self.create_dtype(&tensor.dtype);
        let shape_dims = &tensor.shape.shape;
        let rank = Int::from_i64(shape_dims.len() as i64);
        let shape = self.create_shape(shape_dims);

        Ok(self.make_tensor_fn.apply(&[&dtype, device, &shape, &rank]))
    }

    pub fn create_dtype(&self, dtype: &DType) -> Dynamic {
        let constructor = match dtype {
            DType::F32 => &self.dtype_sort.variants[0].constructor,
            DType::BF16 => &self.dtype_sort.variants[1].constructor,
            DType::Bool => &self.dtype_sort.variants[2].constructor,
        };

        constructor.apply(&[])
    }

    pub fn create_device(&self, device: &Device) -> Dynamic {
        let (constructor, value) = match device {
            Device::Cpu(value) => (&self.device_sort.variants[0].constructor, value),
            Device::Cuda(value) => (&self.device_sort.variants[1].constructor, value),
        };

        let value = z3::ast::String::new_const(value.clone());
        constructor.apply(&[&value])
    }

    pub fn create_shape(&self, dims: &[SymInt]) -> Array {
        let mut shape = Array::fresh_const("shape", &Sort::int(), &self.symint_sort.sort);

        for (i, dim) in dims.iter().cloned().enumerate() {
            let index = Int::from_i64(i as i64);
            let value = match dim {
                SymInt::Int(value) => self.symint_sort.variants[0]
                    .constructor
                    .apply(&[&Int::from_i64(value)]),
                SymInt::Sym(value) => self.symint_sort.variants[1]
                    .constructor
                    .apply(&[&z3::ast::String::from(value)]),
            };
            shape = shape.store(&index, &value);
        }

        shape
    }

    fn setup_subtyping_axioms(&mut self) -> Result<(), Error> {
        let t = Dynamic::new_const("t", &self.anytype_sort.sort);
        let t1 = Dynamic::new_const("t1", &self.anytype_sort.sort);
        let t2 = Dynamic::new_const("t2", &self.anytype_sort.sort);
        let t3 = Dynamic::new_const("t3", &self.anytype_sort.sort);

        // Reflexivity: ∀t. subtype(t, t)
        let reflexivity = z3::ast::forall_const(
            &[&t],
            &[],
            &self
                .subtype_any_fn
                .apply(&[&t, &t])
                .try_into()
                .map_err(Error::Z3)?,
        );

        // Transitivity: ∀t1,t2,t3. subtype(t1,t2) ∧ subtype(t2,t3) → subtype(t1,t3)
        let a: Bool = self
            .subtype_any_fn
            .apply(&[&t1, &t2])
            .try_into()
            .map_err(Error::Z3)?;

        let b: Bool = self
            .subtype_any_fn
            .apply(&[&t2, &t3])
            .try_into()
            .map_err(Error::Z3)?;

        let c: Bool = self
            .subtype_any_fn
            .apply(&[&t1, &t3])
            .try_into()
            .map_err(Error::Z3)?;

        println!("Adding transitivity");
        let transitivity = z3::ast::forall_const(
            &[&t1, &t2, &t3],
            &[],
            &Bool::implies(&Bool::and(&[&a, &b]), &c),
        );

        self.solver.assert(&reflexivity);
        self.solver.assert(&transitivity);

        Ok(())
    }

    fn setup_tensor_axioms(&mut self) {
        let dtype = Dynamic::new_const("dtype", &self.dtype_sort.sort);
        let shape = Dynamic::new_const("shape", &self.shape_sort.sort);
        let device = Dynamic::new_const("device", &self.device_sort.sort);
        let rank = Int::new_const("rank");

        // Tensor constructor axiom: make_tensor creates valid tensors
        let tensor = self.make_tensor_fn.apply(&[&dtype, &device, &shape, &rank]);

        // dtype(make_tensor(d, s, r)) = d
        let dtype_axiom = z3::ast::forall_const(
            &[&dtype, &shape, &rank],
            &[],
            &self.tensor_dtype_fn.apply(&[&tensor]).eq(&dtype),
        );
        self.solver.assert(&dtype_axiom);

        // shape(make_tensor(d, s, r)) = s
        let shape_axiom = z3::ast::forall_const(
            &[&dtype, &shape, &rank],
            &[],
            &self.tensor_shape_fn.apply(&[&tensor]).eq(&shape),
        );
        self.solver.assert(&shape_axiom);

        // rank(make_tensor(d, s, r)) = r
        let rank_axiom = z3::ast::forall_const(
            &[&dtype, &shape, &rank],
            &[],
            &self.tensor_rank_fn.apply(&[&tensor]).eq(&rank),
        );
        self.solver.assert(&rank_axiom);
    }

    fn setup_shape_axioms(&mut self) -> Result<(), Error> {
        let t1 = Dynamic::new_const("tensor1", &self.tensor_sort.sort);
        let t2 = Dynamic::new_const("tensor2", &self.tensor_sort.sort);

        // Tensor compatibility: compatible tensors have compatible dtypes and shapes
        let compatible_def = z3::ast::forall_const(
            &[&t1, &t2],
            &[],
            &Bool::iff(
                &self
                    .tensor_compatible_fn
                    .apply(&[&t1, &t2])
                    .try_into()
                    .map_err(Error::Z3)?,
                Bool::and(&[
                    // Compatible dtypes
                    &self
                        .subtype_dtype_fn
                        .apply(&[
                            &self.tensor_dtype_fn.apply(&[&t1]),
                            &self.tensor_dtype_fn.apply(&[&t2]),
                        ])
                        .try_into()
                        .map_err(Error::Z3)?,
                    // Same rank
                    &self
                        .tensor_rank_fn
                        .apply(&[&t1])
                        .eq(self.tensor_rank_fn.apply(&[&t2])),
                ]),
            ),
        );
        self.solver.assert(&compatible_def);

        // Broadcasting compatibility (simplified - same rank for now)
        let broadcast_def = z3::ast::forall_const(
            &[&t1, &t2],
            &[],
            &Bool::iff(
                &self
                    .broadcast_compatible_fn
                    .apply(&[&t1, &t2])
                    .try_into()
                    .map_err(Error::Z3)?,
                self.tensor_rank_fn
                    .apply(&[&t1])
                    .eq(self.tensor_rank_fn.apply(&[&t2])),
            ),
        );
        self.solver.assert(&broadcast_def);

        Ok(())
    }
}
