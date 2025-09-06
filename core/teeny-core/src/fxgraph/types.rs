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
    DatatypeAccessor, DatatypeBuilder, DatatypeSort, FuncDecl, Solver, Sort, Symbol,
    ast::{Array, Ast, Bool, Dynamic, Int},
};

use crate::{
    error::Error,
    fxgraph::{dtype::DType, shape::SymInt, tensor::Tensor},
};

#[derive(Debug)]
pub struct TypeTheory {
    solver: Solver,

    pub any_type_sort: Sort,
    pub dtype_sort: Sort,
    pub tensor_sort: Sort,
    pub shape_sort: Sort,
    pub device_sort: Sort,
    pub symint_sort: DatatypeSort,

    // subtype function
    pub subtype_fn: FuncDecl,
    pub tensor_dtype_fn: FuncDecl,
    pub tensor_shape_fn: FuncDecl,
    pub tensor_rank_fn: FuncDecl,
    pub shape_dim_fn: FuncDecl,

    // numeric types
    pub dtype_f32: Dynamic,
    pub dtype_bf16: Dynamic,
    pub dtype_bool: Dynamic,

    // constructors
    pub make_tensor_fn: FuncDecl,
    pub tensor_compatible_fn: FuncDecl,
    pub broadcast_compatible_fn: FuncDecl,

    // cache for device types
    pub devices: HashMap<String, Dynamic>,
}

unsafe impl Send for TypeTheory {}
unsafe impl Sync for TypeTheory {}

impl TypeTheory {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Result<Self, Error> {
        let solver = Solver::new();

        let any_type_sort = Sort::uninterpreted(Symbol::String("Any".to_string()));
        let dtype_sort = Sort::uninterpreted(Symbol::String("DType".to_string()));
        let tensor_sort = Sort::uninterpreted(Symbol::String("Tensor".to_string()));
        let device_sort = Sort::uninterpreted(Symbol::String("Device".to_string()));
        let symint_sort = DatatypeBuilder::new("SymInt")
            .variant("Int", vec![("value", DatatypeAccessor::Sort(Sort::int()))])
            .variant(
                "Sym",
                vec![("value", DatatypeAccessor::Sort(Sort::string()))],
            )
            .finish();
        let shape_sort = Sort::array(&Sort::int(), &symint_sort.sort);

        let dtype_f32 = Dynamic::new_const("F32", &dtype_sort);
        let dtype_bf16 = Dynamic::new_const("BF16", &dtype_sort);
        let dtype_bool = Dynamic::new_const("Bool", &dtype_sort);

        let subtype_fn = FuncDecl::new("subtype", &[&dtype_sort, &dtype_sort], &Sort::bool());
        let tensor_dtype_fn = FuncDecl::new("tensor_dtype", &[&tensor_sort], &dtype_sort);
        let tensor_shape_fn = FuncDecl::new("tensor_shape", &[&tensor_sort], &shape_sort);
        let tensor_rank_fn = FuncDecl::new("tensor_rank", &[&tensor_sort], &Sort::int());
        let shape_dim_fn = FuncDecl::new("shape_dim", &[&shape_sort], &Sort::int());

        let make_tensor_fn = FuncDecl::new(
            "make_tensor",
            &[&dtype_sort, &device_sort, &shape_sort, &Sort::int()],
            &tensor_sort,
        );
        let tensor_compatible_fn = FuncDecl::new(
            "tensor_compatible",
            &[&tensor_sort, &tensor_sort],
            &Sort::bool(),
        );
        let broadcast_compatible_fn = FuncDecl::new(
            "broadcast_compatible",
            &[&tensor_sort, &tensor_sort],
            &Sort::bool(),
        );

        let mut type_theory = Self {
            solver,
            any_type_sort,
            dtype_sort,
            tensor_sort,
            shape_sort,
            device_sort,
            symint_sort,

            // numeric types
            dtype_f32,
            dtype_bf16,
            dtype_bool,

            // subtype function
            subtype_fn,

            // type accessors
            tensor_dtype_fn,
            tensor_shape_fn,
            tensor_rank_fn,
            shape_dim_fn,

            // constructors
            make_tensor_fn,

            // compatibility functions
            tensor_compatible_fn,
            broadcast_compatible_fn,

            // type constants for devices
            devices: HashMap::new(),
        };

        type_theory.setup_subtyping_axioms()?;
        type_theory.setup_tensor_axioms();
        type_theory.setup_shape_axioms();

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
            let device = Dynamic::new_const(device_desc.clone(), &self.device_sort);
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

        Ok(self.make_tensor_fn.apply(&[dtype, device, &shape, &rank]))
    }

    pub fn create_dtype(&self, dtype: &DType) -> &Dynamic {
        match dtype {
            DType::F32 => &self.dtype_f32,
            DType::BF16 => &self.dtype_bf16,
            DType::Bool => &self.dtype_bool,
        }
    }

    pub fn create_shape(&self, dims: &[SymInt]) -> Array {
        let mut shape = Array::fresh_const("shape", &Sort::int(), &self.symint_sort.sort);

        for (i, dim) in dims.iter().cloned().enumerate() {
            let index = Int::from_i64(i as i64);
            let value = match dim {
                SymInt::Int(value) => self.symint_sort.variants[0]
                    .constructor
                    .apply(&[&Int::from_i64(value)]),
                SymInt::Str(value) => self.symint_sort.variants[1]
                    .constructor
                    .apply(&[&z3::ast::String::from(value)]),
            };
            shape = shape.store(&index, &value);
        }

        shape
    }

    fn setup_subtyping_axioms(&mut self) -> Result<(), Error> {
        let t = Dynamic::new_const("t", &self.any_type_sort);
        let t1 = Dynamic::new_const("t1", &self.any_type_sort);
        let t2 = Dynamic::new_const("t2", &self.any_type_sort);
        let t3 = Dynamic::new_const("t3", &self.any_type_sort);

        // Reflexivity: ∀t. subtype(t, t)
        let reflexive = z3::ast::forall_const(
            &[&t],
            &[],
            &self.subtype_fn.apply(&[&t, &t]).try_into().unwrap(),
        );
        self.solver.assert(&reflexive);

        // Transitivity: ∀t1,t2,t3. subtype(t1,t2) ∧ subtype(t2,t3) → subtype(t1,t3)
        let a = self
            .subtype_fn
            .apply(&[&t1, &t2])
            .try_into()
            .map_err(Error::Z3)?;
        let b = self
            .subtype_fn
            .apply(&[&t2, &t3])
            .try_into()
            .map_err(Error::Z3)?;
        let c = self
            .subtype_fn
            .apply(&[&t1, &t3])
            .try_into()
            .map_err(Error::Z3)?;
        let transitive = z3::ast::forall_const(
            &[&t1, &t2, &t3],
            &[],
            &Bool::implies(&Bool::and(&[&a, &b]), &c),
        );
        self.solver.assert(&transitive);

        // add additional subtype axioms here

        Ok(())
    }

    fn setup_tensor_axioms(&mut self) {
        let dtype = Dynamic::new_const("dtype", &self.dtype_sort);
        let shape = Dynamic::new_const("shape", &self.shape_sort);
        let device = Dynamic::new_const("device", &self.device_sort);
        let rank = Int::new_const("rank");

        // Tensor constructor axiom: make_tensor creates valid tensors
        let tensor = self.make_tensor_fn.apply(&[&dtype, &device, &shape, &rank]);

        // dtype(make_tensor(d, s, r)) = d
        let dtype_axiom = z3::ast::forall_const(
            &[&dtype, &shape, &rank],
            &[],
            &self.tensor_dtype_fn.apply(&[&tensor])._eq(&dtype),
        );
        self.solver.assert(&dtype_axiom);

        // shape(make_tensor(d, s, r)) = s
        let shape_axiom = z3::ast::forall_const(
            &[&dtype, &shape, &rank],
            &[],
            &self.tensor_shape_fn.apply(&[&tensor])._eq(&shape),
        );
        self.solver.assert(&shape_axiom);

        // rank(make_tensor(d, s, r)) = r
        let rank_axiom = z3::ast::forall_const(
            &[&dtype, &shape, &rank],
            &[],
            &self.tensor_rank_fn.apply(&[&tensor])._eq(&rank),
        );
        self.solver.assert(&rank_axiom);
    }

    fn setup_shape_axioms(&mut self) {
        let t1 = Dynamic::new_const("tensor1", &self.tensor_sort);
        let t2 = Dynamic::new_const("tensor2", &self.tensor_sort);

        // Tensor compatibility: compatible tensors have compatible dtypes and shapes
        let compatible_def = z3::ast::forall_const(
            &[&t1, &t2],
            &[],
            &Bool::iff(
                &self
                    .tensor_compatible_fn
                    .apply(&[&t1, &t2])
                    .try_into()
                    .unwrap(),
                Bool::and(&[
                    // Compatible dtypes
                    &self
                        .subtype_fn
                        .apply(&[
                            &self.tensor_dtype_fn.apply(&[&t1]),
                            &self.tensor_dtype_fn.apply(&[&t2]),
                        ])
                        .try_into()
                        .unwrap(),
                    // Same rank
                    &self
                        .tensor_rank_fn
                        .apply(&[&t1])
                        ._eq(self.tensor_rank_fn.apply(&[&t2])),
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
                    .unwrap(),
                self.tensor_rank_fn
                    .apply(&[&t1])
                    ._eq(self.tensor_rank_fn.apply(&[&t2])),
            ),
        );
        self.solver.assert(&broadcast_def);
    }
}
