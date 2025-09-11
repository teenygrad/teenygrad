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

use std::str::FromStr;

use egg::EGraph;
use z3::{
    DatatypeAccessor, DatatypeBuilder, DatatypeSort, Sort,
    ast::{Dynamic, Int},
};

use crate::{
    error::Error,
    fxgraph::{analysis::GraphAnalysis, lang::FxGraphLang, shape::SymInt, types::TypeTheory},
};

pub fn symint_builder() -> DatatypeSort {
    DatatypeBuilder::new("SymInt")
        .variant("Int", vec![("value", DatatypeAccessor::Sort(Sort::int()))])
        .variant(
            "Sym",
            vec![("value", DatatypeAccessor::Sort(Sort::string()))],
        )
        .finish()
}

pub fn create_symint_ty(
    egraph: &mut EGraph<FxGraphLang, GraphAnalysis>,
    symint: &SymInt,
) -> Result<Dynamic, Error> {
    let next_id = egraph.analysis.next_id();
    let th = &egraph.analysis.type_theory;

    let result = match symint {
        SymInt::Int(value) => {
            let constructor = &th.symint_sort.variants[0].constructor;
            let value = Int::from_i64(*value);
            constructor.apply(&[&value])
        }
        SymInt::Sym(value) => {
            let constructor = &th.symint_sort.variants[1].constructor;
            let value = z3::ast::String::from_str(value)
                .map_err(|e| Error::Z3(format!("Failed to convert SymInt to String: {}", e)))?;
            constructor.apply(&[&value])
        }
    };

    let ty = Dynamic::new_const(format!("#{}", next_id), &th.symint_sort.sort);
    th.solver.assert(result.eq(&ty));

    Ok(result)
}
