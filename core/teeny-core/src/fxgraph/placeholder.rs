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

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::{EGraph, Id};
use z3::ast::{Ast, Dynamic};

use crate::error::Error;
use crate::fxgraph::analysis::GraphAnalysis;
use crate::fxgraph::lang::FxGraphLang;
use crate::fxgraph::types::{Type, TypeInfo, TypeTheory};
use crate::fxgraph::value::Value;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Placeholder {
    pub name: String,
    pub target: Option<String>,
    pub users: Vec<Id>,
    pub example_input: Value,
}

impl FromStr for Placeholder {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for Placeholder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        format!("{:?}", self).fmt(f)
    }
}

impl TypeInfo for Placeholder {
    fn ty(&self, egraph: &mut EGraph<FxGraphLang, GraphAnalysis>) -> Result<Type, Error> {
        let next_id = egraph.analysis.next_id();
        let example_input_ty = self.example_input.ty(egraph)?;
        let ty = Dynamic::new_const(format!("#{}", next_id), &example_input_ty.get_sort());

        let solver = &egraph.analysis.type_theory.solver;
        solver.assert(ty.eq(&example_input_ty));
        Ok(ty)
    }
}
