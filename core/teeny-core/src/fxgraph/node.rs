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

use egg::{EGraph, Id};

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        torch::add::add_ty,
        types::{Type, TypeInfo},
    },
};

pub fn node_ty(
    egraph: &mut EGraph<FxGraphLang, GraphAnalysis>,
    node: &FxGraphLang,
) -> Result<Type, Error> {
    match node {
        FxGraphLang::Placeholder(p) => p.ty(egraph),
        FxGraphLang::Value(v) => v.ty(egraph),
        FxGraphLang::Add(args) => add_ty(egraph, args),
        _ => todo!("unsupported node: {node:?}"),
    }
}

impl TypeInfo for Id {
    fn ty(&self, egraph: &mut EGraph<FxGraphLang, GraphAnalysis>) -> Result<Type, Error> {
        let node = egraph.id_to_node(*self).clone();

        match node {
            FxGraphLang::Placeholder(p) => p.ty(egraph),
            FxGraphLang::Value(v) => v.ty(egraph),
            _ => todo!("unsupported node: {node:?}"),
        }
    }
}
