/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use egg::{EGraph, Id};

use crate::{
    errors::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        torch::{add::add_ty, output::output_ty},
        types::{Type, TypeInfo, ty_kwargs::TyKwArgs},
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
        FxGraphLang::KwArgs(args) => Ok(Type::KwArgs(TyKwArgs::new(egraph, args)?)),
        FxGraphLang::Output(args) => output_ty(egraph, args),
        _ => todo!("unsupported node: {node:?}"),
    }
}

impl TypeInfo for Id {
    fn ty(&self, egraph: &mut EGraph<FxGraphLang, GraphAnalysis>) -> Result<Type, Error> {
        let node = egraph.id_to_node(*self).clone();

        match node {
            FxGraphLang::Placeholder(p) => p.ty(egraph),
            FxGraphLang::Value(v) => v.ty(egraph),
            FxGraphLang::Add(args) => add_ty(egraph, &args),
            FxGraphLang::KwArgs(args) => Ok(Type::KwArgs(TyKwArgs::new(egraph, &args)?)),
            FxGraphLang::Output(args) => output_ty(egraph, &args),
            _ => todo!("unsupported node: {node:?}"),
        }
    }
}
