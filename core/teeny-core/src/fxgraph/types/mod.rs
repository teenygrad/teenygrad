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

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        types::{ty_kwargs::TyKwArgs, ty_tensor::TyTensor},
    },
};

pub mod ty_kwargs;
pub mod ty_tensor;

pub trait TypeInfo {
    fn ty(&self, egraph: &mut EGraph<FxGraphLang, GraphAnalysis>) -> Result<Type, Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    F32,
    BF16,
    Bool,
    Tensor(TyTensor),
    SymInt,
    List(Vec<Type>),
    Tuple(Vec<Type>),
    KwArgs(TyKwArgs),
}
