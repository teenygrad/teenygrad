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

use egg::EGraph;

use crate::{
    errors::Error,
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
