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

use egg::{Analysis, DidMerge, EGraph};

use crate::{
    error::Error,
    fxgraph::{
        lang::FxGraphLang,
        types::{Type, TypeTheory},
    },
};

// Analysis for tracking tensor properties and optimization opportunities
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeAnalysis {
    r#type: Type,
}

impl NodeAnalysis {
    pub fn new(r#type: Type) -> Self {
        Self { r#type }
    }
}

#[derive(Debug)]
pub struct GraphAnalysis {
    pub type_theory: TypeTheory,
}

impl GraphAnalysis {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            type_theory: TypeTheory::new()?,
        })
    }
}

impl Analysis<FxGraphLang> for GraphAnalysis {
    type Data = NodeAnalysis;

    fn make(_egraph: &mut EGraph<FxGraphLang, Self>, _enode: &FxGraphLang) -> Self::Data {
        // Analyse node
        todo!()
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        todo!("merge not implemented for {a:?} and {b:?}")
    }
}
