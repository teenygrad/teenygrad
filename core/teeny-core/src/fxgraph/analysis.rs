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
use z3::ast;

use crate::fxgraph::lang::FxGraphLang;
use crate::fxgraph::placeholder::Placeholder;

// Analysis for tracking tensor properties and optimization opportunities
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphAnalysis {
    Unknown,
    Placeholder { r#type: ast::Dynamic },
}

impl Default for GraphAnalysis {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Analysis<FxGraphLang> for GraphAnalysis {
    type Data = GraphAnalysis;

    fn make(egraph: &mut EGraph<FxGraphLang, Self>, enode: &FxGraphLang) -> Self::Data {
        match enode {
            FxGraphLang::Placeholder(placeholder) => analyse_placeholder(egraph, placeholder),
            _ => unimplemented!("make not implemented for {:?}", enode),
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        todo!("merge not implemented for {a:?} and {b:?}")
    }
}

fn analyse_placeholder(
    _egraph: &mut EGraph<FxGraphLang, GraphAnalysis>,
    placeholder: &Placeholder,
) -> GraphAnalysis {
    todo!("analyse_placeholder not implemented for {:?}", placeholder)
    // let any_type_sort = &ANY_TYPE_SORT.lock().unwrap().0;

    // // Initially we don't know the type of the placeholder, so we use the any type sort
    // // to create a dynamic type, later on we will assert the type of the placeholder
    // // from the example inputs.
    // GraphAnalysis::Placeholder {
    //     r#type: ast::Dynamic::new_const(placeholder.name.clone(), any_type_sort),
    // }
}
