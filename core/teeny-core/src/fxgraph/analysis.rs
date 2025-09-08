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

    fn make(_egraph: &mut EGraph<FxGraphLang, Self>, _enode: &FxGraphLang) -> Self::Data {
        // The current approach to analysis isn't flexible it has not support
        // for helper structs (such as type theories), there we do not do any analysis
        // and just return Unknown. The type cheecking will be done after the egraph is
        // built.
        GraphAnalysis::Unknown
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        todo!("merge not implemented for {a:?} and {b:?}")
    }
}
