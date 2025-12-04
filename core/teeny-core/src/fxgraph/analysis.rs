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
    fxgraph::{lang::FxGraphLang, node::node_ty, types::Type},
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
pub struct GraphAnalysis {}

impl GraphAnalysis {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {})
    }
}

impl Analysis<FxGraphLang> for GraphAnalysis {
    type Data = NodeAnalysis;

    fn make(
        egraph: &mut EGraph<FxGraphLang, Self>,
        enode: &FxGraphLang,
        _id: egg::Id,
    ) -> Self::Data {
        let node_ty = node_ty(egraph, enode);
        if let Err(e) = node_ty {
            // aarghh - egg doesn't support fallible analysis
            panic!("Error creating type for node: {enode:?}: {e:?}");
        }

        // let solver = &egraph.analysis.type_theory.solver;

        // println!("type check: {:?}", solver.check());
        // println!("type check: {:?}", solver.get_model());

        println!("node: {:?}, node_ty: {:?}", enode, node_ty);
        NodeAnalysis::new(node_ty.unwrap())
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        todo!("merge not implemented for {a:?} and {b:?}")
    }
}
