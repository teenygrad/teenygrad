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
