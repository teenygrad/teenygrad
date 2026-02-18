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

use ndarray::Array1;
use teeny_core::{
    graph::{NodeRef, tensor_f32},
    nn::Module,
};

use crate::error::Result;

#[derive(Debug)]
pub struct VectorAdd<'data> {
    pub v1: NodeRef<'data>,
    pub v2: NodeRef<'data>,
}

#[allow(clippy::new_without_default)]
impl<'data> VectorAdd<'data> {
    pub fn new() -> Self
where {
        let v1 = tensor_f32(Array1::from(vec![1.0f32, 2.0, 3.0]).mapv(|x| x).into_dyn());
        let v2 = tensor_f32(Array1::from(vec![4.0f32, 5.0, 6.0]).mapv(|x| x).into_dyn());

        Self { v1, v2 }
    }
}

impl<'data> Module<'data, NodeRef<'data>, NodeRef<'data>> for VectorAdd<'data> {
    fn forward(&mut self, _x: NodeRef<'data>) -> Result<NodeRef<'data>> {
        Ok(&self.v1 + &self.v2)
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        vec![]
    }
}

pub async fn run() -> Result<()> {
    let _model = VectorAdd::new();

    // compile the model
    // run the model
    // evaluate the model

    Ok(())
}
