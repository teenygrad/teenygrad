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
    fn forward(&self, _x: NodeRef<'data>) -> Result<NodeRef<'data>> {
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
