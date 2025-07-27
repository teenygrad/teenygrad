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
    dtype,
    graph::{NodeRef, tensor},
    nn::Module,
};

use crate::error::Result;

#[derive(Debug)]
pub struct VectorAdd<N: dtype::Dtype> {
    pub v1: NodeRef<N>,
    pub v2: NodeRef<N>,
}

#[allow(clippy::new_without_default)]
impl<N: dtype::Dtype> VectorAdd<N> {
    pub fn new() -> Self
    where
        N: dtype::Dtype + Copy + 'static,
        f32: Into<N>,
    {
        let v1 = tensor(
            Array1::from(vec![1.0f32, 2.0, 3.0])
                .mapv(|x| N::from_f32(x))
                .into_dyn(),
        );
        let v2 = tensor(
            Array1::from(vec![4.0f32, 5.0, 6.0])
                .mapv(|x| N::from_f32(x))
                .into_dyn(),
        );

        Self { v1, v2 }
    }
}

impl Module<f32, NodeRef<f32>, NodeRef<f32>> for VectorAdd<f32> {
    fn forward(&self, _x: NodeRef<f32>) -> Result<NodeRef<f32>> {
        Ok(&self.v1 + &self.v2)
    }

    fn parameters(&self) -> Vec<NodeRef<f32>> {
        vec![]
    }
}

pub async fn run() -> Result<()> {
    let _model: VectorAdd<f32> = VectorAdd::new();

    // compile the model
    // run the model
    // evaluate the model

    Ok(())
}
