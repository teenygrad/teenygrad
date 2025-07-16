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

use std::sync::Arc;

use teeny_core::{
    dtype,
    graph::{self},
    tensor::shape::DynamicShape,
};

#[derive(Debug)]
pub struct VectorAdd<T: dtype::Dtype> {
    pub v1: Arc<graph::Node<DynamicShape, T>>,
    pub v2: Arc<graph::Node<DynamicShape, T>>,
}

#[allow(clippy::new_without_default)]
impl<T: dtype::Dtype> VectorAdd<T> {
    pub fn new() -> Self
    where
        T: dtype::Dtype + Copy + 'static,
        f32: Into<T>,
    {
        let v1 = [1.0, 2.0, 3.0].map(|x| x.into());
        let v2 = [4.0, 5.0, 6.0].map(|x| x.into());

        Self {
            v1: Arc::new(graph::tensor(&v1)),
            v2: Arc::new(graph::tensor(&v2)),
        }
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _model: VectorAdd<f32> = VectorAdd::new();

    // compile the model
    // run the model
    // evaluate the model

    Ok(())
}
