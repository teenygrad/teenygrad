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

use teeny_core::dtype;
use teeny_runtime::tensor::Tensor;

#[derive(Debug)]
pub struct VectorAdd<T: dtype::Dtype> {
    pub v1: Tensor<T>,
    pub v2: Tensor<T>,
}

#[allow(clippy::new_without_default)]
impl<T: dtype::Dtype> VectorAdd<T> {
    pub fn new() -> Self
    where
        T: dtype::Dtype + Copy + 'static,
        f32: Into<T>,
    {
        // Create arrays with the correct type by converting f32 to T
        let data: Vec<T> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(|x| x.into())
            .collect();
        let _array = ndarray::Array1::from_vec(data).into_dyn();

        todo!()
        // Self {
        //     v1: <D>::from_ndarray(array.clone()),
        //     v2: <D>::from_ndarray(array),
        // }
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _model: VectorAdd<f32> = VectorAdd::new();

    Ok(())
}
