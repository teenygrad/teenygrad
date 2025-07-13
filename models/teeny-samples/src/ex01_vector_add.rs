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

use teeny_core::tensor1::{Device, num};

#[derive(Debug)]
pub struct VectorAdd<T: num::Num, D: Device<T>> {
    pub v1: <D as Device<T>>::Tensor,
    pub v2: <D as Device<T>>::Tensor,
}

#[allow(clippy::new_without_default)]
impl<T: num::Num, D: Device<T>> VectorAdd<T, D> {
    pub fn new() -> Self
    where
        T: num::Num + Copy + 'static,
        f32: Into<T>,
    {
        // Create arrays with the correct type by converting f32 to T
        let data: Vec<T> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(|x| x.into())
            .collect();
        let array = ndarray::Array1::from_vec(data).into_dyn();

        Self {
            v1: <D>::from_ndarray(array.clone()),
            v2: <D>::from_ndarray(array),
        }
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // We need to specify concrete types since the compiler can't infer them
    // For now, let's use a placeholder - you'll need to implement a concrete Device
    // let _model: VectorAdd<SomeConcreteDevice, f32> = VectorAdd::default();

    Ok(())
}
