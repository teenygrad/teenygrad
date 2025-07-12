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

use teeny_core::nn::module::ModuleNoInput;
use teeny_core::tensor1::{self, DTensor};

#[derive(Debug)]
pub struct VectorAdd {
    pub v1: DTensor<f32>,
    pub v2: DTensor<f32>,
}

impl VectorAdd {
    pub fn new() -> Self {
        Self {
            v1: tensor1::from_ndarray(
                ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_dyn(),
            ),
            v2: tensor1::from_ndarray(
                ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_dyn(),
            ),
        }
    }
}

impl Default for VectorAdd {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleNoInput<DTensor<f32>> for VectorAdd {
    fn forward(&self) -> DTensor<f32> {
        // self.v1 + self.v2
        unimplemented!()
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model = VectorAdd::default();

    // compile the model for the device
    // send the model to the device
    // run the model
    // retrieve the results
    // check the results

    Ok(())
}
