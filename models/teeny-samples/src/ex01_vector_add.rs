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

use teeny_core::nn::module::Module;
use teeny_core::tensor1::{self, TensorRef};
use teeny_macros::{JitModule, jit};

#[derive(Debug, JitModule)]
pub struct VectorAdd {
    pub v1: TensorRef<f32>,
    pub v2: TensorRef<f32>,
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

impl Module<f32, TensorRef<f32>> for VectorAdd {
    #[jit]
    fn forward(&self) -> TensorRef<f32> {
        &self.v1 + &self.v2
    }

    fn parameters(&self) -> Vec<TensorRef<f32>> {
        vec![self.v1.clone(), self.v2.clone()]
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _model = VectorAdd::default();

    Ok(())
}
