/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use super::Loss;

pub struct SparseCategoricalCrossEntropy {}

impl SparseCategoricalCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SparseCategoricalCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + 'static> Loss<T> for SparseCategoricalCrossEntropy {
    fn backward(&self) {
        todo!()
    }

    fn compute(
        &self,
        _out: &dyn teeny_tensor::tensor::Tensor<T>,
        _y: &dyn teeny_tensor::tensor::Tensor<T>,
    ) -> T {
        todo!()
    }
}
