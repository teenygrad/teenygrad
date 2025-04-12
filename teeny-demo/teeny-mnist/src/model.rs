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

use teeny_model::{Model, Parameter};
use teeny_tensor::tensor::Tensor;

pub struct BobNet<T> {
    l1: Box<dyn Tensor<T>>,
    l2: Box<dyn Tensor<T>>,
}

impl<T: Tensor<T> + Clone> Model<T> for BobNet<T> {
    fn parameters(&self) -> Vec<Box<dyn Parameter>> {
        vec![]
    }

    fn forward(&self, x: Box<dyn Tensor<T>>) -> Box<dyn Tensor<T>> {
        x.dot(&*self.l1).relu().dot(&*self.l2).log_softmax()
    }
}
