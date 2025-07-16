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

use crate::{
    dtype,
    graph::{Node, NodeOp, ops::sigmoid::SigmoidOp},
    tensor::shape::Shape,
};

pub fn sigmoid<S: Shape, N: dtype::Dtype>(input: &Arc<Node<S, N>>) -> Arc<Node<S, N>> {
    Arc::new(NodeOp::Sigmoid(SigmoidOp::new(input.clone())).into())
}

// impl<T: num::Num> Module1<T, &Tensor, Tensor> for Sigmoid {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         input.sigmoid()
//     }
// }
