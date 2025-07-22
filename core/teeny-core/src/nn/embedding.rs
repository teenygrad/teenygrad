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

use crate::{
    dtype::Dtype,
    graph::{NodeRef, zeros},
    tensor::shape::DynamicShape,
};

pub struct Embedding<N: Dtype> {
    pub weight: NodeRef<N>,
}

impl<N: Dtype> Embedding<N> {
    pub fn new(vocab_size: usize, hidden_size: usize, _padding_token_id: Option<usize>) -> Self {
        let embedding_shape = DynamicShape::new(&[hidden_size, vocab_size]);

        Self {
            weight: zeros(embedding_shape),
        }
    }
}
