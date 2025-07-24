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

use derive_builder::Builder;

use crate::{
    dtype::Dtype,
    error::{Error, Result},
    graph::NodeRef,
    nn::Module,
    tensor::{FloatTensor, LongTensor},
};

#[derive(Debug, Clone, Builder)]
pub struct Embedding<N: Dtype> {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,

    #[builder(default)]
    pub max_norm: Option<f32>,

    pub norm_type: N,
    pub scale_grad_by_freq: bool,
    pub weight: FloatTensor<N>,
    pub freeze: bool,
    pub sparse: bool,
}

impl<N: Dtype> Embedding<N> {
    pub fn from_pretrained(
        _vocab_size: usize,
        _hidden_size: usize,
        _padding_token_id: Option<usize>,
        _weights: FloatTensor<N>,
    ) -> Self {
        todo!()
    }
}

impl<N: Dtype> Module<N, LongTensor, FloatTensor<N>> for Embedding<N> {
    type Err = Error;

    fn forward(&self, _input_ids: LongTensor) -> Result<FloatTensor<N>> {
        todo!()
    }

    fn parameters(&self) -> Vec<NodeRef<N>> {
        todo!()
    }
}
