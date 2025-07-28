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
    dtype::{Dtype, DtypeEnum},
    error::Result,
    graph::{self, NodeOp, NodeRef, ops::tensor::TensorOp},
    nn::Module,
    tensor::{FloatTensor, LongTensor},
};

#[derive(Debug, Clone, Builder)]
pub struct Embedding<N: Dtype> {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    pub weight: ndarray::Array<N, ndarray::IxDyn>,

    #[builder(default)]
    pub max_norm: Option<N>,

    #[builder(default = N::DTYPE)]
    pub norm_type: DtypeEnum,

    #[builder(default)]
    pub scale_grad_by_freq: bool,

    #[builder(default)]
    pub freeze: bool,

    #[builder(default)]
    pub sparse: bool,
}

impl<'data, N: Dtype> Module<'data, N, LongTensor<'data>, FloatTensor<'data, N>> for Embedding<N> {
    fn forward(&self, input_ids: LongTensor<'data>) -> Result<FloatTensor<'data, N>> {
        let input_ids = input_ids.0;

        let tokens = match &input_ids.op {
            NodeOp::Tensor(TensorOp { input, .. }) => input.map(|x| self.weight[*x]),
            _ => unreachable!(),
        };

        println!("tokens: {tokens:?}");
        Ok(graph::tensor(tokens))
    }

    fn parameters(&self) -> Vec<NodeRef<'data, N>> {
        todo!()
    }
}
