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
use ndarray::Array2;

use crate::{
    dtype::DtypeEnum,
    error::Result,
    graph::{
        self, NodeOp, NodeRef,
        ops::tensor::{TensorBF16Op, TensorUsizeOp},
    },
    nn::Module,
    tensor::{FloatTensor, LongTensor},
};

#[derive(Debug, Clone, Builder)]
pub struct Embedding<'data> {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    pub weight: NodeRef<'data>,

    #[builder(default)]
    pub max_norm: Option<f32>,

    #[builder(default = DtypeEnum::F32)]
    pub norm_type: DtypeEnum,

    #[builder(default)]
    pub scale_grad_by_freq: bool,

    #[builder(default)]
    pub freeze: bool,

    #[builder(default)]
    pub sparse: bool,
}

impl<'data> Module<'data, LongTensor<'data>, FloatTensor<'data>> for Embedding<'data> {
    fn forward(&self, input_ids: LongTensor<'data>) -> Result<FloatTensor<'data>> {
        let tokens = match (&input_ids.0.op, &self.weight.0.op) {
            (
                NodeOp::TensorUsize(TensorUsizeOp { input: ids, .. }),
                NodeOp::TensorBF16(TensorBF16Op { input: weights, .. }),
            ) => Array2::from_shape_fn((ids.len(), self.embedding_dim), |(i, j)| {
                weights[[ids[i], j]]
            }),
            _ => {
                println!("input_ids: {:?}", input_ids.0.op);
                println!("weight: {:?}", self.weight.0.op);
                panic!("Unsupported input_ids and weight types");
            }
        };

        Ok(graph::tensor_bf16(tokens.into_dyn()))
    }

    fn parameters(&self) -> Vec<NodeRef<'data>> {
        vec![self.weight.clone()]
    }
}
