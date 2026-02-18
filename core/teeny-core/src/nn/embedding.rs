/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use derive_builder::Builder;
use ndarray::Array2;

use crate::{
    dtype::DtypeEnum,
    error::Result,
    graph::tensor::{FloatTensor, LongTensor},
    graph::{
        self, NodeOp, NodeRef,
        ops::tensor::{TensorBF16Op, TensorUsizeOp},
    },
    nn::Module,
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
    fn forward(&mut self, input_ids: LongTensor<'data>) -> Result<FloatTensor<'data>> {
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
