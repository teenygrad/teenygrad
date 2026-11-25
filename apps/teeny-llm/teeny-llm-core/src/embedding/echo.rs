/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! [`EchoEmbeddingModel`], the reference [`EmbeddingModel`].

use futures::future::BoxFuture;

use super::{
    EmbeddingCapabilities, EmbeddingInput, EmbeddingModel, EmbeddingRequest, Embeddings, Pooling,
    truncate_normalized,
};
use crate::{Error, ModelInfo, Result};

/// An embedding model without weights: each word (or token id) is hashed into one of
/// [`EchoEmbeddingModel::DIMENSIONS`] buckets, and the counts normalised.
///
/// Inputs that share words therefore have similar vectors, which is enough to exercise
/// frontends and search code. It accepts token ids, reduces dimensions, and counts a token
/// per word.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoEmbeddingModel;

impl EchoEmbeddingModel {
    pub const MODEL_ID: &'static str = "teeny-echo-embed";
    pub const DIMENSIONS: u32 = 16;
    const MAX_INPUT_TOKENS: u32 = 512;
}

/// FNV-1a: stable across runs and platforms, unlike the standard library's hasher.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

impl EmbeddingModel for EchoEmbeddingModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities {
            max_input_tokens: Self::MAX_INPUT_TOKENS,
            max_batch: 64,
            dimensions: Self::DIMENSIONS,
            reducible_dimensions: true,
            token_input: true,
            pooling: Pooling::Mean,
            normalized: true,
        }
    }

    fn embed(&self, request: EmbeddingRequest) -> BoxFuture<'_, Result<Embeddings>> {
        Box::pin(async move {
            let dims = request.dimensions.unwrap_or(Self::DIMENSIONS) as usize;
            let mut vectors = Vec::with_capacity(request.inputs.len());
            let mut prompt_tokens = 0u32;
            for (i, input) in request.inputs.iter().enumerate() {
                let keys: Vec<u64> = match input {
                    EmbeddingInput::Text(t) => {
                        t.split_whitespace().map(|w| fnv1a(w.as_bytes())).collect()
                    }
                    EmbeddingInput::Tokens(ids) => {
                        ids.iter().map(|id| fnv1a(&id.to_le_bytes())).collect()
                    }
                };
                if keys.len() > Self::MAX_INPUT_TOKENS as usize {
                    return Err(Error::InvalidParameter {
                        parameter: format!("input[{i}]"),
                        detail: format!(
                            "{} tokens, more than the model's {}",
                            keys.len(),
                            Self::MAX_INPUT_TOKENS
                        ),
                    });
                }
                let mut v = vec![0.0f32; Self::DIMENSIONS as usize];
                for k in &keys {
                    v[(k % u64::from(Self::DIMENSIONS)) as usize] += 1.0;
                }
                prompt_tokens =
                    prompt_tokens.saturating_add(u32::try_from(keys.len()).unwrap_or(u32::MAX));
                vectors.push(truncate_normalized(v, dims));
            }
            Ok(Embeddings {
                vectors,
                prompt_tokens,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed(inputs: Vec<EmbeddingInput>, dimensions: Option<u32>) -> Result<Embeddings> {
        let model: &dyn EmbeddingModel = &EchoEmbeddingModel;
        futures::executor::block_on(model.embed_checked(EmbeddingRequest {
            model: EchoEmbeddingModel::MODEL_ID.into(),
            inputs,
            dimensions,
            user: None,
        }))
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn test_echo_embeddings_are_unit_vectors_that_reflect_shared_words() {
        let text = |s: &str| EmbeddingInput::Text(s.into());
        let e = embed(
            vec![text("red apple pie"), text("red apple"), text("quantum")],
            None,
        )
        .unwrap();
        assert_eq!(e.vectors.len(), 3);
        assert_eq!(e.prompt_tokens, 6);
        for v in &e.vectors {
            assert_eq!(v.len(), EchoEmbeddingModel::DIMENSIONS as usize);
            assert!((cosine(v, v) - 1.0).abs() < 1e-5);
        }
        assert!(cosine(&e.vectors[0], &e.vectors[1]) > cosine(&e.vectors[0], &e.vectors[2]));
        // Stable across runs.
        assert_eq!(
            embed(vec![text("red apple")], None).unwrap().vectors[0],
            e.vectors[1]
        );
    }

    #[test]
    fn test_echo_embeddings_take_tokens_and_reduce_dimensions() {
        let e = embed(vec![EmbeddingInput::Tokens(vec![1, 2, 3])], Some(4)).unwrap();
        assert_eq!(e.vectors[0].len(), 4);
        assert_eq!(e.prompt_tokens, 3);
        let long = EmbeddingInput::Text("w ".repeat(600));
        assert!(matches!(
            embed(vec![long], None),
            Err(Error::InvalidParameter { .. })
        ));
    }
}
