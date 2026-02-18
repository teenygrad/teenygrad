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

use ndarray::Array1;
use teeny_core::graph::{NodeRef, arange, inverse, pow, tensor_f32};

use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;

pub fn compute_default_rope_parameters<'data>(
    config: &Qwen3Config,
    _seq_len: Option<usize>,
) -> (NodeRef<'data>, NodeRef<'data>) {
    let base = tensor_f32(Array1::from(vec![config.rope_theta]).into_dyn());
    let partial_rotary_factor = config.partial_rotary_factor.unwrap_or(1.0);
    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);
    let dim = head_dim as f32 * partial_rotary_factor;

    let attention_factor = tensor_f32(Array1::from(vec![1.0]).into_dyn()); // Unused in this type of RoPE
    let inv_freq = inverse(pow(
        base,
        arange::<f32>(0.0, dim, 2.0) / tensor_f32(Array1::from(vec![dim]).into_dyn()),
    ));

    (inv_freq, attention_factor)
}
