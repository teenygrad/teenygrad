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

use ndarray::Array1;
use teeny_core::{
    dtype::Value,
    graph::{NodeRef, arange, inverse, pow, tensor_f32},
};

use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;

pub fn compute_default_rope_parameters<'data>(
    config: &Qwen3Config,
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
        arange(Value::F32(0.0), Value::F32(dim), Value::F32(2.0))
            / tensor_f32(Array1::from(vec![dim]).into_dyn()),
    ));

    (inv_freq, attention_factor)
}
