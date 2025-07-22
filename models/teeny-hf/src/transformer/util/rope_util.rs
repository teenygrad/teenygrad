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

use teeny_core::graph::{NodeRef, arange, inverse, pow, tensor};

use crate::transformer::config::model_config::IPretrainedConfig;

pub fn compute_default_rope_parameters(
    config: &impl IPretrainedConfig,
) -> (NodeRef<f32>, NodeRef<f32>) {
    let base = tensor(&[config.rope_theta() as f32]);
    let partial_rotary_factor = config.partial_rotary_factor().unwrap_or(1.0);
    let head_dim = config
        .head_dim()
        .unwrap_or(config.hidden_size() / config.num_attention_heads());
    let dim = head_dim as f32 * partial_rotary_factor;

    let attention_factor = tensor(&[1.0]); // Unused in this type of RoPE
    let inv_freq = inverse(pow(base, arange(0.0, dim, 2.0) / tensor(&[dim])));

    (inv_freq, attention_factor)
}
