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

//! Checks `Inkling2Config` against the real `config.json` of
//! `thinkingmachines/Inkling-2-Mini`, committed as a fixture.

use std::path::Path;

use teeny_llm_core::models::inkling2::Inkling2Config;

fn fixture() -> Inkling2Config {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2-mini-config.json");
    Inkling2Config::load(&p).expect("the shipped config should load")
}

#[test]
fn test_reads_the_shipped_config() {
    let t = fixture().text_config;
    assert_eq!(t.hidden_size, 2304);
    assert_eq!(t.num_hidden_layers, 24);
    assert_eq!(t.num_attention_heads, 24);
    assert_eq!(t.num_key_value_heads, 8);
    assert_eq!(t.head_dim, 128);
    assert_eq!(t.vocab_size, 201024);
    assert_eq!(t.unpadded_vocab_size, 200058);
}

#[test]
fn test_reads_the_mixture_of_experts() {
    let t = fixture().text_config;
    assert_eq!(t.n_routed_experts, 128);
    assert_eq!(t.num_experts_per_tok, 8);
    // Inkling-1 had two always-on experts; Mini has none.
    assert_eq!(t.n_shared_experts, 0);
    assert_eq!(t.intermediate_size, 1152);
}

#[test]
fn test_the_core_loops_three_times() {
    let t = fixture().text_config;
    let l = &t.loop_config;
    assert_eq!((l.core_start, l.core_end, l.num_passes), (6, 18, 3));
    assert_eq!(l.gate_rank, 64);
    // 24 distinct layers, but 12 of them run three times: 24 + 12 * 2 = 48.
    assert_eq!(t.effective_depth(), 48);
}

#[test]
fn test_position_is_relative_not_rope() {
    let t = fixture().text_config;
    assert_eq!(t.d_rel, 16);
    assert_eq!(t.rel_extent, 1024);
    assert_eq!(t.sconv_kernel_size, 4);
}

#[test]
fn test_knows_which_layers_slide() {
    let t = fixture().text_config;
    assert_eq!(t.sliding_window_size, 1024);
    assert_eq!(t.local_layer_ids.len(), 20);
    // 20 of 24 layers are local, so only four attend globally.
    let global: Vec<usize> = (0..t.num_hidden_layers)
        .filter(|&l| !t.is_sliding(l))
        .collect();
    assert_eq!(global.len(), 4);
}

#[test]
fn test_softcaps_are_present_and_parsed() {
    let t = fixture().text_config;
    assert_eq!(t.mlp_gate_softcap, 4.0);
    assert_eq!(t.mlp_up_softcap, 25.0);
    assert_eq!(t.final_logit_softcapping, Some(30.0));
}

#[test]
fn test_rejects_a_loop_core_outside_the_stack() {
    let mut t = fixture().text_config;
    t.loop_config.core_end = t.num_hidden_layers + 1;
    let err = t
        .validate()
        .expect_err("a core past the last layer should be refused");
    assert!(err.to_string().contains("is not inside"), "got: {err}");
}

#[test]
fn test_rejects_more_active_experts_than_exist() {
    let mut t = fixture().text_config;
    t.num_experts_per_tok = t.n_routed_experts + 1;
    let err = t
        .validate()
        .expect_err("routing more experts than exist should be refused");
    assert!(err.to_string().contains("only"), "got: {err}");
}
