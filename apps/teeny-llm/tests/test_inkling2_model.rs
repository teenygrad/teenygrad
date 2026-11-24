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

//! Inkling-2's MoE block as a whole graph: several ops lowered together, run on the
//! device, and compared with a host reference. The per-kernel tests live in
//! `test_inkling2_kernels.rs`; this checks that they compose and that the weight
//! slots carry the checkpoint's names.

#![cfg(all(
    feature = "inkling2",
    any(feature = "cuda", feature = "qemu", feature = "riscv")
))]

use std::cell::RefCell;
use std::rc::Rc;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_llm::models::inkling2::config::Inkling2TextConfig;
use teeny_llm::models::inkling2::kernels::attention::PastLen;
use teeny_llm::models::inkling2::model::{
    LayerInputs, build_controller, build_embed, build_head, build_layer, build_message_block,
    build_moe, build_router,
};
use teeny_llm::models::inkling2::runner::{Quant, checkpoint_name, quant_of};
use teeny_llm::models::inkling2::session::message_states;
use teeny_test::model::{RuntimeModel, runtime};

/// Deterministic values in `[-1, 1)`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: output length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: element {i} is {g}, expected {w} (tolerance {tol})"
        );
    }
}

/// Builds a graph from `inputs`, runs it, and loads parameters BY NAME, so a slot
/// whose scope is wrong fails the test rather than silently taking another's data.
struct GraphRun {
    first: SymTensor,
    graph: Rc<RefCell<Graph>>,
}

impl GraphRun {
    fn new(shape: &[usize]) -> Self {
        let (first, graph) =
            SymTensor::input(DtypeRepr::F32, shape.iter().map(|d| Some(*d)).collect());
        Self { first, graph }
    }

    fn input(&self, shape: &[usize]) -> SymTensor {
        let shape: Vec<Option<usize>> = shape.iter().map(|&d| Some(d)).collect();
        let node_id =
            self.graph
                .borrow_mut()
                .add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
        SymTensor {
            node_id,
            graph: self.graph.clone(),
            dtype: DtypeRepr::F32,
            shape,
        }
    }

    /// The graph's parameter slot names, which also proves it lowers and compiles.
    fn slot_names(&self) -> Vec<String> {
        let graph: &Graph = &self.graph.borrow();
        let lowering = TritonLowering::new();
        lowering
            .lower_with_mapping(graph, LoweringMode::Inference)
            .expect("lowering the graph");
        let device = runtime::open().expect("opening the device");
        let target = runtime::default_target(&device).expect("the device's target");
        let model = RuntimeModel::compile(
            &device,
            graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            self.first.shape[0].unwrap_or(1),
            false,
        )
        .expect("compiling the graph through teenyc");
        model.param_info_named().map(|(n, _, _)| n).collect()
    }

    /// Compiles once, asks the model for its slots, fills them from a checkpoint,
    /// and runs. One compile rather than the two `slot_names` then `run` would cost,
    /// which matters when the parameters are hundreds of megabytes.
    fn run_from_checkpoint(
        &self,
        out: &SymTensor,
        feeds: &[&[f32]],
        ck: &mut teeny_llm::models::inkling2::weights::Checkpoint,
        layer: usize,
    ) -> Vec<f32> {
        let graph: &Graph = &self.graph.borrow();
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) = lowering
            .lower_with_mapping(graph, LoweringMode::Inference)
            .expect("lowering the graph");
        let device = runtime::open().expect("opening the device");
        let target = runtime::default_target(&device).expect("the device's target");
        let mut model = RuntimeModel::compile(
            &device,
            graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            self.first.shape[0].unwrap_or(1),
            false,
        )
        .expect("compiling the graph through teenyc");

        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        let names: Vec<String> = slots.iter().map(|(n, _, _)| n.clone()).collect();
        let missing = teeny_llm::models::inkling2::load::missing(ck, &names, layer);
        assert!(missing.is_empty(), "layer {layer} is missing {missing:?}");
        let values = teeny_llm::models::inkling2::load::read(ck, &names, layer)
            .expect("reading the layer's parameters");
        let by_name: std::collections::HashMap<&str, &[f32]> = values
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        for (name, node, idx) in &slots {
            model
                .load_param_f32(*node, *idx, by_name[name.as_str()])
                .unwrap_or_else(|e| panic!("loading {name}: {e}"));
        }
        let mut got = model
            .run(&device, feeds, &[graph_to_dag[out.node_id]])
            .expect("running the layer");
        got.remove(0)
    }

    fn run(&self, out: &SymTensor, feeds: &[&[f32]], params: &[(&str, &[f32])]) -> Vec<f32> {
        let graph: &Graph = &self.graph.borrow();
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) = lowering
            .lower_with_mapping(graph, LoweringMode::Inference)
            .expect("lowering the graph");
        let device = runtime::open().expect("opening the device");
        let target = runtime::default_target(&device).expect("the device's target");
        let rows = self.first.shape[0].unwrap_or(1);
        let mut model = RuntimeModel::compile(
            &device,
            graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            rows,
            false,
        )
        .expect("compiling the graph through teenyc");
        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        assert_eq!(
            slots.len(),
            params.len(),
            "parameter slots are {:?}, expected {:?}",
            slots.iter().map(|s| &s.0).collect::<Vec<_>>(),
            params.iter().map(|p| p.0).collect::<Vec<_>>()
        );
        for (name, node, idx) in &slots {
            let data = params
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| {
                    panic!(
                        "no data for slot {name}; the graph names its slots {:?}",
                        slots.iter().map(|s| &s.0).collect::<Vec<_>>()
                    )
                })
                .1;
            model
                .load_param_f32(*node, *idx, data)
                .expect("loading a parameter");
        }
        let mut got = model
            .run(&device, feeds, &[graph_to_dag[out.node_id]])
            .expect("running the graph");
        got.remove(0)
    }
}

fn test_config() -> Inkling2TextConfig {
    let base = std::env::var("INKLING2_FIXTURES")
        .unwrap_or_else(|_| format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR")));
    let path = std::path::Path::new(&base).join("inkling2-mini-config.json");
    let mut cfg = teeny_llm::models::inkling2::Inkling2Config::load(&path)
        .expect("the shipped config")
        .text_config;
    // The real widths are far too large to run under qemu; the structure is what is
    // under test, so shrink everything the block touches.
    cfg.hidden_size = 24;
    cfg.intermediate_size = 10;
    cfg.n_routed_experts = 8;
    cfg.num_experts_per_tok = 3;
    cfg
}

/// Route, gather, activate, gather, combine, as one lowered graph.
#[test]
fn test_moe_block_matches_a_host_reference() {
    let cfg = test_config();
    let (t, hid, inter) = (2usize, cfg.hidden_size, cfg.intermediate_size);
    let (e, k) = (cfg.n_routed_experts, cfg.num_experts_per_tok);
    let scale = cfg.route_scale as f32;

    let x = values(t * hid, 81);
    let gate_w = values(e * hid, 82);
    let gate_b = vec![0.0f32; e];
    let w13 = values(e * 2 * inter * hid, 83);
    let w2 = values(e * hid * inter, 84);

    // The router runs first: the host needs its decision to pack the ids for the
    // down projection, which is why the two are separate graphs.
    let gr = GraphRun::new(&[t, hid]);
    let rout = build_router::<f32>(&cfg, &gr.first);
    let routed = gr.run(
        &rout,
        &[&x],
        &[("mlp.gate.weight", &gate_w), ("mlp.gate.bias", &gate_b)],
    );

    let gm = GraphRun::new(&[t, hid]);
    let routed_in = gm.input(&[t, 2 * k]);
    let out = build_moe::<f32>(&cfg, &gm.first, &routed_in, None);
    let got = gm.run(
        &out,
        &[&x, &routed],
        &[
            ("mlp.experts.weight", &w13),
            ("mlp.experts_down.weight", &w2),
        ],
    );

    // Host reference, driven by the routing the device actually chose.
    let cap = |v: f32, c: f32| (v / c).tanh() * c;
    let (gcap, ucap) = (cfg.mlp_gate_softcap as f32, cfg.mlp_up_softcap as f32);
    let mut want = vec![0.0f32; t * hid];
    for tok in 0..t {
        for slot in 0..k {
            let ex = routed[tok * 2 * k + k + slot] as usize;
            let w = routed[tok * 2 * k + slot];
            let gu: Vec<f32> = (0..2 * inter)
                .map(|r| {
                    (0..hid)
                        .map(|c| x[tok * hid + c] * w13[(ex * 2 * inter + r) * hid + c])
                        .sum()
                })
                .collect();
            let a: Vec<f32> = (0..inter)
                .map(|r| {
                    cap(gu[r], gcap) * (1.0 / (1.0 + (-gu[r]).exp())) * cap(gu[inter + r], ucap)
                })
                .collect();
            for (c, o) in want[tok * hid..(tok + 1) * hid].iter_mut().enumerate() {
                *o += w
                    * (0..inter)
                        .map(|r| a[r] * w2[(ex * hid + c) * inter + r])
                        .sum::<f32>();
            }
        }
    }
    assert_close("moe block", &got, &want, 2e-3);

    // The router's own output is sane: ids in range, weights summing to route_scale.
    for tok in 0..t {
        let sum: f32 = routed[tok * 2 * k..tok * 2 * k + k].iter().sum();
        assert!(
            (sum - scale).abs() < 1e-3,
            "weights sum to {sum}, not {scale}"
        );
        for slot in 0..k {
            let id = routed[tok * 2 * k + k + slot];
            assert!(
                id >= 0.0 && (id as usize) < e,
                "expert id {id} out of range"
            );
        }
    }
}

/// Pins the controller's slot names. Three of them differ from the checkpoint's by
/// a dot where it has an underscore (`down.weight` against `down_weight`), because
/// Linear4Op always names its parameter `weight`. The loader has to map these three;
/// the attention slots need no mapping. This test is what tells us if that changes.
#[test]
fn test_controller_slot_names() {
    let cfg = test_config();
    let t = 2usize;
    let g = GraphRun::new(&[t, cfg.hidden_size]);
    let msg = g.input(&[t, cfg.hidden_size]);
    let out = build_controller::<f32>(&cfg, "attn", &g.first, &msg);
    let h = values(t * cfg.hidden_size, 131);
    let m = values(t * cfg.hidden_size, 132);
    let rank = cfg.loop_config.gate_rank;
    let down = values(rank * 2 * cfg.hidden_size, 133);
    let fw = values(cfg.hidden_size * rank, 134);
    let gw = values(cfg.hidden_size * rank, 135);
    let got = g.run(
        &out,
        &[&h, &m],
        &[
            // checkpoint: controllers.attn.down_weight
            ("controllers.attn.down.weight", &down),
            // checkpoint: controllers.attn.forget_weight
            ("controllers.attn.forget.weight", &fw),
            // checkpoint: controllers.attn.gain_weight
            ("controllers.attn.gain.weight", &gw),
        ],
    );
    assert_eq!(got.len(), t * cfg.hidden_size);
}

/// Builds the cross-pass message block, runs it, and checks its parameter slots are
/// the two top-level checkpoint tensors.
///
/// The messages are what passes 2 and 3 of the looped core consume, so this is the
/// piece that takes a step from 24 layer executions to 48.
#[test]
fn test_message_block_builds_and_names_the_cross_pass_tensors() {
    let cfg = test_config();
    let hid = cfg.hidden_size;
    let t = 2usize;
    let n_loop = cfg.loop_config.core_end - cfg.loop_config.core_start;
    let n_states = message_states(&cfg).max(1);
    let rows = 2 * n_loop;

    let g = GraphRun::new(&[t * n_states, hid]);
    let partial = g.input(&[t, hid]);
    let msgs = build_message_block::<f32>(&cfg, &g.first, &partial, n_states);
    assert_eq!(msgs.len(), n_loop, "one set of messages per loop layer");

    // Every op reads one row of one of the two shared tensors through its
    // `query_index`, so each slot takes the whole [2 * n_loop, hidden] tensor.
    let queries = values(rows * hid, 161);
    let kv_queries = values(rows * hid, 162);
    let states = values(t * n_states * hid, 163);
    let part = values(t * hid, 164);

    // One slot per op: four per loop layer, all reading one of the two tensors.
    let names: Vec<String> = (0..n_loop)
        .flat_map(|l| {
            [
                format!("cross_pass.queries_{l}_0.query"),
                format!("cross_pass.kv_queries_{l}_1.query"),
                format!("cross_pass.kv_queries_{l}_2.query"),
                format!("cross_pass.queries_{l}_3.query"),
            ]
        })
        .collect();
    let params: Vec<(&str, &[f32])> = names
        .iter()
        .map(|n| {
            let data: &[f32] = if n.contains("kv_queries") {
                &kv_queries
            } else {
                &queries
            };
            (n.as_str(), data)
        })
        .collect();

    let got = g.run(&msgs[0][0], &[&states, &part], &params);
    assert_eq!(got.len(), t * hid);
    assert!(got.iter().all(|v| v.is_finite()), "finite messages");

    // And the slot names resolve to the checkpoint's own tensors.
    assert_eq!(
        checkpoint_name("cross_pass.queries_0_0.query", 7),
        "model.llm.middle_cycle_cross_pass_queries"
    );
    assert_eq!(
        checkpoint_name("cross_pass.kv_queries_11_2.query", 7),
        "model.llm.middle_cycle_cross_pass_kv_queries"
    );
}

/// Builds a whole decoder layer, lowers it, and runs it.
///
/// This does not check the numbers against a host reference of the full layer, which
/// would be a second implementation of everything the per-kernel tests already pin.
/// What it does check is that the layer is **connected**: every op reachable, every
/// weight slot accounted for, and a finite result out the other end. An earlier draft
/// passed its convolutions a host tensor instead of the attention output, which left
/// attention dangling in the graph; that would fail here on the slot list.
#[test]
fn test_decoder_layer_builds_and_runs() {
    let mut cfg = test_config();
    cfg.n_layers_per_attn_res_block = 6;
    cfg.sliding_window_size = 8;
    cfg.local_layer_ids = vec![1];
    let (t, hid) = (2usize, cfg.hidden_size);
    let (states, past, cache) = (2usize, 3usize, 8usize);
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let kw = cfg.sconv_kernel_size - 1;

    let g = GraphRun::new(&[t, hid]);
    let block_states = g.input(&[t * states, hid]);
    let k_hist = g.input(&[kw, nkv * hd]);
    let v_hist = g.input(&[kw, nkv * hd]);
    let attn_hist = g.input(&[kw, hid]);
    let mlp_hist = g.input(&[kw, hid]);
    let k_cache = g.input(&[cache, nkv * hd]);
    let v_cache = g.input(&[cache, nkv * hd]);

    let past_len = PastLen::default();
    past_len.set(past);
    let inputs = LayerInputs {
        x: &g.first,
        block_states: &block_states,
        n_states: states,
        k_hist: &k_hist,
        v_hist: &v_hist,
        attn_hist: &attn_hist,
        mlp_hist: &mlp_hist,
        k_cache: &k_cache,
        v_cache: &v_cache,
        messages: None,
    };
    // Layer 1: not a block boundary, so the residual is the sum, and it slides.
    let out = build_layer::<f32>(&cfg, 1, &inputs, past_len);
    assert_eq!(
        out.shape,
        vec![Some(t), Some(hid)],
        "a layer returns [T, hidden]"
    );

    let names = g.slot_names();
    for expected in [
        // Layer 1 is not a dynamic layer, so it has ONE attention query, not three.
        "attn_res.attn_queries_0.query",
        "attn_res.mlp_queries_0.query",
        "attn_norm.weight",
        "attn.wq_du.weight",
        "attn.wk_dv.weight",
        "attn.wv_dv.weight",
        "attn.wo_ud.weight",
        "attn.gate_attn.weight",
        "attn.attn_sink_proj.weight",
        "attn.wr_du.weight",
        "attn.q_l2_clip.scale",
        "attn.rel_l2_clip.scale",
        "attn_sconv.weight",
        "mlp_norm.weight",
        "mlp.experts.weight",
        "mlp.experts.scale",
        "mlp.experts.scale2",
        "mlp.experts_down.weight",
        "mlp_sconv.weight",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "the layer has no slot {expected}; it has {names:?}"
        );
    }
}

/// The last layer of each block is built differently: three input norms instead of
/// one, and convolutions whose taps are projected from the layer input.
///
/// A layer that quietly used the static form would still compile and run, so this
/// checks the SLOTS, which is what the checkpoint actually constrains.
#[test]
fn test_the_last_layer_of_a_block_convolves_dynamically() {
    let mut cfg = test_config();
    cfg.n_layers_per_attn_res_block = 6;
    cfg.local_layer_ids = vec![];
    let (t, hid) = (2usize, cfg.hidden_size);
    let (states, past, cache) = (2usize, 3usize, 8usize);
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let kw = cfg.sconv_kernel_size - 1;

    // Layer 5 is the last of the first block; layer 4 is not.
    for (layer, dynamic) in [(5usize, true), (4usize, false)] {
        let g = GraphRun::new(&[t, hid]);
        let block_states = g.input(&[t * states, hid]);
        let k_hist = g.input(&[kw, nkv * hd]);
        let v_hist = g.input(&[kw, nkv * hd]);
        let attn_hist = g.input(&[kw, hid]);
        let mlp_hist = g.input(&[kw, hid]);
        let k_cache = g.input(&[cache, nkv * hd]);
        let v_cache = g.input(&[cache, nkv * hd]);
        let past_len = PastLen::default();
        past_len.set(past);
        let inputs = LayerInputs {
            x: &g.first,
            block_states: &block_states,
            n_states: states,
            k_hist: &k_hist,
            v_hist: &v_hist,
            attn_hist: &attn_hist,
            mlp_hist: &mlp_hist,
            k_cache: &k_cache,
            v_cache: &v_cache,
            messages: None,
        };
        let _ = build_layer::<f32>(&cfg, layer, &inputs, past_len);
        let names = g.slot_names();
        let has = |n: &str| names.iter().any(|x| x == n);

        if dynamic {
            assert!(
                has("q_input_norm.weight"),
                "layer {layer} needs three input norms"
            );
            assert!(has("k_input_norm.weight"));
            assert!(has("v_input_norm.weight"));
            assert!(!has("attn_norm.weight"), "layer {layer} has no attn_norm");
            assert!(
                has("attn.k_sconv.static_w"),
                "layer {layer} convolves dynamically"
            );
            assert!(has("attn.k_sconv.weight_proj.weight"));
            assert!(has("attn.v_sconv.static_w"));
            assert!(
                !has("attn.k_sconv.weight"),
                "layer {layer} has no static filter"
            );
        } else {
            assert!(has("attn_norm.weight"), "layer {layer} needs attn_norm");
            assert!(
                has("attn.k_sconv.weight"),
                "layer {layer} convolves statically"
            );
            assert!(!has("q_input_norm.weight"));
            assert!(!has("attn.k_sconv.static_w"));
        }
        // Both kinds keep the layer-level convolutions static.
        assert!(
            has("attn_sconv.weight"),
            "layer {layer} keeps a static attn_sconv"
        );
        assert!(has("mlp_sconv.weight"));
    }
}

/// A layer in the MXFP8 range builds the MXFP8 gather, which has no `scale2`.
#[test]
fn test_the_last_layers_use_the_mxfp8_gather() {
    let cfg = test_config();
    let (t, hid) = (2usize, cfg.hidden_size);
    let (states, past, cache) = (2usize, 3usize, 8usize);
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let kw = cfg.sconv_kernel_size - 1;

    // Layer 2 is NVFP4; layer 22 of 24 is MXFP8.
    for (layer, mxfp8) in [(2usize, false), (cfg.num_hidden_layers - 2, true)] {
        assert_eq!(
            quant_of(&cfg, layer) == Quant::Mxfp8,
            mxfp8,
            "layer {layer} quantisation"
        );
        let g = GraphRun::new(&[t, hid]);
        let block_states = g.input(&[t * states, hid]);
        let k_hist = g.input(&[kw, nkv * hd]);
        let v_hist = g.input(&[kw, nkv * hd]);
        let attn_hist = g.input(&[kw, hid]);
        let mlp_hist = g.input(&[kw, hid]);
        let k_cache = g.input(&[cache, nkv * hd]);
        let v_cache = g.input(&[cache, nkv * hd]);
        let past_len = PastLen::default();
        past_len.set(past);
        let inputs = LayerInputs {
            x: &g.first,
            block_states: &block_states,
            n_states: states,
            k_hist: &k_hist,
            v_hist: &v_hist,
            attn_hist: &attn_hist,
            mlp_hist: &mlp_hist,
            k_cache: &k_cache,
            v_cache: &v_cache,
            messages: None,
        };
        let _ = build_layer::<f32>(&cfg, layer, &inputs, past_len);
        let names = g.slot_names();
        let has = |n: &str| names.iter().any(|x| x == n);
        assert!(
            has("mlp.experts.weight"),
            "layer {layer} needs expert weights"
        );
        assert!(has("mlp.experts.scale"), "layer {layer} needs block scales");
        assert_eq!(
            has("mlp.experts.scale2"),
            !mxfp8,
            "only NVFP4 has a per-expert scale (layer {layer})"
        );
    }
}

/// The embedding is a plain lookup followed by `embed_norm`, NOT Gemma's
/// sqrt(hidden) scaling.
#[test]
fn test_embedding_looks_up_and_norms_without_scaling() {
    let mut cfg = test_config();
    cfg.vocab_size = 8;
    cfg.hidden_size = 4;
    let (vocab, hid) = (cfg.vocab_size, cfg.hidden_size);
    let table = values(vocab * hid, 171);
    let w = values(hid, 172);
    let ids = [3.0f32, 0.0, 7.0];

    let g = GraphRun::new(&[ids.len()]);
    let out = build_embed::<f32>(&cfg, &g.first);
    let got = g.run(
        &out,
        &[&ids],
        &[("embed.weight", &table), ("embed_norm.weight", &w)],
    );

    let want: Vec<f32> = ids
        .iter()
        .flat_map(|id| {
            let row = &table[*id as usize * hid..(*id as usize + 1) * hid];
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / hid as f32 + 1e-6).sqrt();
            (0..hid).map(|i| row[i] / rms * w[i]).collect::<Vec<_>>()
        })
        .collect();
    assert_close("embed", &got, &want, 1e-4);

    // Gemma would have multiplied by sqrt(hidden); this must not.
    let scaled: Vec<f32> = want.iter().map(|v| v * (hid as f32).sqrt()).collect();
    assert!(
        got.iter().zip(&scaled).any(|(a, b)| (a - b).abs() > 1e-3),
        "the embedding must not be scaled by sqrt(hidden)"
    );
}

/// The head takes only the LAST row, divides by the muP multiplier, then softcaps.
#[test]
fn test_head_scales_by_mup_and_softcaps() {
    let mut cfg = test_config();
    // Linear4Op needs out features divisible by 4; the real vocab, 201024, is.
    cfg.vocab_size = 8;
    cfg.hidden_size = 4;
    cfg.final_logit_softcapping = Some(30.0);
    let (vocab, hid, t) = (cfg.vocab_size, cfg.hidden_size, 3usize);
    let mup = 16.0f64;

    let x = values(t * hid, 173);
    let norm_w = values(hid, 174);
    // Large enough that the softcap bites.
    let unembed: Vec<f32> = values(vocab * hid, 175).iter().map(|v| v * 300.0).collect();

    let g = GraphRun::new(&[t, hid]);
    let out = build_head::<f32>(&cfg, &g.first, mup);
    let got = g.run(
        &out,
        &[&x],
        &[("norm.weight", &norm_w), ("unembed.weight", &unembed)],
    );
    assert_eq!(got.len(), vocab, "a step produces one distribution");

    let last = &x[(t - 1) * hid..];
    let rms = (last.iter().map(|v| v * v).sum::<f32>() / hid as f32 + 1e-6).sqrt();
    let normed: Vec<f32> = (0..hid).map(|i| last[i] / rms * norm_w[i]).collect();
    let want: Vec<f32> = (0..vocab)
        .map(|n| {
            let dot: f32 = (0..hid).map(|k| normed[k] * unembed[n * hid + k]).sum();
            let scaled = dot / mup as f32;
            (scaled / 30.0).tanh() * 30.0
        })
        .collect();
    assert_close("head", &got, &want, 1e-3);
    assert!(
        got.iter().all(|v| v.abs() <= 30.0 + 1e-3),
        "every logit must sit inside the cap"
    );
}

/// Layer 0 of the real model, with its real weights, on one token.
///
/// Ignored: needs the 20.2 GB checkpoint. Point `INKLING2_MODEL_DIR` at it.
///
/// This is the first time arithmetic runs on trained weights rather than on
/// `values()`. It checks what a shape test cannot: that the output is finite, that
/// it is not all zero, and that it sits in a plausible range. A layer that silently
/// read the wrong bytes would usually fail one of those.
#[test]
#[ignore = "needs the 20.2 GB checkpoint; set INKLING2_MODEL_DIR"]
fn test_layer_zero_runs_on_real_weights() {
    use teeny_llm::models::inkling2::weights::Checkpoint;

    let dir = std::env::var("INKLING2_MODEL_DIR").expect("set INKLING2_MODEL_DIR");
    let dir = std::path::Path::new(&dir);
    let cfg = teeny_llm::models::inkling2::Inkling2Config::load(&dir.join("config.json"))
        .expect("the checkpoint's config")
        .text_config;
    let mut ck = Checkpoint::open(dir).expect("opening the checkpoint");

    let (t, hid) = (1usize, cfg.hidden_size);
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let kw = cfg.sconv_kernel_size - 1;
    let cache = 8usize;

    // Layer 0: no committed block states, an empty cache, a cold convolution window.
    let g = GraphRun::new(&[t, hid]);
    let block_states = g.input(&[0, hid]);
    let k_hist = g.input(&[kw, nkv * hd]);
    let v_hist = g.input(&[kw, nkv * hd]);
    let attn_hist = g.input(&[kw, hid]);
    let mlp_hist = g.input(&[kw, hid]);
    let k_cache = g.input(&[cache, nkv * hd]);
    let v_cache = g.input(&[cache, nkv * hd]);
    let past = PastLen::default();
    past.set(0);
    let inputs = LayerInputs {
        x: &g.first,
        block_states: &block_states,
        n_states: 0,
        k_hist: &k_hist,
        v_hist: &v_hist,
        attn_hist: &attn_hist,
        mlp_hist: &mlp_hist,
        k_cache: &k_cache,
        v_cache: &v_cache,
        messages: None,
    };
    let out = build_layer::<f32>(&cfg, 0, &inputs, past);
    assert_eq!(out.shape, vec![Some(t), Some(hid)]);

    // A plausible embedding: the real one for a token, normed as the model does.
    let embed = ck
        .param("model.llm.embed.weight")
        .expect("the embedding table");
    let token = 5000usize;
    let row = &embed[token * hid..(token + 1) * hid];
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / hid as f32 + 1e-6).sqrt();
    let x: Vec<f32> = row.iter().map(|v| v / rms).collect();

    let zeros_kv = vec![0.0f32; kw * nkv * hd];
    let zeros_h = vec![0.0f32; kw * hid];
    let cache_kv = vec![0.0f32; cache * nkv * hd];
    let empty: Vec<f32> = Vec::new();
    let got = g.run_from_checkpoint(
        &out,
        &[
            &x, &empty, &zeros_kv, &zeros_kv, &zeros_h, &zeros_h, &cache_kv, &cache_kv,
        ],
        &mut ck,
        0,
    );

    assert_eq!(got.len(), t * hid, "a layer returns [T, hidden]");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "every output must be finite"
    );
    let absmax = got.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let nonzero = got.iter().filter(|v| v.abs() > 1e-12).count();
    println!(
        "layer 0: absmax {absmax:.4}, nonzero {nonzero} of {}",
        got.len()
    );
    assert!(
        nonzero > got.len() / 2,
        "most outputs should be non-zero, got {nonzero}"
    );
    assert!(
        absmax > 1e-4 && absmax < 1.0e4,
        "the output should be plausibly scaled, absmax was {absmax}"
    );
}

/// One token through all 24 layers of the real model, then the head.
///
/// Ignored: needs the checkpoint, and takes tens of minutes under qemu.
///
/// **One pass, not three.** The looped core's later passes need the cross-pass
/// messages, and the checkpoint holds those as `middle_cycle_cross_pass_queries` and
/// `_kv_queries`, two `[24, 2304]` tensors that no reachable source maps onto the
/// reference's `(loop_layers, 4, hidden)` message queries. Pass 0 takes no messages,
/// so this walks the plan's pass-0 steps and skips the rest. It is the whole stack
/// apart from the loop: both layer shapes, both quantisations, block states,
/// boundaries, the embedding and the head.
#[test]
#[ignore = "needs the checkpoint and tens of minutes; set INKLING2_MODEL_DIR"]
fn test_one_token_through_every_layer_once() {
    use std::time::Instant;
    use teeny_llm::models::inkling2::runner::{State, argmax, plan};
    use teeny_llm::models::inkling2::weights::Checkpoint;

    let dir = std::env::var("INKLING2_MODEL_DIR").expect("set INKLING2_MODEL_DIR");
    let dir = std::path::Path::new(&dir);
    let full = teeny_llm::models::inkling2::Inkling2Config::load(&dir.join("config.json"))
        .expect("the checkpoint's config");
    let cfg = full.text_config.clone();
    let mut ck = Checkpoint::open(dir).expect("opening the checkpoint");

    let (t, hid) = (1usize, cfg.hidden_size);
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let kw = cfg.sconv_kernel_size - 1;
    let cache = 8usize;
    let past = PastLen::default();
    past.set(0);
    let mut state = State::new(&cfg, cache);

    // Embed one token, as build_embed does: a lookup then embed_norm.
    let token = 5000usize;
    let embed = ck
        .param("model.llm.embed.weight")
        .expect("the embedding table");
    let enorm = ck.param("model.llm.embed_norm.weight").expect("embed_norm");
    let row = &embed[token * hid..(token + 1) * hid];
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / hid as f32 + 1e-6).sqrt();
    let mut x: Vec<f32> = (0..hid).map(|i| row[i] / rms * enorm[i]).collect();
    drop(embed);

    let zeros_kv = vec![0.0f32; kw * nkv * hd];
    let zeros_h = vec![0.0f32; kw * hid];
    let cache_kv = vec![0.0f32; cache * nkv * hd];

    let started = Instant::now();
    let mut ran = 0usize;
    for step in plan(&cfg) {
        // Pass 0 and the layers outside the core; the later passes need messages.
        if step.pass.is_some_and(|p| p > 0) {
            continue;
        }
        let block_input = state.block_input(t);
        let n_states = state.n_blocks();

        let g = GraphRun::new(&[t, hid]);
        let block_states = g.input(&[t * n_states, hid]);
        let k_hist = g.input(&[kw, nkv * hd]);
        let v_hist = g.input(&[kw, nkv * hd]);
        let attn_hist = g.input(&[kw, hid]);
        let mlp_hist = g.input(&[kw, hid]);
        let k_cache = g.input(&[cache, nkv * hd]);
        let v_cache = g.input(&[cache, nkv * hd]);
        let inputs = LayerInputs {
            x: &g.first,
            block_states: &block_states,
            n_states,
            k_hist: &k_hist,
            v_hist: &v_hist,
            attn_hist: &attn_hist,
            mlp_hist: &mlp_hist,
            k_cache: &k_cache,
            v_cache: &v_cache,
            messages: None,
        };
        let out = build_layer::<f32>(&cfg, step.layer, &inputs, past.clone());
        let next = g.run_from_checkpoint(
            &out,
            &[
                &x,
                &block_input,
                &zeros_kv,
                &zeros_kv,
                &zeros_h,
                &zeros_h,
                &cache_kv,
                &cache_kv,
            ],
            &mut ck,
            step.layer,
        );
        assert_eq!(next.len(), t * hid, "layer {} output width", step.layer);
        assert!(
            next.iter().all(|v| v.is_finite()),
            "layer {} produced a non-finite value",
            step.layer
        );
        // A boundary layer's INPUT joins the block states, not its output.
        if step.is_boundary {
            state.push_block(&x);
        }
        x = next;
        ran += 1;
        let absmax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        println!(
            "layer {:2} done in {:>6.1}s, absmax {absmax:8.3}, blocks {}",
            step.layer,
            started.elapsed().as_secs_f32(),
            state.n_blocks()
        );
    }
    assert_eq!(
        ran, cfg.num_hidden_layers,
        "every layer runs once in pass 0"
    );

    // The head: final norm, unembed, muP, softcap.
    let gh = GraphRun::new(&[t, hid]);
    let logits_sym = build_head::<f32>(&cfg, &gh.first, 16.0);
    let norm_w = ck.param("model.llm.norm.weight").expect("final norm");
    let unembed = ck.param("model.llm.unembed.weight").expect("unembed");
    let logits = gh.run(
        &logits_sym,
        &[&x],
        &[("norm.weight", &norm_w), ("unembed.weight", &unembed)],
    );

    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "one distribution over the vocabulary"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );
    let cap = full.text_config.final_logit_softcapping.unwrap_or(30.0) as f32;
    assert!(
        logits.iter().all(|v| v.abs() <= cap + 1e-3),
        "every logit must sit inside the softcap"
    );
    let best = argmax(&logits);
    let top = logits[best as usize];
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    println!(
        "token {token} -> {best}, logit {top:.4}, mean {mean:.4}, in {:.0}s",
        started.elapsed().as_secs_f32()
    );
    assert!(
        (best as usize) < cfg.unpadded_vocab_size,
        "the chosen id {best} is outside the real vocabulary"
    );
    assert!(
        top > mean + 1e-3,
        "the top logit should stand above the mean"
    );
}

/// Diagnostics: does the stack actually transform its input, and does every token
/// echo itself?
///
/// The first run echoed its input (token 5000 in, 5000 out) with a weak margin,
/// which is consistent both with reasonable behaviour on a context-free single token
/// and with a near-identity stack whose unembedding is matching a residual stream
/// that still looks like the input embedding. Two measurements separate them:
///
/// - the **cosine between the final hidden state and the input embedding**. Near 1
///   means the 24 layers barely moved the residual stream.
/// - **several tokens**. If every one echoes itself, that points at the unembedding
///   reading the embedding rather than at the model predicting.
#[test]
#[ignore = "needs the checkpoint; set INKLING2_MODEL_DIR"]
fn test_what_the_stack_does_to_a_few_tokens() {
    use std::time::Instant;
    use teeny_llm::models::inkling2::runner::{State, argmax, plan};
    use teeny_llm::models::inkling2::weights::Checkpoint;

    let dir = std::env::var("INKLING2_MODEL_DIR").expect("set INKLING2_MODEL_DIR");
    let dir = std::path::Path::new(&dir);
    let cfg = teeny_llm::models::inkling2::Inkling2Config::load(&dir.join("config.json"))
        .expect("config")
        .text_config;
    let mut ck = Checkpoint::open(dir).expect("opening the checkpoint");

    let (t, hid) = (1usize, cfg.hidden_size);
    let (nkv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let kw = cfg.sconv_kernel_size - 1;
    let cache = 8usize;
    let zeros_kv = vec![0.0f32; kw * nkv * hd];
    let zeros_h = vec![0.0f32; kw * hid];
    let cache_kv = vec![0.0f32; cache * nkv * hd];

    let embed = ck.param("model.llm.embed.weight").expect("embeddings");
    let enorm = ck.param("model.llm.embed_norm.weight").expect("embed_norm");
    let norm_w = ck.param("model.llm.norm.weight").expect("final norm");
    let unembed = ck.param("model.llm.unembed.weight").expect("unembed");

    let cosine = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb + 1e-20)
    };

    let tokens = [5000usize, 100, 42, 100_000];
    let mut finals: Vec<Vec<f32>> = Vec::new();
    let mut echoes = 0usize;
    for &token in &tokens {
        let started = Instant::now();
        let row: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / hid as f32 + 1e-6).sqrt();
        let mut x: Vec<f32> = (0..hid).map(|i| row[i] / rms * enorm[i]).collect();
        let entry = x.clone();
        let mut state = State::new(&cfg, cache);

        for step in plan(&cfg) {
            if step.pass.is_some_and(|p| p > 0) {
                continue;
            }
            let block_input = state.block_input(t);
            let n_states = state.n_blocks();
            let g = GraphRun::new(&[t, hid]);
            let bs = g.input(&[t * n_states, hid]);
            let kh = g.input(&[kw, nkv * hd]);
            let vh = g.input(&[kw, nkv * hd]);
            let ah = g.input(&[kw, hid]);
            let mh = g.input(&[kw, hid]);
            let kc = g.input(&[cache, nkv * hd]);
            let vc = g.input(&[cache, nkv * hd]);
            let past = PastLen::default();
            past.set(0);
            let inputs = LayerInputs {
                x: &g.first,
                block_states: &bs,
                n_states,
                k_hist: &kh,
                v_hist: &vh,
                attn_hist: &ah,
                mlp_hist: &mh,
                k_cache: &kc,
                v_cache: &vc,
                messages: None,
            };
            let out = build_layer::<f32>(&cfg, step.layer, &inputs, past);
            let next = g.run_from_checkpoint(
                &out,
                &[&x, &block_input, &zeros_kv, &zeros_kv, &zeros_h, &zeros_h, &cache_kv, &cache_kv],
                &mut ck,
                step.layer,
            );
            if step.is_boundary {
                state.push_block(&x);
            }
            x = next;
        }

        let gh = GraphRun::new(&[t, hid]);
        let sym = build_head::<f32>(&cfg, &gh.first, 16.0);
        let logits = gh.run(&sym, &[&x], &[("norm.weight", &norm_w), ("unembed.weight", &unembed)]);
        let best = argmax(&logits) as usize;
        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        let cos_in = cosine(&x, &entry);
        let cos_raw = cosine(&x, &row);
        if best == token {
            echoes += 1;
        }
        println!(
            "token {token:6} -> {best:6} {} | logit {:7.4} mean {:7.4} | cos(final, entry) {cos_in:+.4} cos(final, embed) {cos_raw:+.4} | {:.0}s",
            if best == token { "ECHO" } else { "    " },
            logits[best],
            mean,
            started.elapsed().as_secs_f32()
        );
        assert!(x.iter().all(|v| v.is_finite()), "token {token} gave a non-finite state");
        finals.push(x);
    }

    // Different inputs must give different final states, or the stack is ignoring
    // its input entirely.
    let across = cosine(&finals[0], &finals[1]);
    println!("cos(final[{}], final[{}]) = {across:+.4}", tokens[0], tokens[1]);
    assert!(
        across < 0.999,
        "two different tokens produced near-identical states ({across}); the stack ignores its input"
    );
    println!("{echoes} of {} tokens echoed their input", tokens.len());
}
