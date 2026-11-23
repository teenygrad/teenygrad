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
    LayerInputs, build_controller, build_layer, build_moe, build_router,
};
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
    let out = build_moe::<f32>(&cfg, &gm.first, &routed_in);
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
    let kv_len = past + t;
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
    let out = build_layer::<f32>(&cfg, 1, &inputs, past, kv_len, past_len);
    assert_eq!(
        out.shape,
        vec![Some(t), Some(hid)],
        "a layer returns [T, hidden]"
    );

    let names = g.slot_names();
    for expected in [
        "attn_res.attn_queries_0.query",
        "attn_res.attn_queries_2.query",
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
        let _ = build_layer::<f32>(&cfg, layer, &inputs, past, past + t, past_len);
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
