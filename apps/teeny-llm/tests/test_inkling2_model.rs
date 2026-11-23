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

#![cfg(all(feature = "inkling2", any(feature = "cuda", feature = "qemu")))]

use std::cell::RefCell;
use std::rc::Rc;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_llm_core::models::inkling2::config::Inkling2TextConfig;
use teeny_llm_core::models::inkling2::model::{build_moe, build_router, slot_ids};
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
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/inkling2-mini-config.json");
    let mut cfg = teeny_llm_core::models::inkling2::Inkling2Config::load(&path)
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
    let ids = slot_ids(&routed, t, k);

    let gm = GraphRun::new(&[t, hid]);
    let routed_in = gm.input(&[t, 2 * k]);
    let ids_in = gm.input(&[t * k, 2]);
    let out = build_moe::<f32>(&cfg, &gm.first, &routed_in, &ids_in);
    let got = gm.run(
        &out,
        &[&x, &routed, &ids],
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
