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

//! Each LLM kernel on its own: a one-op graph compiled through teenyc, run on the device
//! (CUDA, or RISC-V under qemu), and compared with a host reference of the formula in the
//! kernel's module docs. Covers the shared kernels (`teeny_llm::kernels`) and
//! FunctionGemma's own, without needing the gated model.

#![cfg(any(feature = "cuda", feature = "qemu"))]

use std::cell::RefCell;
use std::rc::Rc;

use teeny_core::graph::{CustomData, CustomOp, DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_kernels::graph::TritonLowering;
use teeny_llm::kernels::{
    attention::{AttentionSpec, GqaAttentionOp, PastLen},
    last_row::LastRowOp,
    linear::Linear4Op,
    rope::{RopeOp, rope_tables},
};
use teeny_llm::models::functiongemma::kernels::{
    embed::GemmaEmbedOp, gelu_mul::GemmaGeluMulOp, rmsnorm::GemmaRmsNormOp,
};
use teeny_test::model::{RuntimeModel, runtime};

/// Deterministic values in `[-1, 1)`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// A graph whose inputs are created in feed order.
struct OneOp {
    graph: Rc<RefCell<Graph>>,
    first: SymTensor,
}

impl OneOp {
    fn new(shape: &[usize]) -> Self {
        let (first, graph) =
            SymTensor::input(DtypeRepr::F32, shape.iter().map(|&d| Some(d)).collect());
        Self { graph, first }
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

    /// Records `op` on the first input plus `others`, compiles, loads `params` (in slot
    /// order), runs on `feeds` and returns the op's output.
    fn run(
        &self,
        op: impl CustomOp + 'static,
        others: &[&SymTensor],
        feeds: &[&[f32]],
        params: &[&[f32]],
    ) -> Vec<f32> {
        // Parameter slots are named after the node's scope, as the model names its weights.
        let out = {
            let _scope = name_scope("op");
            self.first.record_custom(CustomData::new(op), others, None)
        };
        let graph: &Graph = &self.graph.borrow();
        let lowering = TritonLowering::new();
        let (_, graph_to_dag) = lowering
            .lower_with_mapping(graph, LoweringMode::Inference)
            .expect("lowering the one-op graph");
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
        .expect("compiling the op through teenyc");
        let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
        assert_eq!(slots.len(), params.len(), "parameter slots: {slots:?}");
        for ((_, node, idx), data) in slots.iter().zip(params) {
            model
                .load_param_f32(*node, *idx, data)
                .expect("loading a parameter");
        }
        let mut got = model
            .run(&device, feeds, &[graph_to_dag[out.node_id]])
            .expect("running the op");
        got.remove(0)
    }
}

/// Asserts `got` matches `want` to `tol`, relative to each value's magnitude (at least 1).
fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: output length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: element {i} is {g}, expected {w} (tolerance {tol})"
        );
    }
}

#[test]
fn test_last_row_selects_the_last_row() {
    let (t, n) = (5, 1500);
    let x = values(t * n, 1);
    let g = OneOp::new(&[t, n]);
    let got = g.run(LastRowOp::<f32>::default(), &[], &[&x], &[]);
    assert_close("last_row", &got, &x[(t - 1) * n..], 0.0);
}

/// `y[m, n] = sum_k x[m, k] w[n, k]`, in both program orders.
#[test]
fn test_linear4_matches_a_host_matmul() {
    let (m, k, n) = (3, 80, 36);
    let x = values(m * k, 2);
    let w = values(n * k, 3);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            (0..k)
                .map(|j| x[(i / n) * k + j] * w[(i % n) * k + j])
                .sum()
        })
        .collect();
    for (label, op) in [
        ("rows", Linear4Op::<f32>::new(k, n, 16)),
        ("reuse", Linear4Op::<f32>::new(k, n, 16).with_weight_reuse()),
    ] {
        let g = OneOp::new(&[m, k]);
        let got = g.run(op, &[], &[&x], &[&w]);
        assert_close(&format!("linear4 ({label})"), &got, &want, 1e-5);
    }
}

/// Hugging Face's `rotate_half` RoPE, with the tables `rope_tables` builds.
#[test]
fn test_rope_matches_rotate_half() {
    let (t, heads, hd) = (3, 2, 16);
    let x = values(t * heads * hd, 4);
    let (cos, sin) = rope_tables(10_000.0, hd, &[4, 5, 6]);
    let half = hd / 2;
    let mut want = vec![0.0; x.len()];
    for r in 0..t {
        for h in 0..heads {
            let base = r * heads * hd + h * hd;
            for i in 0..half {
                let (x1, x2) = (x[base + i], x[base + half + i]);
                let (c, s) = (cos[r * hd + i], sin[r * hd + i]);
                want[base + i] = x1 * c - x2 * s;
                want[base + half + i] = x2 * cos[r * hd + half + i] + x1 * sin[r * hd + half + i];
            }
        }
    }
    let g = OneOp::new(&[t, heads * hd]);
    let c = g.input(&[t, hd]);
    let s = g.input(&[t, hd]);
    let got = g.run(
        RopeOp::<f32>::new(heads, hd),
        &[&c, &s],
        &[&x, &cos, &sin],
        &[],
    );
    assert_close("rope", &got, &want, 1e-5);
}

/// Causal grouped-query attention against a partly filled cache, with and without a
/// sliding window.
#[test]
fn test_gqa_attention_matches_a_host_softmax() {
    let (heads, kv_heads, hd, cache_rows) = (4, 2, 16, 8);
    for (past, t, window) in [(0, 4, None), (3, 2, None), (5, 3, Some(4))] {
        let q = values(t * heads * hd, 10 + past as u64);
        let k_new = values(t * kv_heads * hd, 20 + past as u64);
        let v_new = values(t * kv_heads * hd, 30 + past as u64);
        let k_cache = values(cache_rows * kv_heads * hd, 40);
        let v_cache = values(cache_rows * kv_heads * hd, 50);
        let scale = 1.0 / (hd as f32).sqrt();

        let row = kv_heads * hd;
        let key = |j: usize, kvh: usize, d: usize| {
            if j < past {
                k_cache[j * row + kvh * hd + d]
            } else {
                k_new[(j - past) * row + kvh * hd + d]
            }
        };
        let val = |j: usize, kvh: usize, d: usize| {
            if j < past {
                v_cache[j * row + kvh * hd + d]
            } else {
                v_new[(j - past) * row + kvh * hd + d]
            }
        };
        let mut want = vec![0.0f32; q.len()];
        for r in 0..t {
            let pos = past + r;
            for h in 0..heads {
                let kvh = h / (heads / kv_heads);
                let qrow = &q[r * heads * hd + h * hd..][..hd];
                let visible: Vec<usize> = (0..=pos)
                    .filter(|&j| window.is_none_or(|w| pos - j < w))
                    .collect();
                let logits: Vec<f32> = visible
                    .iter()
                    .map(|&j| (0..hd).map(|d| qrow[d] * key(j, kvh, d)).sum::<f32>() * scale)
                    .collect();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..hd {
                    want[r * heads * hd + h * hd + d] = visible
                        .iter()
                        .zip(&exps)
                        .map(|(&j, e)| e / sum * val(j, kvh, d))
                        .sum();
                }
            }
        }

        let past_len = PastLen::default();
        past_len.set(past);
        let spec = AttentionSpec {
            heads,
            kv_heads,
            head_dim: hd,
            window,
            scale,
        };
        let g = OneOp::new(&[t, heads * hd]);
        let kn = g.input(&[t, row]);
        let vn = g.input(&[t, row]);
        let kc = g.input(&[cache_rows, row]);
        let vc = g.input(&[cache_rows, row]);
        let got = g.run(
            GqaAttentionOp::<f32>::new(spec, past_len),
            &[&kn, &vn, &kc, &vc],
            &[&q, &k_new, &v_new, &k_cache, &v_cache],
            &[],
        );
        assert_close(
            &format!("attention (past {past}, {t} tokens, window {window:?})"),
            &got,
            &want,
            1e-4,
        );
    }
}

/// `y = gelu_tanh(gate) * up`, with PyTorch's tanh approximation.
#[test]
fn test_gemma_gelu_mul_matches_the_tanh_approximation() {
    let n = 2500;
    let gate: Vec<f32> = values(n, 5).iter().map(|v| v * 4.0).collect();
    let up = values(n, 6);
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&x, &u)| {
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh()) * u
        })
        .collect();
    let g = OneOp::new(&[n]);
    let u = g.input(&[n]);
    let got = g.run(GemmaGeluMulOp::<f32>::default(), &[&u], &[&gate, &up], &[]);
    assert_close("gelu_mul", &got, &want, 1e-5);
}

/// `y = x / sqrt(mean(x^2) + 1e-6) * (1 + w)` over rows inside a wider tensor, as the
/// per-head norms use it.
#[test]
fn test_gemma_rmsnorm_scales_by_one_plus_w() {
    let (t, heads, r) = (3, 2, 48);
    let x = values(t * heads * r, 7);
    let w = values(r, 8);
    let mut want = vec![0.0; x.len()];
    for (row, out) in x.chunks(r).zip(want.chunks_mut(r)) {
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / r as f32 + 1e-6).sqrt();
        for i in 0..r {
            out[i] = row[i] / rms * (1.0 + w[i]);
        }
    }
    let g = OneOp::new(&[t, heads * r]);
    let got = g.run(GemmaRmsNormOp::<f32>::new(r), &[], &[&x], &[&w]);
    assert_close("rmsnorm", &got, &want, 1e-5);
}

/// `y[t, :] = w[ids[t], :] * sqrt(hidden)`, with the ids fed as f32.
#[test]
fn test_gemma_embed_scales_the_looked_up_rows() {
    let (vocab, hidden) = (50, 1100);
    let ids = [7.0f32, 0.0, 49.0, 7.0];
    let w = values(vocab * hidden, 9);
    let scale = (hidden as f32).sqrt();
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            w[id as usize * hidden..][..hidden]
                .iter()
                .map(move |v| v * scale)
        })
        .collect();
    let g = OneOp::new(&[ids.len()]);
    let got = g.run(GemmaEmbedOp::<f32>::new(vocab, hidden), &[], &[&ids], &[&w]);
    assert_close("embed", &got, &want, 1e-5);
}
