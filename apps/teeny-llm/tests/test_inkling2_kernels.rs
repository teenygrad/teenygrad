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

//! Inkling-2's own kernels, each as a one-op graph compiled through teenyc, run on the
//! device (CUDA, or RISC-V under qemu) and compared with a host reference of the
//! formula in the kernel's module docs. Needs no model weights.

#![cfg(all(
    feature = "inkling2",
    any(feature = "cuda", feature = "qemu", feature = "riscv")
))]
use std::cell::RefCell;
use std::rc::Rc;

use teeny_core::graph::{CustomData, CustomOp, DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_kernels::graph::TritonLowering;
use teeny_llm::models::inkling2::kernels::{
    attention::{AttentionSpec, PastLen, SinkAttentionOp},
    block_attn::BlockAttnOp,
    cat2::Cat2Op,
    dyn_sconv::DynSconvOp,
    expert_gather::ExpertGatherOp,
    expert_gather_mxfp8::ExpertGatherMxfp8Op,
    expert_gather_nvfp4::ExpertGatherNvfp4Op,
    gated_mul::GatedMulOp,
    l2_clip::{L2ClipOp, ScaledL2ClipOp},
    mix::MessageMixOp,
    norm::{RmsNormOp, UnweightedRmsNormOp},
    nvfp4::{Nvfp4DequantOp, UnpackByteOp},
    rel_bias::RelBiasOp,
    router::TopKRouterOp,
    scatter_add::ExpertScatterAddOp,
    sconv::SconvOp,
    softcap::SoftcapOp,
    swiglu::SwigluOp,
};
use teeny_llm::models::inkling2::weights::{Checkpoint, pack_rows};
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

/// `y[t, c] = sum(k) w[c, k] * x[t + k, c]`, a causal depthwise conv of width 4.
///
/// The input carries its own history: `T + K - 1` rows in, `T` rows out.
#[test]
fn test_sconv_matches_a_host_convolution() {
    const K: usize = 4;
    let (t, c) = (5usize, 300usize);
    let x = values((t + K - 1) * c, 11);
    let w = values(c * K, 12);
    let want: Vec<f32> = (0..t * c)
        .map(|i| {
            let (row, ch) = (i / c, i % c);
            let conv: f32 = (0..K).map(|k| w[ch * K + k] * x[(row + k) * c + ch]).sum();
            // The convolution carries its own residual and normalisation.
            (conv + x[(row + K - 1) * c + ch]) / ((K + 1) as f32).sqrt()
        })
        .collect();
    let (hist, cur) = x.split_at((K - 1) * c);
    let g = OneOp::new(&[K - 1, c]);
    let x_in = g.input(&[t, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[&x_in], &[hist, cur], &[&w]);
    assert_close("sconv", &got, &want, 1e-5);
}

/// One token with no history is the filter's last tap alone, which is what decoding
/// does at the very start of a sequence.
#[test]
fn test_sconv_with_zero_history_keeps_only_the_last_tap() {
    const K: usize = 4;
    let c = 128usize;
    let mut x = vec![0.0f32; K * c];
    let row = values(c, 13);
    x[(K - 1) * c..].copy_from_slice(&row); // the only non-zero token
    let w = values(c * K, 14);
    let want: Vec<f32> = (0..c)
        .map(|ch| (w[ch * K + (K - 1)] * row[ch] + row[ch]) / ((K + 1) as f32).sqrt())
        .collect();
    let (hist, cur) = x.split_at((K - 1) * c);
    let g = OneOp::new(&[K - 1, c]);
    let x_in = g.input(&[1, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[&x_in], &[hist, cur], &[&w]);
    assert_close("sconv first token", &got, &want, 1e-5);
}

/// A channel count that is not a multiple of the block, so the tail is masked.
#[test]
fn test_sconv_masks_a_ragged_channel_tail() {
    const K: usize = 4;
    let (t, c) = (2usize, 257usize); // BLOCK is 256
    let x = values((t + K - 1) * c, 15);
    let w = values(c * K, 16);
    let want: Vec<f32> = (0..t * c)
        .map(|i| {
            let (row, ch) = (i / c, i % c);
            let conv: f32 = (0..K).map(|k| w[ch * K + k] * x[(row + k) * c + ch]).sum();
            (conv + x[(row + K - 1) * c + ch]) / ((K + 1) as f32).sqrt()
        })
        .collect();
    let (hist, cur) = x.split_at((K - 1) * c);
    let g = OneOp::new(&[K - 1, c]);
    let x_in = g.input(&[t, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[&x_in], &[hist, cur], &[&w]);
    assert_close("sconv ragged", &got, &want, 1e-5);
}

/// `y = tanh(x / cap) * cap`, at each of the three caps the model ships with.
#[test]
fn test_softcap_matches_a_host_tanh() {
    let n = 1500usize;
    // Scaled well past each cap, so the saturating part of tanh is exercised.
    let x: Vec<f32> = values(n, 21).iter().map(|v| v * 60.0).collect();
    for cap in [4.0f32, 25.0, 30.0] {
        let want: Vec<f32> = x.iter().map(|v| (v / cap).tanh() * cap).collect();
        let g = OneOp::new(&[n]);
        let got = g.run(SoftcapOp::<f32>::new(cap), &[], &[&x], &[]);
        assert_close(&format!("softcap {cap}"), &got, &want, 1e-5);
    }
}

/// Far outside the cap the output is the cap itself, with the sign of the input.
#[test]
fn test_softcap_saturates_at_the_cap() {
    let x = vec![-1.0e4f32, -500.0, 0.0, 500.0, 1.0e4];
    let cap = 4.0f32;
    let want: Vec<f32> = x.iter().map(|v| (v / cap).tanh() * cap).collect();
    let g = OneOp::new(&[x.len()]);
    let got = g.run(SoftcapOp::<f32>::new(cap), &[], &[&x], &[]);
    assert_close("softcap saturation", &got, &want, 1e-5);
    assert!(
        (got[0] + cap).abs() < 1e-4,
        "large negative should sit at -cap: {}",
        got[0]
    );
    assert!(
        (got[4] - cap).abs() < 1e-4,
        "large positive should sit at +cap: {}",
        got[4]
    );
    assert_eq!(got[2], 0.0, "zero should stay zero");
}

/// `v = x * scale[head]; y = v * min(threshold / (|v| + 1e-6), 1)`.
fn l2_clip_reference(x: &[f32], scale: &[f32], heads: usize, hd: usize, thr: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for row in 0..x.len() / (heads * hd) {
        for h in 0..heads {
            let base = row * heads * hd + h * hd;
            let v: Vec<f32> = (0..hd).map(|i| x[base + i] * scale[h * hd + i]).collect();
            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            let ratio = (thr / (norm + 1e-6)).min(1.0);
            for i in 0..hd {
                y[base + i] = v[i] * ratio;
            }
        }
    }
    y
}

#[test]
fn test_scaled_l2_clip_matches_the_reference() {
    let (t, heads, hd) = (4usize, 6usize, 128usize);
    let thr = 22.627_417f32; // the model's qk_l2_clip_threshold
    // Big enough that some head vectors clip and others do not.
    let x: Vec<f32> = values(t * heads * hd, 22).iter().map(|v| v * 5.0).collect();
    let scale: Vec<f32> = values(heads * hd, 23).iter().map(|v| v + 1.0).collect();
    let want = l2_clip_reference(&x, &scale, heads, hd, thr);
    let g = OneOp::new(&[t, heads * hd]);
    let got = g.run(
        ScaledL2ClipOp::<f32>::new(heads, hd, thr),
        &[],
        &[&x],
        &[&scale],
    );
    assert_close("scaled l2 clip", &got, &want, 1e-4);
}

/// A short vector keeps the scale applied to it and is NOT stretched up to the
/// threshold: that is what clamping the ratio at 1 means.
#[test]
fn test_scaled_l2_clip_leaves_short_vectors_alone() {
    let (heads, hd) = (2usize, 64usize);
    let thr = 100.0f32; // far longer than anything here
    let x: Vec<f32> = values(heads * hd, 24).iter().map(|v| v * 0.01).collect();
    let scale = vec![1.0f32; heads * hd];
    let g = OneOp::new(&[1, heads * hd]);
    let got = g.run(
        ScaledL2ClipOp::<f32>::new(heads, hd, thr),
        &[],
        &[&x],
        &[&scale],
    );
    assert_close("l2 clip passthrough", &got, &x, 1e-6);
}

/// A long vector comes out at exactly the threshold.
#[test]
fn test_scaled_l2_clip_shortens_to_the_threshold() {
    let (heads, hd) = (1usize, 32usize);
    let thr = 3.0f32;
    let x = vec![1.0f32; hd]; // norm sqrt(32) = 5.66, well over
    let scale = vec![1.0f32; hd];
    let g = OneOp::new(&[1, hd]);
    let got = g.run(
        ScaledL2ClipOp::<f32>::new(heads, hd, thr),
        &[],
        &[&x],
        &[&scale],
    );
    let norm = got.iter().map(|a| a * a).sum::<f32>().sqrt();
    assert!(
        (norm - thr).abs() < 1e-3,
        "clipped norm {norm} should be {thr}"
    );
}

/// A head_dim that is not a power of two, so the block is padded and masked.
#[test]
fn test_scaled_l2_clip_masks_a_ragged_head() {
    let (t, heads, hd) = (2usize, 3usize, 100usize);
    let thr = 8.0f32; // the model's rel_l2_clip_threshold
    let x: Vec<f32> = values(t * heads * hd, 25).iter().map(|v| v * 3.0).collect();
    let scale: Vec<f32> = values(heads * hd, 26).iter().map(|v| v + 1.0).collect();
    let want = l2_clip_reference(&x, &scale, heads, hd, thr);
    let g = OneOp::new(&[t, heads * hd]);
    let got = g.run(
        ScaledL2ClipOp::<f32>::new(heads, hd, thr),
        &[],
        &[&x],
        &[&scale],
    );
    assert_close("l2 clip ragged", &got, &want, 1e-4);
}

/// `bias[t, h, j] = logits[t, h, (past + t) - j]` inside `0 <= dist < extent`, else 0.
fn rel_bias_reference(
    logits: &[f32],
    t: usize,
    heads: usize,
    extent: usize,
    past: usize,
    kv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; t * heads * kv];
    for tok in 0..t {
        for h in 0..heads {
            let row = tok * heads + h;
            for j in 0..kv {
                let dist = (past + tok) as isize - j as isize;
                if dist >= 0 && (dist as usize) < extent {
                    out[row * kv + j] = logits[row * extent + dist as usize];
                }
            }
        }
    }
    out
}

#[test]
fn test_rel_bias_gathers_by_backward_distance() {
    let (t, heads, extent, past) = (4usize, 3usize, 64usize, 10usize);
    let kv = past + t;
    let logits = values(t * heads * extent, 31);
    let want = rel_bias_reference(&logits, t, heads, extent, past, kv);
    let g = OneOp::new(&[t * heads, extent]);
    let got = g.run(
        RelBiasOp::<f32>::new(heads, extent, {
            let p = PastLen::default();
            p.set(past);
            p
        }),
        &[],
        &[&logits],
        &[],
    );
    assert_close("rel bias", &got, &want, 0.0);
}

/// Beyond `rel_extent` the bias is exactly zero, which is what bounds the scheme.
#[test]
fn test_rel_bias_is_zero_past_the_extent() {
    let (t, heads, extent, past) = (2usize, 1usize, 8usize, 40usize);
    // kv is now derived from PastLen at launch rather than passed in.
    let logits: Vec<f32> = (0..t * heads * extent).map(|i| 1.0 + i as f32).collect();
    let g = OneOp::new(&[t * heads, extent]);
    let got = g.run(
        RelBiasOp::<f32>::new(heads, extent, {
            let p = PastLen::default();
            p.set(past);
            p
        }),
        &[],
        &[&logits],
        &[],
    );
    // Query 0 sits at absolute position 40, so keys 0..=32 are further back than 8.
    for (j, &b) in got.iter().enumerate().take(past - extent + 1) {
        assert_eq!(b, 0.0, "key {j} is beyond the extent and must be zero");
    }
    // The nearest key is distance 0, the first entry of the profile.
    assert_eq!(
        got[past], logits[0],
        "distance 0 should read the profile head"
    );
}

/// Keys after the query give a negative distance and must be zero, not wrap round.
#[test]
fn test_rel_bias_zeroes_future_keys() {
    let (t, heads, extent, past) = (3usize, 2usize, 32usize, 0usize);
    let kv = t; // no cache, so queries and keys are the same positions
    let logits = values(t * heads * extent, 32);
    let want = rel_bias_reference(&logits, t, heads, extent, past, kv);
    let g = OneOp::new(&[t * heads, extent]);
    let got = g.run(
        RelBiasOp::<f32>::new(heads, extent, {
            let p = PastLen::default();
            p.set(past);
            p
        }),
        &[],
        &[&logits],
        &[],
    );
    assert_close("rel bias future", &got, &want, 0.0);
    // Query 0, head 0 attends to keys 1 and 2, both in the future.
    assert_eq!(got[1], 0.0);
    assert_eq!(got[2], 0.0);
}

/// A kv length that is not a multiple of the block, so the tail is masked.
#[test]
fn test_rel_bias_masks_a_ragged_kv_tail() {
    let (t, heads, extent, past) = (2usize, 2usize, 1024usize, 255usize);
    let kv = past + t; // 257, against a block of 256
    let logits = values(t * heads * extent, 33);
    let want = rel_bias_reference(&logits, t, heads, extent, past, kv);
    let g = OneOp::new(&[t * heads, extent]);
    let got = g.run(
        RelBiasOp::<f32>::new(heads, extent, {
            let p = PastLen::default();
            p.set(past);
            p
        }),
        &[],
        &[&logits],
        &[],
    );
    assert_close("rel bias ragged", &got, &want, 0.0);
}

/// The router, as `Inkling2LoopTopKRouter` does it: the bias steers the choice, the
/// weights come from the unbiased scores.
fn router_reference(
    logits: &[f32],
    bias: &[f32],
    t: usize,
    e: usize,
    k: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (mut w_out, mut i_out) = (vec![0.0f32; t * k], vec![0.0f32; t * k]);
    for row in 0..t {
        let scores: Vec<f32> = (0..e)
            .map(|j| 1.0 / (1.0 + (-logits[row * e + j]).exp()))
            .collect();
        let mut pool: Vec<f32> = (0..e).map(|j| scores[j] + bias[j]).collect();
        let mut picked = Vec::new();
        for _ in 0..k {
            let mut best = 0usize;
            for j in 1..e {
                if pool[j] > pool[best] {
                    best = j;
                }
            }
            picked.push(best);
            pool[best] = f32::NEG_INFINITY;
        }
        let total: f32 = picked.iter().map(|&j| scores[j]).sum();
        for (slot, &j) in picked.iter().enumerate() {
            w_out[row * k + slot] = scores[j] / (total + 1e-20) * scale;
            i_out[row * k + slot] = j as f32;
        }
    }
    (w_out, i_out)
}

#[test]
fn test_router_picks_and_weights_like_the_reference() {
    let (t, e, k) = (5usize, 128usize, 8usize); // the model's own shape
    let scale = 8.0f32; // route_scale
    let logits: Vec<f32> = values(t * e, 41).iter().map(|v| v * 4.0).collect();
    let bias: Vec<f32> = values(e, 42).iter().map(|v| v * 0.1).collect();
    let (want_w, want_i) = router_reference(&logits, &bias, t, e, k, scale);
    let g = OneOp::new(&[t, e]);
    let got = g.run(
        TopKRouterOp::<f32>::new(e, k, scale),
        &[],
        &[&logits],
        &[&bias],
    );
    for row in 0..t {
        let w = &got[row * 2 * k..row * 2 * k + k];
        let i = &got[row * 2 * k + k..(row + 1) * 2 * k];
        assert_close(
            &format!("router weights row {row}"),
            w,
            &want_w[row * k..(row + 1) * k],
            1e-5,
        );
        assert_eq!(i, &want_i[row * k..(row + 1) * k], "router ids row {row}");
    }
}

/// The weights of each token sum to `route_scale`, since they are normalised first.
#[test]
fn test_router_weights_sum_to_the_scale() {
    let (t, e, k) = (3usize, 64usize, 4usize);
    let scale = 8.0f32;
    let logits: Vec<f32> = values(t * e, 43).iter().map(|v| v * 3.0).collect();
    let bias = vec![0.0f32; e];
    let g = OneOp::new(&[t, e]);
    let got = g.run(
        TopKRouterOp::<f32>::new(e, k, scale),
        &[],
        &[&logits],
        &[&bias],
    );
    for row in 0..t {
        let sum: f32 = got[row * 2 * k..row * 2 * k + k].iter().sum();
        assert!(
            (sum - scale).abs() < 1e-3,
            "row {row} weights sum to {sum}, not {scale}"
        );
    }
}

/// A bias large enough to force one expert in must change the CHOICE but leave the
/// chosen expert's weight at its unbiased score. This is the detail that would be
/// invisible if the weights were gathered from the biased scores instead.
#[test]
fn test_router_bias_steers_the_choice_but_not_the_weight() {
    let (t, e, k) = (1usize, 32usize, 2usize);
    let scale = 1.0f32;
    let logits: Vec<f32> = (0..e).map(|j| 3.0 - 0.1 * j as f32).collect(); // expert 0 best
    let mut bias = vec![0.0f32; e];
    bias[e - 1] = 10.0; // force the worst expert to be picked first
    let g = OneOp::new(&[t, e]);
    let got = g.run(
        TopKRouterOp::<f32>::new(e, k, scale),
        &[],
        &[&logits],
        &[&bias],
    );
    assert_eq!(
        got[k] as usize,
        e - 1,
        "the biased expert should be chosen first"
    );
    let (want_w, _) = router_reference(&logits, &bias, t, e, k, scale);
    assert_close("biased router weights", &got[..k], &want_w, 1e-5);
    // Its weight is its own sigmoid score, nowhere near the 10.0 bias.
    let s_last = 1.0 / (1.0 + (-logits[e - 1]).exp());
    let s_first = 1.0 / (1.0 + (-logits[0]).exp());
    assert!(
        (got[0] - s_last / (s_last + s_first)).abs() < 1e-5,
        "weight came from the biased score"
    );
}

/// `y[s, n] = sum_k x[token(s), k] * W[expert(s), n, k]`.
#[test]
fn test_expert_gather_uses_the_routed_weight_matrix() {
    let (t, e, n, k, top_k) = (3usize, 6usize, 12usize, 40usize, 2usize);
    let x = values(t * k, 51);
    let w = values(e * n * k, 52);
    // A router row: top_k weights (unused here) then top_k ids, as floats.
    let ids = [1usize, 4, 0, 5, 3, 2];
    let mut packed = vec![0.0f32; t * 2 * top_k];
    for tok in 0..t {
        for slot in 0..top_k {
            packed[tok * 2 * top_k + top_k + slot] = ids[tok * top_k + slot] as f32;
        }
    }
    let want: Vec<f32> = (0..t * top_k * n)
        .map(|i| {
            let (slot, col) = (i / n, i % n);
            let (tok, which) = (slot / top_k, slot % top_k);
            let ex = ids[tok * top_k + which];
            (0..k)
                .map(|j| x[tok * k + j] * w[(ex * n + col) * k + j])
                .sum()
        })
        .collect();
    let g = OneOp::new(&[t, k]);
    let ids_in = g.input(&[t, 2 * top_k]);
    let got = g.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_in],
        &[&x, &packed],
        &[&w],
    );
    assert_close("expert gather", &got, &want, 1e-4);
}

/// Two slots of the same token pointing at the same expert must give the same row:
/// the gather reads the parameter, it does not consume it.
#[test]
fn test_expert_gather_repeats_an_expert_consistently() {
    let (t, e, n, k, top_k) = (1usize, 4usize, 8usize, 32usize, 2usize);
    let x = values(t * k, 53);
    let w = values(e * n * k, 54);
    let mut packed = vec![0.0f32; t * 2 * top_k];
    packed[top_k] = 2.0; // both slots choose expert 2
    packed[top_k + 1] = 2.0;
    let g = OneOp::new(&[t, k]);
    let ids_in = g.input(&[t, 2 * top_k]);
    let got = g.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_in],
        &[&x, &packed],
        &[&w],
    );
    assert_close("repeated expert", &got[n..], &got[..n], 0.0);
}

/// A reduction width that is not a multiple of the block, so the tail is masked.
#[test]
fn test_expert_gather_masks_a_ragged_reduction() {
    let (t, e, n, k, top_k) = (2usize, 3usize, 5usize, 300usize, 1usize); // BLOCK is 256
    let x = values(t * k, 55);
    let w = values(e * n * k, 56);
    let ids = [2usize, 0];
    let mut packed = vec![0.0f32; t * 2 * top_k];
    for tok in 0..t {
        packed[tok * 2 * top_k + top_k] = ids[tok] as f32;
    }
    let want: Vec<f32> = (0..t * n)
        .map(|i| {
            let (tok, col) = (i / n, i % n);
            (0..k)
                .map(|j| x[tok * k + j] * w[(ids[tok] * n + col) * k + j])
                .sum()
        })
        .collect();
    let g = OneOp::new(&[t, k]);
    let ids_in = g.input(&[t, 2 * top_k]);
    let got = g.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_in],
        &[&x, &packed],
        &[&w],
    );
    assert_close("expert gather ragged", &got, &want, 1e-4);
}

/// `y = softcap(gate) * sigmoid(gate) * softcap(up)`, gate first in the row.
fn swiglu_reference(x: &[f32], s: usize, i: usize, gcap: f32, ucap: f32) -> Vec<f32> {
    let cap = |v: f32, c: f32| (v / c).tanh() * c;
    (0..s * i)
        .map(|idx| {
            let (row, col) = (idx / i, idx % i);
            // gate and up are interleaved, as `w13_weight` stores them
            let g = x[row * 2 * i + col * 2];
            let u = x[row * 2 * i + col * 2 + 1];
            cap(g, gcap) * (1.0 / (1.0 + (-g).exp())) * cap(u, ucap)
        })
        .collect()
}

#[test]
fn test_swiglu_matches_the_reference() {
    let (s, i) = (4usize, 300usize); // not a multiple of the 256 block
    let (gcap, ucap) = (4.0f32, 25.0f32); // the model's caps
    // Well outside both caps, so the saturating region is exercised.
    let x: Vec<f32> = values(s * 2 * i, 61).iter().map(|v| v * 40.0).collect();
    let want = swiglu_reference(&x, s, i, gcap, ucap);
    let g = OneOp::new(&[s, 2 * i]);
    let got = g.run(SwigluOp::<f32>::new(i, gcap, ucap), &[], &[&x], &[]);
    assert_close("swiglu", &got, &want, 1e-4);
}

/// The sigmoid sees the RAW gate, not the capped one. With a gate far past its cap
/// the two differ sharply: sigmoid(40) is ~1 while sigmoid(softcap(40, 4)) is ~0.98.
#[test]
fn test_swiglu_sigmoid_sees_the_uncapped_gate() {
    let (s, i) = (1usize, 1usize);
    let (gcap, ucap) = (4.0f32, 25.0f32);
    let x = vec![40.0f32, 1.0]; // gate 40, up 1 -- interleaved, so adjacent
    let g = OneOp::new(&[s, 2 * i]);
    let got = g.run(SwigluOp::<f32>::new(i, gcap, ucap), &[], &[&x], &[]);
    let cap = |v: f32, c: f32| (v / c).tanh() * c;
    let right = cap(40.0, gcap) * (1.0 / (1.0 + (-40.0f32).exp())) * cap(1.0, ucap);
    let wrong = cap(40.0, gcap) * (1.0 / (1.0 + (-cap(40.0, gcap)).exp())) * cap(1.0, ucap);
    assert!(
        (got[0] - right).abs() < 1e-4,
        "got {}, want {right}",
        got[0]
    );
    assert!(
        (right - wrong).abs() > 0.05,
        "the test is only meaningful if the two readings differ"
    );
}

/// `out[t, n] = sum(slot) weight[t, slot] * y[t * top_k + slot, n]`.
#[test]
fn test_scatter_add_combines_the_expert_outputs() {
    let (t, top_k, n) = (3usize, 4usize, 300usize);
    let y = values(t * top_k * n, 62);
    let mut packed = vec![0.0f32; t * 2 * top_k];
    let w = values(t * top_k, 63);
    for tok in 0..t {
        for slot in 0..top_k {
            packed[tok * 2 * top_k + slot] = w[tok * top_k + slot];
        }
    }
    let want: Vec<f32> = (0..t * n)
        .map(|idx| {
            let (tok, col) = (idx / n, idx % n);
            (0..top_k)
                .map(|slot| w[tok * top_k + slot] * y[(tok * top_k + slot) * n + col])
                .sum()
        })
        .collect();
    let g = OneOp::new(&[t * top_k, n]);
    let win = g.input(&[t, 2 * top_k]);
    let got = g.run(
        ExpertScatterAddOp::<f32>::new(n, top_k),
        &[&win],
        &[&y, &packed],
        &[],
    );
    assert_close("scatter add", &got, &want, 1e-4);
}

/// Zero weights contribute nothing, so a token whose router row is zero comes out
/// zero however large its expert outputs are.
#[test]
fn test_scatter_add_zero_weights_contribute_nothing() {
    let (t, top_k, n) = (2usize, 2usize, 16usize);
    let y: Vec<f32> = values(t * top_k * n, 64)
        .iter()
        .map(|v| v * 1000.0)
        .collect();
    let mut packed = vec![0.0f32; t * 2 * top_k];
    // Token 0 has all-zero weights; token 1 puts everything on its second slot.
    packed[2 * top_k + 1] = 1.0;
    let g = OneOp::new(&[t * top_k, n]);
    let win = g.input(&[t, 2 * top_k]);
    let got = g.run(
        ExpertScatterAddOp::<f32>::new(n, top_k),
        &[&win],
        &[&y, &packed],
        &[],
    );
    assert_close("zero-weight token", &got[..n], &vec![0.0f32; n], 0.0);
    assert_close("single-slot token", &got[n..], &y[3 * n..4 * n], 1e-5);
}

/// The router and the two MoE kernels compose: route, gather, activate, combine.
#[test]
fn test_router_gather_swiglu_scatter_compose() {
    let (t, e, hidden, inter, top_k) = (2usize, 8usize, 24usize, 10usize, 3usize);
    let (gcap, ucap, scale) = (4.0f32, 25.0f32, 8.0f32);
    let x = values(t * hidden, 71);
    let bias = vec![0.0f32; e];
    let logits = values(t * e, 72);
    let w13 = values(e * 2 * inter * hidden, 73);
    let w2 = values(e * hidden * inter, 74);

    // 1. route
    let gr = OneOp::new(&[t, e]);
    let routed = gr.run(
        TopKRouterOp::<f32>::new(e, top_k, scale),
        &[],
        &[&logits],
        &[&bias],
    );
    // 2. gather the fused gate-and-up projection
    let gg = OneOp::new(&[t, hidden]);
    let ids_in = gg.input(&[t, 2 * top_k]);
    let gate_up = gg.run(
        ExpertGatherOp::<f32>::new(e, 2 * inter, hidden, top_k),
        &[&ids_in],
        &[&x, &routed],
        &[&w13],
    );
    // 3. activate
    let ga = OneOp::new(&[t * top_k, 2 * inter]);
    let act = ga.run(
        SwigluOp::<f32>::new(inter, gcap, ucap),
        &[],
        &[&gate_up],
        &[],
    );
    // 4. gather the down projection, one slot each this time
    let gd = OneOp::new(&[t * top_k, inter]);
    let id2 = gd.input(&[t * top_k, 2]);
    let flat_ids: Vec<f32> = (0..t * top_k)
        .flat_map(|s| [0.0, routed[(s / top_k) * 2 * top_k + top_k + s % top_k]])
        .collect();
    let down = gd.run(
        ExpertGatherOp::<f32>::new(e, hidden, inter, 1),
        &[&id2],
        &[&act, &flat_ids],
        &[&w2],
    );
    // 5. combine
    let gs = OneOp::new(&[t * top_k, hidden]);
    let win = gs.input(&[t, 2 * top_k]);
    let got = gs.run(
        ExpertScatterAddOp::<f32>::new(hidden, top_k),
        &[&win],
        &[&down, &routed],
        &[],
    );

    // Host reference of the whole block.
    let (rw, ri) = router_reference(&logits, &bias, t, e, top_k, scale);
    let cap = |v: f32, c: f32| (v / c).tanh() * c;
    let mut want = vec![0.0f32; t * hidden];
    for tok in 0..t {
        for slot in 0..top_k {
            let ex = ri[tok * top_k + slot] as usize;
            let gu: Vec<f32> = (0..2 * inter)
                .map(|r| {
                    (0..hidden)
                        .map(|c| x[tok * hidden + c] * w13[(ex * 2 * inter + r) * hidden + c])
                        .sum()
                })
                .collect();
            // gate and up are interleaved down the rows of w13, not two halves.
            let a: Vec<f32> = (0..inter)
                .map(|r| {
                    let (g, u) = (gu[2 * r], gu[2 * r + 1]);
                    cap(g, gcap) * (1.0 / (1.0 + (-g).exp())) * cap(u, ucap)
                })
                .collect();
            for (c, slot_out) in want[tok * hidden..(tok + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                let d: f32 = (0..inter)
                    .map(|r| a[r] * w2[(ex * hidden + c) * inter + r])
                    .sum();
                *slot_out += rw[tok * top_k + slot] * d;
            }
        }
    }
    assert_close("moe block", &got, &want, 2e-3);
}

/// Attention with a relative position bias and a valueless per-query sink.
#[allow(clippy::too_many_arguments)]
fn sink_attention_reference(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    kn: &[f32],
    vn: &[f32],
    bias: &[f32],
    sinks: &[f32],
    t: usize,
    h: usize,
    kvh: usize,
    hd: usize,
    past: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let kv = past + t;
    let scale = (hd as f32).powf(-0.5);
    let mut out = vec![0.0f32; t * h * hd];
    for tok in 0..t {
        let qpos = past + tok;
        for head in 0..h {
            let g = head / (h / kvh);
            let mut logits = Vec::new();
            let mut keys = Vec::new();
            for j in 0..=qpos {
                if window.is_some_and(|w| qpos - j >= w) {
                    continue;
                }
                let krow: Vec<f32> = (0..hd)
                    .map(|d| {
                        if j < past {
                            kc[j * kvh * hd + g * hd + d]
                        } else {
                            kn[(j - past) * kvh * hd + g * hd + d]
                        }
                    })
                    .collect();
                let dot: f32 = (0..hd)
                    .map(|d| q[(tok * h + head) * hd + d] * krow[d])
                    .sum();
                logits.push(dot * scale + bias[(tok * h + head) * kv + j]);
                keys.push(j);
            }
            // The sink joins the softmax but has no value vector.
            let sink = sinks[tok * h + head];
            let m = logits.iter().cloned().fold(sink, f32::max);
            let denom: f32 = logits.iter().map(|l| (l - m).exp()).sum::<f32>() + (sink - m).exp();
            for (idx, &j) in keys.iter().enumerate() {
                let p = (logits[idx] - m).exp() / denom;
                for d in 0..hd {
                    let v = if j < past {
                        vc[j * kvh * hd + g * hd + d]
                    } else {
                        vn[(j - past) * kvh * hd + g * hd + d]
                    };
                    out[(tok * h + head) * hd + d] += p * v;
                }
            }
        }
    }
    out
}

struct AttnCase {
    t: usize,
    h: usize,
    kvh: usize,
    hd: usize,
    past: usize,
    cache: usize,
    window: Option<usize>,
}

fn run_sink_attention(c: &AttnCase, sinks: &[f32], seed: u64) -> (Vec<f32>, Vec<f32>) {
    let kv = c.past + c.t;
    let q = values(c.t * c.h * c.hd, seed);
    let kn = values(c.t * c.kvh * c.hd, seed + 1);
    let vn = values(c.t * c.kvh * c.hd, seed + 2);
    let kc_full = values(c.cache * c.kvh * c.hd, seed + 3);
    let vc_full = values(c.cache * c.kvh * c.hd, seed + 4);
    let bias = values(c.t * c.h * kv, seed + 5);

    let want = sink_attention_reference(
        &q, &kc_full, &vc_full, &kn, &vn, &bias, sinks, c.t, c.h, c.kvh, c.hd, c.past, c.window,
    );

    let g = OneOp::new(&[c.t, c.h * c.hd]);
    let kn_in = g.input(&[c.t, c.kvh * c.hd]);
    let vn_in = g.input(&[c.t, c.kvh * c.hd]);
    let kc_in = g.input(&[c.cache, c.kvh * c.hd]);
    let vc_in = g.input(&[c.cache, c.kvh * c.hd]);
    let b_in = g.input(&[c.t * c.h, kv]);
    let s_in = g.input(&[c.t, c.h]);
    let past = PastLen::default();
    past.set(c.past);
    let spec = AttentionSpec {
        heads: c.h,
        kv_heads: c.kvh,
        head_dim: c.hd,
        window: c.window,
        scale: (c.hd as f32).powf(-0.5),
    };
    let got = g.run(
        SinkAttentionOp::<f32>::new(spec, past),
        &[&kn_in, &vn_in, &kc_in, &vc_in, &b_in, &s_in],
        &[&q, &kn, &vn, &kc_full, &vc_full, &bias, sinks],
        &[],
    );
    (got, want)
}

#[test]
fn test_sink_attention_matches_the_reference() {
    let c = AttnCase {
        t: 3,
        h: 4,
        kvh: 2,
        hd: 16,
        past: 5,
        cache: 16,
        window: None,
    };
    let sinks = values(c.t * c.h, 91);
    let (got, want) = run_sink_attention(&c, &sinks, 90);
    assert_close("sink attention", &got, &want, 1e-4);
}

#[test]
fn test_sink_attention_respects_the_sliding_window() {
    let c = AttnCase {
        t: 2,
        h: 2,
        kvh: 1,
        hd: 8,
        past: 9,
        cache: 16,
        window: Some(4),
    };
    let sinks = values(c.t * c.h, 93);
    let (got, want) = run_sink_attention(&c, &sinks, 92);
    assert_close("sink attention windowed", &got, &want, 1e-4);
}

/// A very negative sink takes almost no mass, so the output matches ordinary
/// attention; a very positive one takes nearly all of it and the output collapses
/// towards zero. This is the sink's whole purpose and it must be visible.
#[test]
fn test_sink_absorbs_probability_mass() {
    let c = AttnCase {
        t: 1,
        h: 1,
        kvh: 1,
        hd: 8,
        past: 3,
        cache: 8,
        window: None,
    };
    let (quiet, _) = run_sink_attention(&c, &[-60.0], 94);
    let (greedy, _) = run_sink_attention(&c, &[60.0], 94);
    let mag = |v: &[f32]| v.iter().map(|a| a.abs()).sum::<f32>();
    assert!(
        mag(&greedy) < mag(&quiet) * 0.01,
        "a dominant sink should leave almost nothing: {} vs {}",
        mag(&greedy),
        mag(&quiet)
    );
    // With the sink out of the way the weights sum to one, so the result is a convex
    // combination of the values and cannot be tiny.
    assert!(
        mag(&quiet) > 1e-3,
        "a negligible sink should leave ordinary attention"
    );
}

/// The bias actually reaches the logits: changing one entry changes the output.
#[test]
fn test_bias_changes_the_attention() {
    let c = AttnCase {
        t: 1,
        h: 1,
        kvh: 1,
        hd: 8,
        past: 3,
        cache: 8,
        window: None,
    };
    let sinks = vec![-60.0f32];
    let kv = c.past + c.t;
    let (base, _) = run_sink_attention(&c, &sinks, 95);
    // Same seed, but the reference is recomputed with a biased key, so a mismatch
    // here would mean the kernel ignored the bias.
    let mut bias = values(c.t * c.h * kv, 95 + 5);
    bias[0] += 20.0; // make key 0 dominate
    let q = values(c.t * c.h * c.hd, 95);
    let kn = values(c.t * c.kvh * c.hd, 96);
    let vn = values(c.t * c.kvh * c.hd, 97);
    let kc = values(c.cache * c.kvh * c.hd, 98);
    let vc = values(c.cache * c.kvh * c.hd, 99);
    let want = sink_attention_reference(
        &q, &kc, &vc, &kn, &vn, &bias, &sinks, c.t, c.h, c.kvh, c.hd, c.past, None,
    );
    let g = OneOp::new(&[c.t, c.h * c.hd]);
    let kn_in = g.input(&[c.t, c.kvh * c.hd]);
    let vn_in = g.input(&[c.t, c.kvh * c.hd]);
    let kc_in = g.input(&[c.cache, c.kvh * c.hd]);
    let vc_in = g.input(&[c.cache, c.kvh * c.hd]);
    let b_in = g.input(&[c.t * c.h, kv]);
    let s_in = g.input(&[c.t, c.h]);
    let past = PastLen::default();
    past.set(c.past);
    let spec = AttentionSpec {
        heads: c.h,
        kv_heads: c.kvh,
        head_dim: c.hd,
        window: None,
        scale: (c.hd as f32).powf(-0.5),
    };
    let got = g.run(
        SinkAttentionOp::<f32>::new(spec, past),
        &[&kn_in, &vn_in, &kc_in, &vc_in, &b_in, &s_in],
        &[&q, &kn, &vn, &kc, &vc, &bias, &sinks],
        &[],
    );
    assert_close("biased attention", &got, &want, 1e-4);
    // With key 0 strongly favoured the result must differ from the unbiased run.
    let diff: f32 = got.iter().zip(&base).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-3,
        "the bias did not change the output (diff {diff})"
    );
}

fn rms(row: &[f32]) -> f32 {
    (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + 1e-6).sqrt()
}

/// `y = w * x / sqrt(mean(x^2) + eps)`. NOT Gemma's `1 + w`.
#[test]
fn test_rmsnorm_applies_the_weight_directly() {
    let (rows, r) = (3usize, 100usize);
    let x = values(rows * r, 101);
    let w: Vec<f32> = values(r, 102).iter().map(|v| v + 1.0).collect();
    let want: Vec<f32> = (0..rows * r)
        .map(|i| {
            let (row, col) = (i / r, i % r);
            x[i] / rms(&x[row * r..(row + 1) * r]) * w[col]
        })
        .collect();
    let g = OneOp::new(&[rows, r]);
    let got = g.run(RmsNormOp::<f32>::new(r), &[], &[&x], &[&w]);
    assert_close("rmsnorm", &got, &want, 1e-4);
    // An all-ones weight must be a no-op on the scale, which `1 + w` would double.
    let ones = vec![1.0f32; r];
    let g2 = OneOp::new(&[rows, r]);
    let plain = g2.run(RmsNormOp::<f32>::new(r), &[], &[&x], &[&ones]);
    let expect: Vec<f32> = (0..rows * r)
        .map(|i| x[i] / rms(&x[(i / r) * r..(i / r + 1) * r]))
        .collect();
    assert_close("rmsnorm with unit weight", &plain, &expect, 1e-4);
}

#[test]
fn test_unweighted_rmsnorm_normalises_without_a_scale() {
    let (rows, r) = (4usize, 64usize);
    let x = values(rows * r, 103);
    let want: Vec<f32> = (0..rows * r)
        .map(|i| x[i] / rms(&x[(i / r) * r..(i / r + 1) * r]))
        .collect();
    let g = OneOp::new(&[rows, r]);
    let got = g.run(UnweightedRmsNormOp::<f32>::new(r), &[], &[&x], &[]);
    assert_close("unweighted rmsnorm", &got, &want, 1e-4);
}

/// The key projection's clip: same as the scaled one with an all-ones scale.
#[test]
fn test_unscaled_l2_clip_matches_the_scaled_one_with_unit_scale() {
    let (rows, hd) = (6usize, 64usize);
    let thr = 22.627_417f32;
    let x: Vec<f32> = values(rows * hd, 104).iter().map(|v| v * 6.0).collect();
    let ones = vec![1.0f32; hd];
    let want = l2_clip_reference(&x, &ones, 1, hd, thr);
    let g = OneOp::new(&[rows, hd]);
    let got = g.run(L2ClipOp::<f32>::new(hd, thr), &[], &[&x], &[]);
    assert_close("unscaled l2 clip", &got, &want, 1e-4);
}

/// `y = x * sigmoid(g)`, the output projection's gate.
#[test]
fn test_gated_mul_applies_a_sigmoid_gate() {
    let n = 500usize;
    let x = values(n, 105);
    let gate: Vec<f32> = values(n, 106).iter().map(|v| v * 10.0).collect();
    let want: Vec<f32> = (0..n).map(|i| x[i] / (1.0 + (-gate[i]).exp())).collect();
    let g = OneOp::new(&[n]);
    let gin = g.input(&[n]);
    let got = g.run(GatedMulOp::<f32>::default(), &[&gin], &[&x, &gate], &[]);
    assert_close("gated mul", &got, &want, 1e-5);
}

/// `[T, A]` and `[T, B]` to `[T, A + B]`.
#[test]
fn test_cat2_joins_two_rows() {
    let (t, a, b) = (3usize, 300usize, 140usize); // a crosses the 256 block
    let xa = values(t * a, 111);
    let xb = values(t * b, 112);
    let mut want = Vec::with_capacity(t * (a + b));
    for row in 0..t {
        want.extend_from_slice(&xa[row * a..(row + 1) * a]);
        want.extend_from_slice(&xb[row * b..(row + 1) * b]);
    }
    let g = OneOp::new(&[t, a]);
    let bin = g.input(&[t, b]);
    let got = g.run(Cat2Op::<f32>::new(a, b), &[&bin], &[&xa, &xb], &[]);
    assert_close("cat2", &got, &want, 0.0);
}

/// `out = (1 - forget) * hidden + gain * message`.
#[test]
fn test_message_mix_combines_with_both_gates() {
    let n = 500usize;
    let h = values(n, 113);
    let msg = values(n, 114);
    let f = values(n, 115);
    let gn = values(n, 116);
    let want: Vec<f32> = (0..n)
        .map(|i| (1.0 - f[i]) * h[i] + gn[i] * msg[i])
        .collect();
    let g = OneOp::new(&[n]);
    let (m_in, f_in, g_in) = (g.input(&[n]), g.input(&[n]), g.input(&[n]));
    let got = g.run(
        MessageMixOp::<f32>::default(),
        &[&m_in, &f_in, &g_in],
        &[&h, &msg, &f, &gn],
        &[],
    );
    assert_close("message mix", &got, &want, 1e-5);
}

/// A forget gate of 1 and a gain of 0 discards the hidden state entirely; the
/// reverse keeps it untouched. These are the ends of the gate's range.
#[test]
fn test_message_mix_gates_reach_both_extremes() {
    let n = 32usize;
    let h = values(n, 117);
    let msg = values(n, 118);
    let ones = vec![1.0f32; n];
    let zeros = vec![0.0f32; n];
    let g = OneOp::new(&[n]);
    let (m_in, f_in, g_in) = (g.input(&[n]), g.input(&[n]), g.input(&[n]));
    let forget_all = g.run(
        MessageMixOp::<f32>::default(),
        &[&m_in, &f_in, &g_in],
        &[&h, &msg, &ones, &zeros],
        &[],
    );
    assert_close("forget everything", &forget_all, &vec![0.0f32; n], 0.0);
    let g2 = OneOp::new(&[n]);
    let (m2, f2, g2i) = (g2.input(&[n]), g2.input(&[n]), g2.input(&[n]));
    let keep = g2.run(
        MessageMixOp::<f32>::default(),
        &[&m2, &f2, &g2i],
        &[&h, &msg, &zeros, &zeros],
        &[],
    );
    assert_close("keep everything", &keep, &h, 0.0);
}

/// `out[t, q] = sum_s softmax_s(query[q] . states[t, s] * hidden^-0.5) states[t, s]`.
#[test]
fn test_block_attn_matches_a_host_reference() {
    // Three committed states plus a partial makes the stack of four the reference
    // builds with cat([block_states, partial]).
    let (t, st, hid, q) = (3usize, 3usize, 64usize, 3usize);
    let states = values(t * st * hid, 121);
    let partial = values(t * hid, 125);
    let query = values(q * hid, 122);
    let scale = (hid as f32).powf(-0.5);
    let mut want = vec![0.0f32; t * q * hid];
    for tok in 0..t {
        for qi in 0..q {
            let at = |si: usize, d: usize| {
                if si < st {
                    states[(tok * st + si) * hid + d]
                } else {
                    partial[tok * hid + d]
                }
            };
            let logits: Vec<f32> = (0..=st)
                .map(|si| {
                    (0..hid)
                        .map(|d| query[qi * hid + d] * at(si, d))
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let den: f32 = logits.iter().map(|l| (l - m).exp()).sum();
            for (si, logit) in logits.iter().enumerate() {
                let p = (logit - m).exp() / den;
                for d in 0..hid {
                    want[(tok * q + qi) * hid + d] += p * at(si, d);
                }
            }
        }
    }
    // One op per query: each reads its own row of the shared [Q, hidden] parameter.
    let got: Vec<Vec<f32>> = (0..q)
        .map(|qi| {
            let gi = OneOp::new(&[t * st, hid]);
            let pin = gi.input(&[t, hid]);
            gi.run(
                BlockAttnOp::<f32>::new(st, hid, q, qi),
                &[&pin],
                &[&states, &partial],
                &[&query],
            )
        })
        .collect();
    // `want` is token-major; the ops are query-major, so compare per (token, query).
    for tok in 0..t {
        for qi in 0..q {
            assert_close(
                "block attn",
                &got[qi][tok * hid..(tok + 1) * hid],
                &want[(tok * q + qi) * hid..(tok * q + qi + 1) * hid],
                1e-4,
            );
        }
    }
}

/// With no committed states the stack is the partial alone, the softmax is trivial,
/// and every query returns it. That is the first block boundary of a sequence, and
/// what layer 0 sees.
#[test]
fn test_block_attn_with_one_state_returns_it() {
    let (t, hid, q) = (2usize, 32usize, 3usize);
    let states: Vec<f32> = Vec::new();
    let partial = values(t * hid, 123);
    let query = values(q * hid, 124);
    for qi in 0..q {
        let g = OneOp::new(&[0, hid]);
        let pin = g.input(&[t, hid]);
        let got = g.run(
            BlockAttnOp::<f32>::new(0, hid, q, qi),
            &[&pin],
            &[&states, &partial],
            &[&query],
        );
        for tok in 0..t {
            assert_close(
                "single state",
                &got[tok * hid..(tok + 1) * hid],
                &partial[tok * hid..(tok + 1) * hid],
                1e-5,
            );
        }
    }
}

/// Real NVFP4 bytes from `thinkingmachines/Inkling-2-Mini`'s expert weights, the
/// same 19 KB the RVV work in `experiments/nvfp4-riscv` is tested against.
fn nvfp4_fixture(name: &str) -> Vec<u8> {
    // `INKLING2_FIXTURES` lets the cross-built binary find its data when it runs
    // somewhere the build directory does not exist, which is the case on a board.
    let base = std::env::var("INKLING2_FIXTURES")
        .unwrap_or_else(|_| format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR")));
    let p = std::path::Path::new(&base).join("nvfp4").join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

/// E4M3 to f32, the host reference.
fn e4m3(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    let v = if e == 0 {
        (m as f32 / 8.0) * 0.015_625
    } else {
        (1.0 + m as f32 / 8.0) * (e as f32 - 7.0).exp2()
    };
    if s == 1 { -v } else { v }
}

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// `value = e2m1(nibble) * e4m3(scale[k / 16]) * scale2`, on real weights.
#[test]
fn test_nvfp4_dequant_matches_the_reference() {
    let raw = nvfp4_fixture("w13_rows.bin");
    let scales = nvfp4_fixture("w13_scale.bin");
    let s2_bytes = nvfp4_fixture("w13_scale2.bin");
    let s2 = f32::from_le_bytes([s2_bytes[0], s2_bytes[1], s2_bytes[2], s2_bytes[3]]);

    // One row: 2048 packed bytes, 4096 weights, 256 scales.
    let n = 4096usize;
    let bytes: Vec<f32> = raw[..n / 2].iter().map(|b| *b as f32).collect();
    let sc: Vec<f32> = scales[..n / 16].iter().map(|b| *b as f32).collect();

    let want: Vec<f32> = (0..n)
        .map(|i| {
            let byte = raw[i / 2];
            let nib = if i % 2 == 1 { byte >> 4 } else { byte & 0x0F };
            E2M1[nib as usize] * e4m3(scales[i / 16]) * s2
        })
        .collect();

    let g = OneOp::new(&[1, n / 2]);
    let sc_in = g.input(&[1, n / 16]);
    let got = g.run(
        Nvfp4DequantOp::<f32>::new(n),
        &[&sc_in],
        &[&bytes, &sc],
        &[&[s2]],
    );
    assert_close("nvfp4 dequant", &got, &want, 1e-5);

    // Sanity: these are real weights, not a degenerate buffer.
    let absmax = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(
        absmax > 0.01 && absmax < 10.0,
        "weights look wrong: absmax {absmax}"
    );
    let zeros = want.iter().filter(|v| **v == 0.0).count();
    assert!(
        zeros * 4 < n,
        "too many zeros for a real weight row: {zeros} of {n}"
    );
}

/// Every one of the sixteen E2M1 codes decodes to its documented value.
#[test]
fn test_nvfp4_decodes_all_sixteen_codes() {
    // Eight bytes carry all sixteen nibbles, low nibble first.
    let bytes: Vec<f32> = (0..8u8)
        .map(|i| ((2 * i + 1) * 16 + 2 * i) as f32)
        .collect();
    let n = 16usize;
    // A scale byte of 0x38 is E4M3 1.0, so the output is the raw E2M1 value.
    let sc = vec![0x38u8 as f32; 1];
    let want: Vec<f32> = (0..n)
        .map(|i| {
            let b = (2 * (i / 2) + 1) as u8 * 16 + (2 * (i / 2)) as u8;
            let nib = if i % 2 == 1 { b >> 4 } else { b & 0x0F };
            E2M1[nib as usize]
        })
        .collect();
    assert!((e4m3(0x38) - 1.0).abs() < 1e-6, "0x38 should be E4M3 1.0");
    let g = OneOp::new(&[1, n / 2]);
    let sc_in = g.input(&[1, 1]);
    let got = g.run(
        Nvfp4DequantOp::<f32>::new(n),
        &[&sc_in],
        &[&bytes, &sc],
        &[&[1.0f32]],
    );
    assert_close("all e2m1 codes", &got, &want, 1e-6);
}

/// The NVFP4 gather must equal the f32 gather fed the same weights dequantised.
///
/// This is the test that matters: the packed path and the plain path are different
/// kernels, and the model is only correct if they agree.
#[test]
fn test_nvfp4_gather_equals_the_f32_gather() {
    let (t, e, n, k, top_k) = (2usize, 3usize, 4usize, 64usize, 2usize);
    let raw = nvfp4_fixture("w13_rows.bin");
    let scales = nvfp4_fixture("w13_scale.bin");
    let s2b = nvfp4_fixture("w13_scale2.bin");
    let s2: Vec<f32> = (0..e)
        .map(|i| f32::from_le_bytes([s2b[i * 4], s2b[i * 4 + 1], s2b[i * 4 + 2], s2b[i * 4 + 3]]))
        .collect();

    // e * n rows of k weights: k/2 bytes and k/16 scales each.
    let rows = e * n;
    // Packed four bytes to an f32 element, as the kernel reads them.
    let bytes = pack_rows(
        &raw[..rows * k / 2]
            .iter()
            .map(|b| *b as f32)
            .collect::<Vec<_>>(),
        k / 2,
    );
    // The gather now takes one folded f32 per block: the E4M3 block scale times the
    // per-expert scale, as the loader computes it.
    let sc: Vec<f32> = (0..rows * k / 16)
        .map(|i| e4m3(scales[i]) * s2[i / (k / 16) / n])
        .collect();
    assert_eq!(bytes.len(), rows * k / 8, "four weight bytes per element");
    assert_eq!(sc.len(), rows * k / 16, "one folded scale per sixteen weights");

    // The same weights, dequantised on the host, for the f32 gather.
    let w: Vec<f32> = (0..rows * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            let byte = raw[row * k / 2 + col / 2];
            let nib = if col % 2 == 1 { byte >> 4 } else { byte & 0x0F };
            E2M1[nib as usize] * e4m3(scales[row * k / 16 + col / 16]) * s2[row / n]
        })
        .collect();

    let x = values(t * k, 141);
    let ids = [1usize, 2, 0, 1];
    let mut packed = vec![0.0f32; t * 2 * top_k];
    for tok in 0..t {
        for slot in 0..top_k {
            packed[tok * 2 * top_k + top_k + slot] = ids[tok * top_k + slot] as f32;
        }
    }

    let gf = OneOp::new(&[t, k]);
    let ids_f = gf.input(&[t, 2 * top_k]);
    let plain = gf.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_f],
        &[&x, &packed],
        &[&w],
    );

    let gq = OneOp::new(&[t, k]);
    let ids_q = gq.input(&[t, 2 * top_k]);
    let packed_out = gq.run(
        ExpertGatherNvfp4Op::<f32>::new(e, n, k, top_k),
        &[&ids_q],
        &[&x, &packed],
        &[&bytes, &sc],
    );
    assert_close("nvfp4 gather vs f32 gather", &packed_out, &plain, 1e-4);
    assert!(
        plain.iter().any(|v| v.abs() > 1e-6),
        "the comparison is only meaningful on non-zero output"
    );
}

/// Four bytes unpack from one bit-packed f32, which is what the 0.5-bytes-per-weight
/// NVFP4 encoding needs.
///
/// Both halves of this were broken in the shipped toolchain and are fixed in the
/// local build (teenygrad#170): `T::cast`'s `bitcast` flag was ignored, so the cast
/// converted by value. `BitAnd` on an integer tensor was simply missing and is added
/// in this branch. `Shr` is still not lowered, so the unpacking masks first and
/// divides second, which also keeps every intermediate non-negative - `Div`
/// truncates towards zero, so the top byte has to be masked before it is shifted
/// down, and its sign bit put back by hand.
#[test]
fn test_four_bytes_unpack_from_one_packed_word() {
    let raw = nvfp4_fixture("w13_rows.bin");
    let n = 64usize;
    let words: Vec<f32> = (0..n)
        .map(|i| f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]))
        .collect();
    let negatives = words.iter().filter(|w| w.to_bits() >> 31 == 1).count();
    assert!(
        negatives > 0,
        "the fixture must include words with the top bit set"
    );

    for which in 0..4 {
        let want: Vec<f32> = (0..n).map(|i| raw[i * 4 + which] as f32).collect();
        let g = OneOp::new(&[n]);
        let got = g.run(UnpackByteOp::<f32>::new(which), &[], &[&words], &[]);
        assert_close(&format!("byte {which}"), &got, &want, 0.0);
    }

    // The bitcast is doing real work: a value conversion would give something else
    // entirely for most of these words.
    let by_value: Vec<f32> = words.iter().map(|w| ((*w as i32) & 0xFF) as f32).collect();
    let by_bits: Vec<f32> = (0..n).map(|i| raw[i * 4] as f32).collect();
    let differing = by_value
        .iter()
        .zip(&by_bits)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > n / 2,
        "the two readings must differ for this test to mean anything ({differing} of {n})"
    );
}

/// `w[t,c,j] = static[j,c] + dyn[t, head(c)*K + j]`, with j the BACKWARD distance,
/// then `(conv + x) / sqrt(2K + 1)`.
#[test]
fn test_dyn_sconv_matches_the_reference() {
    const K: usize = 4;
    let (t, heads, hd) = (5usize, 3usize, 8usize);
    let c = heads * hd;
    let hist = values((K - 1) * c, 151);
    let x = values(t * c, 152);
    let dynw = values(t * heads * K, 153);
    let stat = values(K * c, 154);

    // Row r of the virtual [hist; x] stack.
    let row = |r: isize, ch: usize| -> f32 {
        if r < (K - 1) as isize {
            hist[r as usize * c + ch]
        } else {
            x[(r as usize - (K - 1)) * c + ch]
        }
    };
    let want: Vec<f32> = (0..t * c)
        .map(|i| {
            let (tok, ch) = (i / c, i % c);
            let head = ch / hd;
            let conv: f32 = (0..K)
                .map(|j| {
                    let w = stat[j * c + ch] + dynw[tok * heads * K + head * K + j];
                    // j = 0 is the current token.
                    w * row((tok + K - 1 - j) as isize, ch)
                })
                .sum();
            (conv + x[tok * c + ch]) / ((2 * K + 1) as f32).sqrt()
        })
        .collect();

    let g = OneOp::new(&[K - 1, c]);
    let x_in = g.input(&[t, c]);
    let d_in = g.input(&[t, heads * K]);
    let got = g.run(
        DynSconvOp::<f32>::new(c, heads, K),
        &[&x_in, &d_in],
        &[&hist, &x, &dynw],
        &[&stat],
    );
    assert_close("dyn sconv", &got, &want, 1e-4);
}

/// Tap 0 is the CURRENT token, the opposite of the static convolution. With only tap
/// 0 non-zero the output is that token alone, scaled; with only the last tap it is
/// the oldest token in the window.
#[test]
fn test_dyn_sconv_tap_zero_is_the_current_token() {
    const K: usize = 4;
    let (t, heads, hd) = (1usize, 1usize, 4usize);
    let c = heads * hd;
    let hist: Vec<f32> = (0..(K - 1) * c).map(|i| (i + 1) as f32).collect();
    let x = vec![100.0f32; c];
    let dynw = vec![0.0f32; t * heads * K];
    let norm = ((2 * K + 1) as f32).sqrt();

    for (tap, label) in [(0usize, "current"), (K - 1, "oldest")] {
        let mut stat = vec![0.0f32; K * c];
        for ch in 0..c {
            stat[tap * c + ch] = 1.0;
        }
        let g = OneOp::new(&[K - 1, c]);
        let x_in = g.input(&[t, c]);
        let d_in = g.input(&[t, heads * K]);
        let got = g.run(
            DynSconvOp::<f32>::new(c, heads, K),
            &[&x_in, &d_in],
            &[&hist, &x, &dynw],
            &[&stat],
        );
        let want: Vec<f32> = (0..c)
            .map(|ch| {
                let picked = if tap == 0 {
                    x[ch]
                } else {
                    hist[(K - 1 - tap) * c + ch]
                };
                (picked + x[ch]) / norm
            })
            .collect();
        assert_close(&format!("dyn sconv tap {tap} ({label})"), &got, &want, 1e-4);
    }
}

/// The dynamic taps are shared by every channel of a head, so two heads given
/// different taps must differ while channels within a head agree.
#[test]
fn test_dyn_sconv_taps_are_per_head() {
    const K: usize = 4;
    let (t, heads, hd) = (1usize, 2usize, 4usize);
    let c = heads * hd;
    let hist = vec![0.0f32; (K - 1) * c];
    let x = vec![1.0f32; c];
    let stat = vec![0.0f32; K * c];
    // Head 0 gets tap 0 = 1, head 1 gets tap 0 = 2.
    let mut dynw = vec![0.0f32; t * heads * K];
    dynw[0] = 1.0;
    dynw[K] = 2.0;
    let g = OneOp::new(&[K - 1, c]);
    let x_in = g.input(&[t, c]);
    let d_in = g.input(&[t, heads * K]);
    let got = g.run(
        DynSconvOp::<f32>::new(c, heads, K),
        &[&x_in, &d_in],
        &[&hist, &x, &dynw],
        &[&stat],
    );
    let norm = ((2 * K + 1) as f32).sqrt();
    for ch in 0..hd {
        assert!(
            (got[ch] - 2.0 / norm).abs() < 1e-4,
            "head 0 channel {ch}: {}",
            got[ch]
        );
        assert!(
            (got[hd + ch] - 3.0 / norm).abs() < 1e-4,
            "head 1 channel {ch}: {}",
            got[hd + ch]
        );
    }
}

/// MXFP8: an E4M3 byte per weight, an E8M0 scale per 32, no per-expert scale.
///
/// Compared against the plain f32 gather fed the same weights decoded on the host,
/// the same way the NVFP4 gather is checked.
#[test]
fn test_mxfp8_gather_equals_the_f32_gather() {
    let (t, e, n, k, top_k) = (2usize, 3usize, 4usize, 64usize, 2usize);
    let rows = e * n;
    // Deterministic E4M3 and E8M0 bytes. Skip 0xFF: E8M0 reserves it for NaN, and
    // E4M3's 0x7F/0xFF are NaN too.
    let wbytes: Vec<u8> = (0..rows * k).map(|i| ((i * 37 + 11) % 254) as u8).collect();
    let sbytes: Vec<u8> = (0..rows * k / 32)
        .map(|i| (120 + (i % 14)) as u8) // exponents around 2^-7 .. 2^6
        .collect();

    let e8m0 = |b: u8| (b as f32 - 127.0).exp2();
    let w: Vec<f32> = (0..rows * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            e4m3(wbytes[i]) * e8m0(sbytes[row * (k / 32) + col / 32])
        })
        .collect();
    assert!(
        w.iter().all(|v| v.is_finite()),
        "the fixture must avoid NaN encodings"
    );

    let x = values(t * k, 161);
    let ids = [2usize, 0, 1, 2];
    let mut packed = vec![0.0f32; t * 2 * top_k];
    for tok in 0..t {
        for slot in 0..top_k {
            packed[tok * 2 * top_k + top_k + slot] = ids[tok * top_k + slot] as f32;
        }
    }

    let gf = OneOp::new(&[t, k]);
    let ids_f = gf.input(&[t, 2 * top_k]);
    let plain = gf.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_f],
        &[&x, &packed],
        &[&w],
    );

    let bytes = pack_rows(&wbytes.iter().map(|b| *b as f32).collect::<Vec<_>>(), k);
    let scales = pack_rows(
        &sbytes.iter().map(|b| *b as f32).collect::<Vec<_>>(),
        k / 32,
    );
    assert_eq!(bytes.len(), rows * k / 4, "four weight bytes per element");
    let gq = OneOp::new(&[t, k]);
    let ids_q = gq.input(&[t, 2 * top_k]);
    let got = gq.run(
        ExpertGatherMxfp8Op::<f32>::new(e, n, k, top_k),
        &[&ids_q],
        &[&x, &packed],
        &[&bytes, &scales],
    );
    assert_close("mxfp8 gather vs f32 gather", &got, &plain, 1e-3);
    assert!(
        plain.iter().any(|v| v.abs() > 1e-6),
        "the comparison is only meaningful on non-zero output"
    );
}

/// The same comparison with a `k` long enough to span several packed scale elements.
///
/// The gather derives its scale index from `k >> 7` and its byte-within-element from
/// `(k >> 5) & 3` per chunk. With `k = 64` neither crosses a packed element, so a
/// mistake in that arithmetic passes unnoticed; 320 gives ten scales over three
/// elements, the last of them partial, and exercises every value of the selector.
#[test]
fn test_mxfp8_gather_spans_several_scale_elements() {
    let (t, e, n, k, top_k) = (2usize, 3usize, 4usize, 320usize, 2usize);
    let rows = e * n;
    let wbytes: Vec<u8> = (0..rows * k).map(|i| ((i * 37 + 11) % 254) as u8).collect();
    // A different exponent for every scale, so a wrong index gives a wrong answer
    // rather than coincidentally the right one.
    let sbytes: Vec<u8> = (0..rows * k / 32).map(|i| (114 + (i % 26)) as u8).collect();

    let e8m0 = |b: u8| (b as f32 - 127.0).exp2();
    let w: Vec<f32> = (0..rows * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            e4m3(wbytes[i]) * e8m0(sbytes[row * (k / 32) + col / 32])
        })
        .collect();
    assert!(
        w.iter().all(|v| v.is_finite()),
        "the fixture must avoid NaN encodings"
    );

    let x = values(t * k, 163);
    let ids = [2usize, 0, 1, 2];
    let mut packed = vec![0.0f32; t * 2 * top_k];
    for tok in 0..t {
        for slot in 0..top_k {
            packed[tok * 2 * top_k + top_k + slot] = ids[tok * top_k + slot] as f32;
        }
    }

    let gf = OneOp::new(&[t, k]);
    let ids_f = gf.input(&[t, 2 * top_k]);
    let plain = gf.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_f],
        &[&x, &packed],
        &[&w],
    );

    let bytes = pack_rows(&wbytes.iter().map(|b| *b as f32).collect::<Vec<_>>(), k);
    let scales = pack_rows(
        &sbytes.iter().map(|b| *b as f32).collect::<Vec<_>>(),
        k / 32,
    );
    let gq = OneOp::new(&[t, k]);
    let ids_q = gq.input(&[t, 2 * top_k]);
    let got = gq.run(
        ExpertGatherMxfp8Op::<f32>::new(e, n, k, top_k),
        &[&ids_q],
        &[&x, &packed],
        &[&bytes, &scales],
    );
    assert_close(
        "mxfp8 gather over several scale elements",
        &got,
        &plain,
        1e-3,
    );
    assert!(
        plain.iter().any(|v| v.abs() > 1e-6),
        "the comparison is only meaningful on non-zero output"
    );
}

/// An E8M0 scale of 127 is exactly 1.0, so the output is the raw E4M3 values.
#[test]
fn test_mxfp8_scale_127_is_unity() {
    let (t, e, n, k, top_k) = (1usize, 1usize, 1usize, 32usize, 1usize);
    let wbytes: Vec<u8> = (0..k).map(|i| (56 + i) as u8).collect(); // 0x38 is 1.0
    let bytes = pack_rows(&wbytes.iter().map(|b| *b as f32).collect::<Vec<_>>(), k);
    // One scale per 32 weights, so one per row here, padded to a word.
    let scales = pack_rows(&[127.0f32], 1);
    let x = vec![1.0f32; k];
    let packed = vec![0.0f32; 2];
    let want: f32 = wbytes.iter().map(|b| e4m3(*b)).sum();

    let g = OneOp::new(&[t, k]);
    let ids = g.input(&[t, 2 * top_k]);
    let got = g.run(
        ExpertGatherMxfp8Op::<f32>::new(e, n, k, top_k),
        &[&ids],
        &[&x, &packed],
        &[&bytes, &scales],
    );
    assert!(
        (got[0] - want).abs() <= 1e-3 * want.abs().max(1.0),
        "scale 127 should be unity: got {}, want {want}",
        got[0]
    );
}

/// The loader's output is exactly what the gather expects.
///
/// Writes real NVFP4 bytes into a safetensors checkpoint, reads them back through
/// `Checkpoint::param`, and runs the packed gather on the result, comparing against
/// the plain f32 gather fed the same weights decoded on the host. A layout
/// disagreement between loader and kernel would show up here as wrong numbers;
/// anywhere else it would show up as a fluent model saying nonsense.
#[test]
fn test_the_loader_feeds_the_packed_gather_correctly() {
    use safetensors::{Dtype, tensor::TensorView};
    use std::collections::HashMap;

    let (t, e, n, k, top_k) = (2usize, 2usize, 4usize, 64usize, 2usize);
    let rows = e * n;
    let raw = nvfp4_fixture("w13_rows.bin");
    let scales = nvfp4_fixture("w13_scale.bin");
    let wbytes = &raw[..rows * k / 2];
    let sbytes = &scales[..rows * k / 16];
    let amax: Vec<f32> = (0..e).map(|i| 2.0 + i as f32).collect();
    let denom = 4.0f32;

    // A checkpoint holding them with the real dtypes and shapes.
    let dir = std::env::temp_dir().join(format!("inkling2-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let amax_b: Vec<u8> = amax.iter().flat_map(|v| v.to_le_bytes()).collect();
    let denom_b = denom.to_le_bytes().to_vec();
    let tensors = HashMap::from([
        (
            "w13_weight".to_string(),
            TensorView::new(Dtype::U8, vec![e, n, k / 2], wbytes).expect("weights"),
        ),
        (
            "w13_weight_scale".to_string(),
            TensorView::new(Dtype::F8_E4M3, vec![e, n, k / 16], sbytes).expect("scales"),
        ),
        (
            "w13_weight_amax".to_string(),
            TensorView::new(Dtype::F32, vec![e], &amax_b).expect("amax"),
        ),
        (
            "w13_weight_outer_scale_denominator".to_string(),
            TensorView::new(Dtype::F32, vec![1], &denom_b).expect("denominator"),
        ),
    ]);
    std::fs::write(
        dir.join("model.safetensors"),
        safetensors::serialize(&tensors, None).expect("serialize"),
    )
    .expect("write");

    let mut ck = Checkpoint::open(&dir).expect("opening the checkpoint");
    let w_packed = ck.param("w13_weight").expect("packed weights");
    let s2 = ck.expert_scale2("w13_weight").expect("per-expert scale");
    // The loader folds the block scale and the per-expert scale together, so the
    // gather gets one f32 per sixteen weights and no separate per-expert scale.
    let s_folded = ck.folded_scale("w13_weight").expect("folded scales");
    assert_eq!(
        w_packed.len(),
        rows * k / 8,
        "the loader packs to the gather's shape"
    );
    assert_eq!(s_folded.len(), rows * k / 16, "one folded scale per block");
    assert_eq!(s2, vec![0.5, 0.75], "amax over the denominator");

    // The same weights decoded on the host, for the f32 gather.
    let want_w: Vec<f32> = (0..rows * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            let byte = wbytes[row * k / 2 + col / 2];
            let nib = if col % 2 == 1 { byte >> 4 } else { byte & 0x0F };
            E2M1[nib as usize] * e4m3(sbytes[row * k / 16 + col / 16]) * s2[row / n]
        })
        .collect();

    let x = values(t * k, 181);
    let ids = [1usize, 0, 0, 1];
    let mut packed = vec![0.0f32; t * 2 * top_k];
    for tok in 0..t {
        for slot in 0..top_k {
            packed[tok * 2 * top_k + top_k + slot] = ids[tok * top_k + slot] as f32;
        }
    }

    let gf = OneOp::new(&[t, k]);
    let ids_f = gf.input(&[t, 2 * top_k]);
    let plain = gf.run(
        ExpertGatherOp::<f32>::new(e, n, k, top_k),
        &[&ids_f],
        &[&x, &packed],
        &[&want_w],
    );

    let gq = OneOp::new(&[t, k]);
    let ids_q = gq.input(&[t, 2 * top_k]);
    let got = gq.run(
        ExpertGatherNvfp4Op::<f32>::new(e, n, k, top_k),
        &[&ids_q],
        &[&x, &packed],
        &[&w_packed, &s_folded],
    );
    assert_close("loader -> packed gather", &got, &plain, 1e-4);
    assert!(
        plain.iter().any(|v| v.abs() > 1e-6),
        "the output must not be all zero"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
