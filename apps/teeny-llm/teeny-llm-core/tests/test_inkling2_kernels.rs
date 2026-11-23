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

#![cfg(all(feature = "inkling2", any(feature = "cuda", feature = "qemu")))]
use std::cell::RefCell;
use std::rc::Rc;

use teeny_core::graph::{CustomData, CustomOp, DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_kernels::graph::TritonLowering;
use teeny_llm_core::models::inkling2::kernels::{
    l2_clip::ScaledL2ClipOp, rel_bias::RelBiasOp, sconv::SconvOp, softcap::SoftcapOp,
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

    // Used by the multi-input kernels this file will grow; kept so the harness
    // stays identical to test_llm_kernels.rs.
    #[allow(dead_code)]
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
            (0..K).map(|k| w[ch * K + k] * x[(row + k) * c + ch]).sum()
        })
        .collect();
    let g = OneOp::new(&[t + K - 1, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[], &[&x], &[&w]);
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
    let want: Vec<f32> = (0..c).map(|ch| w[ch * K + (K - 1)] * row[ch]).collect();
    let g = OneOp::new(&[K, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[], &[&x], &[&w]);
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
            (0..K).map(|k| w[ch * K + k] * x[(row + k) * c + ch]).sum()
        })
        .collect();
    let g = OneOp::new(&[t + K - 1, c]);
    let got = g.run(SconvOp::<f32>::new(c, K), &[], &[&x], &[&w]);
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
        RelBiasOp::<f32>::new(heads, extent, past, kv),
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
    let kv = past + t;
    let logits: Vec<f32> = (0..t * heads * extent).map(|i| 1.0 + i as f32).collect();
    let g = OneOp::new(&[t * heads, extent]);
    let got = g.run(
        RelBiasOp::<f32>::new(heads, extent, past, kv),
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
        RelBiasOp::<f32>::new(heads, extent, past, kv),
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
        RelBiasOp::<f32>::new(heads, extent, past, kv),
        &[],
        &[&logits],
        &[],
    );
    assert_close("rel bias ragged", &got, &want, 0.0);
}
