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

//! Inkling-2's kernels against the **first-party reference**: `transformers`'
//! `models/inkling2_loop` modules, run once on the workstation to produce
//! `tests/fixtures/reference/inkling2-reference.{json,bin}` (inputs, weights and
//! expected outputs), then replayed here on the device.
//!
//! This exists because the tests in `test_inkling2_kernels.rs` compare each kernel with
//! a host reference written from the same reading of the model that produced the kernel.
//! When that reading is wrong both agree and the test passes anyway, which is exactly
//! what happened with the short convolution's missing residual. These fixtures come from
//! code we did not write, so a shared misreading cannot hide here.
//!
//! The fixture is deliberately small-shaped (hidden 32, 6 tokens): the kernels are
//! shape-generic, and the model's real scalars (thresholds, caps, eps, softcaps) are
//! kept. Regenerate with `scripts/inkling2-gen-reference.py`.

#![cfg(all(
    feature = "inkling2",
    any(feature = "cuda", feature = "qemu", feature = "riscv")
))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use teeny_core::graph::{CustomData, CustomOp, DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_kernels::graph::TritonLowering;
use teeny_llm_core::models::inkling2::kernels::{
    attention::PastLen,
    block_attn::BlockAttnOp,
    dyn_sconv::DynSconvOp,
    l2_clip::{L2ClipOp, ScaledL2ClipOp},
    mix::MessageMixOp,
    norm::{RmsNormOp, UnweightedRmsNormOp},
    rel_bias::RelBiasOp,
    router::TopKRouterOp,
    sconv::SconvOp,
    softcap::SoftcapOp,
    swiglu::SwigluOp,
};
use teeny_test::model::{RuntimeModel, runtime};

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

/// One flat `f32` blob plus a manifest of `name -> (offset, len)` and `name.shape`.
struct Fixture {
    data: Vec<f32>,
    spans: HashMap<String, (usize, usize)>,
    shapes: HashMap<String, Vec<usize>>,
    scalars: HashMap<String, HashMap<String, f64>>,
}

impl Fixture {
    fn load() -> Self {
        // `INKLING2_FIXTURES` lets the cross-built binary find its data on the board.
        let dir = std::env::var("INKLING2_FIXTURES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
            })
            .join("reference");
        let bin = std::fs::read(dir.join("inkling2-reference.bin"))
            .expect("fixtures/reference/inkling2-reference.bin");
        let json = std::fs::read_to_string(dir.join("inkling2-reference.json"))
            .expect("fixtures/reference/inkling2-reference.json");
        let data: Vec<f32> = bin
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // A tiny hand parser: the manifest is a flat object of objects of numbers and
        // number arrays, and pulling in a JSON crate for it is not worth a dependency.
        let (mut spans, mut shapes, mut scalars) = (HashMap::new(), HashMap::new(), HashMap::new());
        let tensors = section(&json, "\"tensors\"");
        for (key, body) in entries(&tensors) {
            if let Some(name) = key.strip_suffix(".shape") {
                shapes.insert(name.to_string(), numbers(&body, "dims"));
            } else {
                let off = numbers(&body, "offset")[0];
                let len = numbers(&body, "len")[0];
                spans.insert(key, (off, len));
            }
        }
        let cases = section(&json, "\"cases\"");
        for (key, body) in entries(&cases) {
            let mut m = HashMap::new();
            let inner = section(&body, "\"scalars\"");
            for part in inner.split(',') {
                let mut it = part.splitn(2, ':');
                if let (Some(k), Some(v)) = (it.next(), it.next()) {
                    let k = k.trim().trim_matches(|c| c == '"' || c == '{' || c == '}');
                    if let Ok(v) = v.trim().trim_matches(|c| c == '}' || c == ' ').parse::<f64>() {
                        m.insert(k.to_string(), v);
                    }
                }
            }
            scalars.insert(key, m);
        }
        Self { data, spans, shapes, scalars }
    }

    fn get(&self, name: &str) -> &[f32] {
        let &(off, len) = self
            .spans
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no tensor {name}"));
        &self.data[off..off + len]
    }

    fn shape(&self, name: &str) -> &[usize] {
        self.shapes
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no shape for {name}"))
    }

    fn scalar(&self, case: &str, name: &str) -> f32 {
        *self
            .scalars
            .get(case)
            .unwrap_or_else(|| panic!("fixture has no case {case}"))
            .get(name)
            .unwrap_or_else(|| panic!("case {case} has no scalar {name}")) as f32
    }

    fn usize_of(&self, case: &str, name: &str) -> usize {
        self.scalar(case, name) as usize
    }
}

/// The body of `"key": { ... }` at the top level of `json`, braces stripped.
fn section(json: &str, key: &str) -> String {
    let start = json.find(key).unwrap_or_else(|| panic!("no {key} in the fixture"));
    let open = json[start..].find('{').expect("an object after the key") + start;
    let mut depth = 0usize;
    for (i, c) in json[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return json[open + 1..open + i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after {key}");
}

/// `"name": { body }` pairs at the top level of `body`.
fn entries(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes: Vec<char> = body.chars().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != '"' {
            i += 1;
            continue;
        }
        let name_start = i + 1;
        let mut j = name_start;
        while j < bytes.len() && bytes[j] != '"' {
            j += 1;
        }
        let name: String = bytes[name_start..j].iter().collect();
        // Find the object that follows.
        let mut k = j + 1;
        while k < bytes.len() && bytes[k] != '{' {
            if bytes[k] == '"' {
                break;
            }
            k += 1;
        }
        if k >= bytes.len() || bytes[k] != '{' {
            i = j + 1;
            continue;
        }
        let mut depth = 0usize;
        let mut e = k;
        while e < bytes.len() {
            match bytes[e] {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            e += 1;
        }
        out.push((name, bytes[k..=e].iter().collect()));
        i = e + 1;
    }
    out
}

/// The numbers of `"key": N` or `"key": [N, M, ...]` inside `body`.
fn numbers(body: &str, key: &str) -> Vec<usize> {
    let at = body
        .find(&format!("\"{key}\""))
        .unwrap_or_else(|| panic!("no {key} in {body}"));
    // Past the key, its quotes and the colon.
    let rest = body[at + key.len() + 2..].trim_start();
    let rest = rest.strip_prefix(':').expect("a colon after the key").trim_start();
    // An array runs to its closing bracket; a bare number to the next comma or brace.
    let end = if rest.starts_with('[') {
        rest.find(']').expect("a closing bracket") + 1
    } else {
        rest.find([',', '}']).unwrap_or(rest.len())
    };
    rest[..end]
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("a number"))
        .collect()
}

// ---------------------------------------------------------------------------
// Running one op (the same harness as test_inkling2_kernels.rs)
// ---------------------------------------------------------------------------

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

    fn run(
        &self,
        op: impl CustomOp + 'static,
        others: &[&SymTensor],
        feeds: &[&[f32]],
        params: &[&[f32]],
    ) -> Vec<f32> {
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

/// Asserts `got` matches the reference to `tol`, relative to each value's magnitude.
fn assert_matches_reference(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: output length");
    let mut worst = (0usize, 0.0f32);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        if err > worst.1 {
            worst = (i, err);
        }
    }
    assert!(
        worst.1 <= tol,
        "{what}: worst element {} is {} against the reference's {} \
         (relative error {}, tolerance {tol})",
        worst.0,
        got[worst.0],
        want[worst.0],
        worst.1
    );
}

/// `y = x @ w.T`, with `w` row-major `[out, in]`.
fn linear(x: &[f32], w: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let rows = x.len() / n_in;
    let mut y = vec![0.0f32; rows * n_out];
    for r in 0..rows {
        for o in 0..n_out {
            y[r * n_out + o] = (0..n_in).map(|i| x[r * n_in + i] * w[o * n_in + i]).sum();
        }
    }
    y
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

#[test]
fn test_rms_norm_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("rms_norm.in.x");
    let w = f.get("rms_norm.w.weight");
    let g = OneOp::new(&[x.len() / w.len(), w.len()]);
    let got = g.run(RmsNormOp::<f32>::new(w.len()), &[], &[x], &[w]);
    assert_matches_reference("rms norm", &got, f.get("rms_norm.out"), 1e-5);
}

#[test]
fn test_unweighted_rms_norm_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("unweighted_rms_norm.in.x");
    let r = *f.shape("unweighted_rms_norm.in.x").last().unwrap();
    let g = OneOp::new(&[x.len() / r, r]);
    let got = g.run(UnweightedRmsNormOp::<f32>::new(r), &[], &[x], &[]);
    assert_matches_reference("unweighted rms norm", &got, f.get("unweighted_rms_norm.out"), 1e-5);
}

#[test]
fn test_softcap_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("softcap.in.x");
    let cap = f.scalar("softcap", "cap");
    let g = OneOp::new(&[x.len()]);
    let got = g.run(SoftcapOp::<f32>::new(cap), &[], &[x], &[]);
    assert_matches_reference("softcap", &got, f.get("softcap.out"), 1e-5);
}

#[test]
fn test_swiglu_matches_the_reference() {
    let f = Fixture::load();
    // The kernel takes gate and up concatenated, as the checkpoint stores them.
    let (gate, up) = (f.get("swiglu.in.gate"), f.get("swiglu.in.up"));
    let i = f.usize_of("swiglu", "intermediate");
    let rows = gate.len() / i;
    // The kernel reads gate and up from interleaved lanes, as `w13_weight` stores
    // them; the reference module takes them as two separate tensors.
    let mut cat = vec![0.0f32; gate.len() * 2];
    for r in 0..rows {
        for c in 0..i {
            cat[r * 2 * i + c * 2] = gate[r * i + c];
            cat[r * 2 * i + c * 2 + 1] = up[r * i + c];
        }
    }
    let g = OneOp::new(&[rows, 2 * i]);
    let got = g.run(
        SwigluOp::<f32>::new(
            i,
            f.scalar("swiglu", "gate_softcap"),
            f.scalar("swiglu", "up_softcap"),
        ),
        &[],
        &[&cat],
        &[],
    );
    assert_matches_reference("swiglu", &got, f.get("swiglu.out"), 1e-5);
}

#[test]
fn test_l2_clip_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("l2_clip.in.x");
    let hd = *f.shape("l2_clip.in.x").last().unwrap();
    let g = OneOp::new(&[x.len() / hd, hd]);
    let got = g.run(
        L2ClipOp::<f32>::new(hd, f.scalar("l2_clip", "threshold")),
        &[],
        &[x],
        &[],
    );
    assert_matches_reference("l2 clip", &got, f.get("l2_clip.out"), 1e-5);
}

#[test]
fn test_scaled_l2_clip_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("scaled_l2_clip.in.x");
    let scale = f.get("scaled_l2_clip.w.scale");
    let sh = f.shape("scaled_l2_clip.w.scale");
    let (heads, hd) = (sh[0], sh[1]);
    let g = OneOp::new(&[x.len() / (heads * hd), heads * hd]);
    let got = g.run(
        ScaledL2ClipOp::<f32>::new(heads, hd, f.scalar("scaled_l2_clip", "threshold")),
        &[],
        &[x],
        &[scale],
    );
    assert_matches_reference("scaled l2 clip", &got, f.get("scaled_l2_clip.out"), 1e-5);
}

#[test]
fn test_sconv_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("sconv.in.x");
    let w = f.get("sconv.w.weight");
    let k = f.usize_of("sconv", "kernel");
    let c = *f.shape("sconv.in.x").last().unwrap();
    // The reference pads the sequence with `k - 1` zero rows; the kernel takes that
    // history as its own first input.
    let hist = vec![0.0f32; (k - 1) * c];
    let g = OneOp::new(&[k - 1, c]);
    let x_in = g.input(&[x.len() / c, c]);
    let got = g.run(SconvOp::<f32>::new(c, k), &[&x_in], &[&hist, x], &[w]);
    assert_matches_reference("sconv", &got, f.get("sconv.out"), 1e-5);
}

#[test]
fn test_dyn_sconv_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("dyn_sconv.in.x");
    let dyn_in = f.get("dyn_sconv.in.dynamic_input");
    let stat = f.get("dyn_sconv.w.static_weight");
    let proj = f.get("dyn_sconv.w.weight_proj");
    let k = f.usize_of("dyn_sconv", "kernel");
    let heads = f.usize_of("dyn_sconv", "num_heads");
    let c = *f.shape("dyn_sconv.in.x").last().unwrap();
    let hidden = *f.shape("dyn_sconv.in.dynamic_input").last().unwrap();
    // The per-token taps are a plain linear projection of the layer input, done outside
    // the kernel exactly as the model graph does it.
    let dynw = linear(dyn_in, proj, hidden, heads * k);
    let hist = vec![0.0f32; (k - 1) * c];
    let g = OneOp::new(&[k - 1, c]);
    let x_in = g.input(&[x.len() / c, c]);
    let d_in = g.input(&[dynw.len() / (heads * k), heads * k]);
    let got = g.run(
        DynSconvOp::<f32>::new(c, heads, k),
        &[&x_in, &d_in],
        &[&hist, x, &dynw],
        &[stat],
    );
    assert_matches_reference("dyn sconv", &got, f.get("dyn_sconv.out"), 1e-5);
}

#[test]
fn test_router_matches_the_reference() {
    let f = Fixture::load();
    let x = f.get("router.in.x");
    let w = f.get("router.w.weight");
    let bias = f.get("router.w.bias");
    let e = f.usize_of("router", "n_experts");
    let k = f.usize_of("router", "top_k");
    let scale = f.scalar("router", "route_scale");
    let hidden = *f.shape("router.in.x").last().unwrap();
    let logits = linear(x, w, hidden, e);
    let rows = x.len() / hidden;
    let g = OneOp::new(&[rows, e]);
    let got = g.run(TopKRouterOp::<f32>::new(e, k, scale), &[], &[&logits], &[bias]);

    let want_w = f.get("router.out.weights");
    let want_i = f.get("router.out.ids");
    // The reference's topk is `sorted=False`, so only the SET of (expert, weight) pairs
    // is defined. Compare as a map.
    for row in 0..rows {
        let mut want: HashMap<usize, f32> = HashMap::new();
        for s in 0..k {
            want.insert(want_i[row * k + s] as usize, want_w[row * k + s]);
        }
        for s in 0..k {
            let id = got[row * 2 * k + k + s] as usize;
            let weight = got[row * 2 * k + s];
            let expected = want
                .get(&id)
                .unwrap_or_else(|| panic!("row {row}: the reference did not pick expert {id}"));
            assert!(
                (weight - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                "row {row} expert {id}: weight {weight} against the reference's {expected}"
            );
        }
    }
}

#[test]
fn test_rel_bias_matches_the_reference() {
    let f = Fixture::load();
    // The kernel is the gather; the projection and clip that build its input are
    // separate ops in the model graph, so they are done here on the host.
    let x = f.get("rel_bias.in.x");
    let lin = f.get("rel_bias.w.linear");
    let scale = f.get("rel_bias.w.scale");
    let proj = f.get("rel_bias.w.proj");
    let heads = f.usize_of("rel_bias", "num_heads");
    let d_rel = f.usize_of("rel_bias", "d_rel");
    let extent = f.usize_of("rel_bias", "rel_extent");
    let thr = f.scalar("rel_bias", "threshold");
    let hidden = *f.shape("rel_bias.in.x").last().unwrap();
    let t = x.len() / hidden;

    let rel = linear(x, lin, hidden, heads * d_rel);
    // The scaled L2 clip, per (token, head) vector of d_rel.
    let mut normed = vec![0.0f32; rel.len()];
    for row in 0..t {
        for h in 0..heads {
            let base = row * heads * d_rel + h * d_rel;
            let v: Vec<f32> = (0..d_rel)
                .map(|i| rel[base + i] * scale[h * d_rel + i])
                .collect();
            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            let ratio = (thr / (norm + 1e-6)).min(1.0);
            for i in 0..d_rel {
                normed[base + i] = v[i] * ratio;
            }
        }
    }
    // rel_logits[t, h, :] = normed[t, h, :] @ proj, with proj [d_rel, extent].
    let mut logits = vec![0.0f32; t * heads * extent];
    for r in 0..t * heads {
        for j in 0..extent {
            logits[r * extent + j] = (0..d_rel)
                .map(|i| normed[r * d_rel + i] * proj[i * extent + j])
                .sum();
        }
    }

    let g = OneOp::new(&[t * heads, extent]);
    let past = PastLen::default();
    past.set(0);
    let got = g.run(
        RelBiasOp::<f32>::new(heads, extent, past),
        &[],
        &[&logits],
        &[],
    );

    // The reference's bias is [1, heads, t, kv]; the kernel's rows are (token, head).
    let kv = t;
    let want = f.get("rel_bias.out");
    let mut reordered = vec![0.0f32; t * heads * kv];
    for h in 0..heads {
        for tok in 0..t {
            for j in 0..kv {
                reordered[(tok * heads + h) * kv + j] = want[(h * t + tok) * kv + j];
            }
        }
    }
    assert_matches_reference("rel bias", &got, &reordered, 1e-4);
}

#[test]
fn test_block_attn_matches_the_reference() {
    let f = Fixture::load();
    let states = f.get("block_attn.in.states");
    let partial = f.get("block_attn.in.partial");
    let query = f.get("block_attn.w.query");
    let s = f.usize_of("block_attn", "states");
    let hidden = f.usize_of("block_attn", "hidden");
    let queries = f.usize_of("block_attn", "queries");
    let t = partial.len() / hidden;

    // The reference norms the stack once and uses it as keys and values; the model
    // graph does that with a separate op, so it is done here.
    let normed = |v: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; v.len()];
        for r in 0..v.len() / hidden {
            let row = &v[r * hidden..(r + 1) * hidden];
            let ms = row.iter().map(|a| a * a).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for i in 0..hidden {
                out[r * hidden + i] = row[i] * inv;
            }
        }
        out
    };
    let ns = normed(states);
    let np = normed(partial);

    for qi in 0..queries {
        let g = OneOp::new(&[t * s, hidden]);
        let p_in = g.input(&[t, hidden]);
        let got = g.run(
            BlockAttnOp::<f32>::new(s, hidden, queries, qi),
            &[&p_in],
            &[&ns, &np],
            &[query],
        );
        assert_matches_reference(
            &format!("block attn query {qi}"),
            &got,
            f.get(&format!("block_attn.out{qi}")),
            1e-4,
        );
    }
}

#[test]
fn test_message_gate_matches_the_reference() {
    let f = Fixture::load();
    // The kernel is the final combine; the low-rank gate projections around it are
    // ordinary linears in the model graph, so they are done here on the host.
    let h = f.get("message_gate.in.x");
    let m = f.get("message_gate.in.message");
    let down = f.get("message_gate.w.down");
    let forget_w = f.get("message_gate.w.forget");
    let gain_w = f.get("message_gate.w.gain");
    let rank = f.usize_of("message_gate", "rank");
    let hidden = *f.shape("message_gate.in.x").last().unwrap();
    let t = h.len() / hidden;

    let norm = |v: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; v.len()];
        for r in 0..t {
            let row = &v[r * hidden..(r + 1) * hidden];
            let ms = row.iter().map(|a| a * a).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for i in 0..hidden {
                out[r * hidden + i] = row[i] * inv;
            }
        }
        out
    };
    let (nh, nm) = (norm(h), norm(m));
    let mut cat = Vec::with_capacity(t * 2 * hidden);
    for r in 0..t {
        cat.extend_from_slice(&nh[r * hidden..(r + 1) * hidden]);
        cat.extend_from_slice(&nm[r * hidden..(r + 1) * hidden]);
    }
    let gh = linear(&cat, down, 2 * hidden, rank);
    let forget: Vec<f32> = linear(&gh, forget_w, rank, hidden)
        .iter()
        .map(|v| v.tanh())
        .collect();
    let gain: Vec<f32> = linear(&gh, gain_w, rank, hidden)
        .iter()
        .map(|v| v.tanh())
        .collect();

    let n = h.len();
    let g = OneOp::new(&[n]);
    let (m_in, f_in, g_in) = (g.input(&[n]), g.input(&[n]), g.input(&[n]));
    let got = g.run(
        MessageMixOp::<f32>::default(),
        &[&m_in, &f_in, &g_in],
        &[h, m, &forget, &gain],
        &[],
    );
    assert_matches_reference("message gate", &got, f.get("message_gate.out"), 1e-4);
}
