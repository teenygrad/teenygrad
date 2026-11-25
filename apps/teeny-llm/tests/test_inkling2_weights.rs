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

//! The checkpoint reader, and whether the name map resolves against the real
//! `thinkingmachines/Inkling-2-Mini` index.
//!
//! The weights themselves are 20.2 GB and not needed here: the mechanics are checked
//! against a synthetic two-shard checkpoint, and the names against the shipped
//! `model.safetensors.index.json`, committed as a fixture.

#![cfg(feature = "inkling2")]

use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype, tensor::TensorView};
use teeny_llm::models::inkling2::Inkling2Config;
use teeny_llm::models::inkling2::load::{Source, resolve};
use teeny_llm::models::inkling2::runner::{Quant, checkpoint_name, plan, quant_of};
use teeny_llm::models::inkling2::weights::{Checkpoint, Encoding, pack_rows};

fn fixtures() -> std::path::PathBuf {
    std::env::var("INKLING2_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"))
}

/// Writes a two-shard checkpoint with the dtypes the real one uses.
fn synthetic(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    // Shard 1: a BF16 matrix. Shard 2: packed NVFP4 bytes and their E4M3 scales.
    let bf16: Vec<u8> = (0..8u16)
        .flat_map(|i| half::bf16::from_f32(i as f32 * 0.5).to_le_bytes())
        .collect();
    let packed: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(7)).collect();
    let scales: Vec<u8> = (0..4u8).map(|i| 0x38 + i).collect();

    let one = HashMap::from([(
        "model.llm.layers.0.attn_norm.weight".to_string(),
        TensorView::new(Dtype::BF16, vec![2, 4], &bf16)?,
    )]);
    std::fs::write(
        dir.join("model-00001.safetensors"),
        safetensors::serialize(&one, None)?,
    )?;

    let amax: Vec<u8> = [4.0f32, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let denom: Vec<u8> = 2.0f32.to_le_bytes().to_vec();
    let two = HashMap::from([
        (
            "model.llm.layers.0.mlp.experts.w13_weight".to_string(),
            TensorView::new(Dtype::U8, vec![2, 16], &packed)?,
        ),
        (
            "model.llm.layers.0.mlp.experts.w13_weight_scale".to_string(),
            TensorView::new(Dtype::F8_E4M3, vec![4], &scales)?,
        ),
        (
            "model.llm.layers.0.mlp.experts.w13_weight_amax".to_string(),
            TensorView::new(Dtype::F32, vec![2], &amax)?,
        ),
        (
            "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator".to_string(),
            TensorView::new(Dtype::F32, vec![1], &denom)?,
        ),
    ]);
    std::fs::write(
        dir.join("model-00002.safetensors"),
        safetensors::serialize(&two, None)?,
    )?;

    let index = serde_json::json!({
        "metadata": {},
        "weight_map": {
            "model.llm.layers.0.attn_norm.weight": "model-00001.safetensors",
            "model.llm.layers.0.mlp.experts.w13_weight": "model-00002.safetensors",
            "model.llm.layers.0.mlp.experts.w13_weight_scale": "model-00002.safetensors",
            "model.llm.layers.0.mlp.experts.w13_weight_amax": "model-00002.safetensors",
            "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator": "model-00002.safetensors",
        }
    });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())?;
    Ok(())
}

#[test]
fn test_reads_across_shards_and_keeps_quantised_bytes_packed() {
    let dir = std::env::temp_dir().join(format!("inkling2-ckpt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    synthetic(&dir).expect("writing the synthetic checkpoint");
    let mut ck = Checkpoint::open(&dir).expect("opening it");

    // BF16 is widened to f32 and comes from the first shard.
    let (w, shape, enc) = ck
        .tensor("model.llm.layers.0.attn_norm.weight")
        .expect("norm weight");
    assert_eq!(shape, vec![2, 4]);
    assert_eq!(enc, Encoding::Float);
    assert_eq!(w, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);

    // Packed nibbles come through as byte VALUES, from the second shard, untouched.
    let (p, pshape, penc) = ck
        .tensor("model.llm.layers.0.mlp.experts.w13_weight")
        .expect("w13");
    assert_eq!(pshape, vec![2, 16]);
    assert_eq!(
        penc,
        Encoding::Bytes,
        "quantised weights must not be converted"
    );
    assert_eq!(p.len(), 32);
    let want: Vec<f32> = (0..32u8).map(|i| i.wrapping_mul(7) as f32).collect();
    assert_eq!(p, want, "a packed byte must arrive as its own value");

    // E4M3 scales likewise: 0x38 is 1.0 as a float, but must arrive as 56.
    let (s, _, senc) = ck
        .tensor("model.llm.layers.0.mlp.experts.w13_weight_scale")
        .expect("scale");
    assert_eq!(senc, Encoding::Bytes);
    assert_eq!(
        s[0], 56.0,
        "0x38 must arrive as the byte 56, not as the float 1.0"
    );

    assert!(
        ck.tensor("model.llm.layers.0.nope").is_err(),
        "a missing tensor must error"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_single_file_checkpoints_are_accepted() {
    let dir = std::env::temp_dir().join(format!("inkling2-single-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let data: Vec<u8> = (0..4u16)
        .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
        .collect();
    let one = HashMap::from([(
        "model.llm.norm.weight".to_string(),
        TensorView::new(Dtype::BF16, vec![4], &data).expect("view"),
    )]);
    std::fs::write(
        dir.join("model.safetensors"),
        safetensors::serialize(&one, None).expect("serialize"),
    )
    .expect("write");
    let mut ck = Checkpoint::open(&dir).expect("opening a single-file checkpoint");
    let (v, _, _) = ck.tensor("model.llm.norm.weight").expect("the tensor");
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// What the real checkpoint says a layer contains, pinned.
///
/// The fixture is the shipped `model.safetensors.index.json` of
/// `thinkingmachines/Inkling-2-Mini`, all 1244 tensors. Writing this test is how the
/// port found out that **the layers are not uniform**: there are six distinct layer
/// shapes, and `model.rs` builds only the common one. See the module docs in
/// [`teeny_llm::models::inkling2`] for what that leaves unimplemented.
#[test]
fn test_the_checkpoint_layer_shapes_are_what_the_port_assumes() {
    let text = std::fs::read_to_string(fixtures().join("inkling2-mini-index.json"))
        .expect("the shipped index fixture");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parsing it");
    let names: std::collections::BTreeSet<String> = v["weight_map"]
        .as_object()
        .expect("weight_map")
        .keys()
        .cloned()
        .collect();
    assert_eq!(names.len(), 1244, "the fixture should be the whole index");

    let cfg = Inkling2Config::load(&fixtures().join("inkling2-mini-config.json"))
        .expect("the shipped config")
        .text_config;
    let has = |n: &str| names.contains(n);

    // Every layer has these, and `checkpoint_name` must resolve each one.
    let common = [
        "mlp_norm.weight",
        "attn.wq_du.weight",
        "attn.wk_dv.weight",
        "attn.wv_dv.weight",
        "attn.wo_ud.weight",
        "attn.gate_attn.weight",
        "attn.attn_sink_proj.weight",
        "attn.wr_du.weight",
        "attn.q_l2_clip.scale",
        "attn.rel_l2_clip.scale",
        "attn.rel_logits_proj.weight",
        "attn_sconv.weight",
        "mlp_sconv.weight",
        "mlp.gate.weight",
        "mlp.gate.bias",
        "mlp.experts.weight",
        "mlp.experts_down.weight",
    ];
    for l in 0..cfg.num_hidden_layers {
        for slot in common {
            let name = checkpoint_name(slot, l);
            assert!(has(&name), "every layer should have {name}");
        }
    }

    // DYNAMIC CONVOLUTION LAYERS. The last layer of each six-layer block (5, 11, 17,
    // 23) has no `attn_norm` and no plain `k_sconv.weight`. It carries three separate
    // input norms and a projected convolution instead, which is the reference's
    // Inkling2LoopDynamicShortConvolution. The port implements neither.
    let dynamic: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|l| (l + 1) % cfg.n_layers_per_attn_res_block == 0)
        .collect();
    assert_eq!(dynamic, vec![5, 11, 17, 23]);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.llm.layers.{l}");
        if dynamic.contains(&l) {
            assert!(
                !has(&format!("{p}.attn_norm.weight")),
                "layer {l} should have no attn_norm"
            );
            assert!(
                has(&format!("{p}.q_input_norm.weight")),
                "layer {l} needs q_input_norm"
            );
            assert!(has(&format!("{p}.k_input_norm.weight")));
            assert!(has(&format!("{p}.v_input_norm.weight")));
            assert!(
                has(&format!("{p}.attn.k_sconv.weight_proj")),
                "layer {l} convolves dynamically"
            );
            assert!(has(&format!("{p}.attn.k_sconv.static_w")));
            assert!(!has(&format!("{p}.attn.k_sconv.weight")));
        } else {
            assert!(
                has(&format!("{p}.attn_norm.weight")),
                "layer {l} needs attn_norm"
            );
            assert!(
                has(&format!("{p}.attn.k_sconv.weight")),
                "layer {l} convolves statically"
            );
            assert!(!has(&format!("{p}.q_input_norm.weight")));
        }
    }

    // CONTROLLERS exist only on the looped core, which is what `plan` assumes.
    let l = &cfg.loop_config;
    for layer in 0..cfg.num_hidden_layers {
        let want = (l.core_start..l.core_end).contains(&layer);
        let name = checkpoint_name("controllers.attn.down.weight", layer);
        assert_eq!(has(&name), want, "controllers on layer {layer}");
    }

    // The quantised gathers' parameter slots must resolve, per quantisation.
    for layer in 0..cfg.num_hidden_layers {
        for slot in [
            "mlp.experts.weight",
            "mlp.experts.scale",
            "mlp.experts_down.scale",
        ] {
            let name = checkpoint_name(slot, layer);
            assert!(has(&name), "layer {layer} slot {slot} -> {name}");
        }
        // Only NVFP4 layers have the per-expert amax the scale2 slot maps to.
        let nvfp4 = layer + 4 < cfg.num_hidden_layers;
        assert_eq!(
            has(&checkpoint_name("mlp.experts.scale2", layer)),
            nvfp4,
            "layer {layer} per-expert scale"
        );
    }

    // TWO QUANTISATIONS. The last four layers hold no `_amax` or
    // `_outer_scale_denominator`, which is the MXFP8 group of hf_quant_config; the
    // rest are NVFP4. The expert gather implements NVFP4 only.
    for layer in 0..cfg.num_hidden_layers {
        let nvfp4 = has(&format!(
            "model.llm.layers.{layer}.mlp.experts.w13_weight_amax"
        ));
        assert_eq!(nvfp4, layer < 20, "layer {layer} quantisation");
    }

    // The attn_res queries are per layer, but attention's stop one short of the last
    // layer while the MLP's do not.
    assert!(has("model.llm.attn_res.attn_queries.22"));
    assert!(
        !has("model.llm.attn_res.attn_queries.23"),
        "the last layer has no attn_queries"
    );
    assert!(has("model.llm.attn_res.mlp_queries.23"));
}

/// `param` packs quantised tensors four bytes to an element, per row, and leaves
/// floats alone.
#[test]
fn test_param_packs_quantised_rows_and_passes_floats_through() {
    let dir = std::env::temp_dir().join(format!("inkling2-param-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    synthetic(&dir).expect("writing the synthetic checkpoint");
    let mut ck = Checkpoint::open(&dir).expect("opening it");

    // A float tensor is unchanged.
    let w = ck
        .param("model.llm.layers.0.attn_norm.weight")
        .expect("norm weight");
    assert_eq!(w, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);

    // The packed weights are [2, 16] bytes, so two rows of sixteen become two rows
    // of four words: eight elements, not the thirty-two raw bytes.
    let p = ck
        .param("model.llm.layers.0.mlp.experts.w13_weight")
        .expect("w13");
    assert_eq!(p.len(), 8, "16 bytes a row packs to 4 words a row");
    let raw: Vec<f32> = (0..32u8).map(|i| i.wrapping_mul(7) as f32).collect();
    assert_eq!(
        p,
        pack_rows(&raw, 16),
        "packed per row, not across the whole tensor"
    );

    // And the bytes survive: word 0 holds the first four bytes little-endian.
    let w0 = p[0].to_le_bytes();
    assert_eq!(w0, [0u8, 7, 14, 21], "the first four bytes of row 0");

    let _ = std::fs::remove_dir_all(&dir);
}

/// NVFP4's per-expert scale is `amax / outer_scale_denominator`, which the
/// checkpoint stores as two tensors rather than one.
#[test]
fn test_expert_scale2_divides_amax_by_the_denominator() {
    let dir = std::env::temp_dir().join(format!("inkling2-scale2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    synthetic(&dir).expect("writing the synthetic checkpoint");
    let mut ck = Checkpoint::open(&dir).expect("opening it");

    let s2 = ck
        .expert_scale2("model.llm.layers.0.mlp.experts.w13_weight")
        .expect("the per-expert scale");
    // amax [4, 8] over a denominator of 2.
    assert_eq!(s2, vec![2.0, 4.0], "one scale per expert");

    assert!(
        ck.expert_scale2("model.llm.layers.0.mlp.experts.w2_weight")
            .is_err(),
        "a layer without amax must error rather than invent a scale"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The slots a layer's graph actually produces, as the graph tests observe them.
///
/// Written out rather than built, because building them needs a device; the graph
/// tests in `test_inkling2_model.rs` check this list against a compiled layer.
fn layer_slots(layer: usize, nvfp4: bool, dynamic: bool, looped: bool) -> Vec<String> {
    let mut v: Vec<String> = [
        "mlp_norm.weight",
        "attn.wq_du.weight",
        "attn.wk_dv.weight",
        "attn.wv_dv.weight",
        "attn.wo_ud.weight",
        "attn.gate_attn.weight",
        "attn.attn_sink_proj.weight",
        "attn.wr_du.weight",
        "attn.q_l2_clip.scale",
        "attn.rel_l2_clip.scale",
        "attn.rel_logits_proj.weight",
        "attn_sconv.weight",
        "mlp_sconv.weight",
        "mlp.gate.weight",
        "mlp.gate.bias",
        "mlp.experts.weight",
        "mlp.experts.scale",
        "mlp.experts_down.weight",
        "mlp.experts_down.scale",
        "attn_res.mlp_queries_0.query",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // The query count varies: none on layer 0, three on the dynamic layers, one
    // everywhere else.
    let n_queries = if layer == 0 {
        0
    } else if dynamic {
        3
    } else {
        1
    };
    for q in 0..n_queries {
        v.push(format!("attn_res.attn_queries_{q}.query"));
    }
    if dynamic {
        v.extend(
            [
                "q_input_norm.weight",
                "k_input_norm.weight",
                "v_input_norm.weight",
                "attn.k_sconv.weight_proj.weight",
                "attn.v_sconv.weight_proj.weight",
                "attn.k_sconv.static_w",
                "attn.v_sconv.static_w",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
    } else {
        v.extend(
            [
                "attn_norm.weight",
                "attn.k_sconv.weight",
                "attn.v_sconv.weight",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
    }
    if nvfp4 {
        v.push("mlp.experts.scale2".to_string());
        v.push("mlp.experts_down.scale2".to_string());
    }
    if looped {
        for which in ["attn", "k", "v", "mlp"] {
            for part in ["down", "forget", "gain"] {
                v.push(format!("controllers.{which}.{part}.weight"));
            }
        }
    }
    v
}

/// Every slot of every step of a full plan resolves to a tensor the real checkpoint
/// has. This is the check the first real load would otherwise perform, one failure
/// at a time, somewhere inside a step.
#[test]
fn test_a_whole_plan_resolves_against_the_real_index() {
    let text = std::fs::read_to_string(fixtures().join("inkling2-mini-index.json"))
        .expect("the shipped index fixture");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parsing it");
    let names: std::collections::BTreeSet<String> = v["weight_map"]
        .as_object()
        .expect("weight_map")
        .keys()
        .cloned()
        .collect();

    let cfg = Inkling2Config::load(&fixtures().join("inkling2-mini-config.json"))
        .expect("the shipped config")
        .text_config;

    let mut missing: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for step in plan(&cfg) {
        let dynamic = (step.layer + 1) % cfg.n_layers_per_attn_res_block == 0;
        let nvfp4 = quant_of(&cfg, step.layer) == Quant::Nvfp4;
        let slots = layer_slots(step.layer, nvfp4, dynamic, step.pass.is_some());
        for (slot, src) in resolve(&slots, step.layer) {
            checked += 1;
            let needed: Vec<String> = match src {
                Source::Tensor(n) | Source::Transposed(n, _) => vec![n],
                Source::ExpertScale(p) => {
                    vec![format!("{p}_amax"), format!("{p}_outer_scale_denominator")]
                }
                // The folded scale needs the block scale and both halves of the
                // per-expert one, since the loader multiplies them together.
                Source::FoldedScale(p) => vec![
                    format!("{p}_scale"),
                    format!("{p}_amax"),
                    format!("{p}_outer_scale_denominator"),
                ],
            };
            for n in needed {
                if !names.contains(&n) {
                    missing.push(format!("layer {} slot {slot} -> {n}", step.layer));
                }
            }
        }
    }
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "{} of {checked} slots do not resolve, e.g.:\n  {}",
        missing.len(),
        missing
            .iter()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    assert!(
        checked > 1500,
        "a 48-step plan should check well over 1500 slots, got {checked}"
    );

    // The attn_queries of the last layer do not exist, so a plan that reached it with
    // three attention queries would be wrong. Confirm the plan never does.
    // 23 attention queries for layers 1..=23, and 24 mlp queries for every layer.
    assert!(!names.contains("model.llm.attn_res.attn_queries.23"));
    assert!(names.contains("model.llm.attn_res.attn_queries.22"));
    assert!(names.contains("model.llm.attn_res.mlp_queries.23"));
    assert_eq!(
        checkpoint_name("attn_res.attn_queries_0.query", 23),
        "model.llm.attn_res.attn_queries.22",
        "the last layer reads the last query, not one past the end"
    );
    assert_eq!(
        checkpoint_name("attn_res.mlp_queries_0.query", 23),
        "model.llm.attn_res.mlp_queries.23",
        "the mlp queries are not offset"
    );
}

/// The same whole-plan resolution, but against a real checkpoint on disk.
///
/// Ignored by default: it needs the 20.2 GB download. Point `INKLING2_MODEL_DIR` at
/// it and run with `--ignored`. This checks what the index fixture cannot: that the
/// shard actually holds each tensor, and that what the loader produces is the size
/// the kernels declare.
#[test]
#[ignore = "needs the 20.2 GB checkpoint; set INKLING2_MODEL_DIR"]
fn test_the_real_checkpoint_satisfies_a_whole_plan() {
    let dir = std::env::var("INKLING2_MODEL_DIR")
        .expect("set INKLING2_MODEL_DIR to the checkpoint directory");
    let dir = Path::new(&dir);
    let cfg = Inkling2Config::load(&dir.join("config.json"))
        .expect("the checkpoint's own config")
        .text_config;
    let mut ck = Checkpoint::open(dir).expect("opening the checkpoint");

    let mut absent: Vec<String> = Vec::new();
    let mut slots_checked = 0usize;
    for step in plan(&cfg) {
        let dynamic = (step.layer + 1) % cfg.n_layers_per_attn_res_block == 0;
        let nvfp4 = quant_of(&cfg, step.layer) == Quant::Nvfp4;
        let slots = layer_slots(step.layer, nvfp4, dynamic, step.pass.is_some());
        slots_checked += slots.len();
        for (slot, name) in teeny_llm::models::inkling2::load::missing(&ck, &slots, step.layer)
        {
            absent.push(format!("layer {} slot {slot} -> {name}", step.layer));
        }
    }
    absent.sort();
    absent.dedup();
    assert!(
        absent.is_empty(),
        "{} of {slots_checked} slots are absent from the checkpoint:\n  {}",
        absent.len(),
        absent
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    // Read one layer for real and check the packed sizes against what the gather
    // declares: experts * n * ceil(row / 4).
    let (e, inter, hid) = (cfg.n_routed_experts, cfg.intermediate_size, cfg.hidden_size);
    let w13 = ck
        .param(&checkpoint_name("mlp.experts.weight", 3))
        .expect("layer 3 expert weights");
    assert_eq!(
        w13.len(),
        e * (2 * inter) * (hid / 2).div_ceil(4),
        "packed w13 is experts * 2*inter * ceil(hidden/2 / 4) elements"
    );
    let s2 = ck
        .expert_scale2(&checkpoint_name("mlp.experts.scale2", 3).replace("_amax", ""))
        .expect("layer 3 per-expert scale");
    assert_eq!(s2.len(), e, "one scale per expert");
    assert!(
        s2.iter().all(|v| v.is_finite() && *v > 0.0),
        "the per-expert scales must be finite and positive"
    );
}

/// A converted checkpoint from a different format version is refused.
///
/// The converted tensors are a cache of how the loader packed and folded the
/// originals. If that changes and the cache does not, nothing downstream can tell:
/// the shapes still fit and the model just produces wrong output. The marker carries
/// the format version so the mismatch is an error at open.
#[test]
fn test_a_stale_converted_checkpoint_is_refused() {
    use safetensors::tensor::{Dtype, TensorView};
    use teeny_llm::models::inkling2::weights::{CONVERTED_FORMAT, CONVERTED_MARKER};

    let dir = std::env::temp_dir().join(format!("inkling2-stale-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    let write = |version: f32| {
        let bytes: Vec<u8> = version.to_le_bytes().to_vec();
        let mut t = std::collections::HashMap::new();
        t.insert(
            CONVERTED_MARKER.to_string(),
            TensorView::new(Dtype::F32, vec![1], &bytes).expect("view"),
        );
        std::fs::write(
            dir.join("model.safetensors"),
            safetensors::serialize(&t, None).expect("serialize"),
        )
        .expect("write");
    };

    write(CONVERTED_FORMAT as f32);
    assert!(
        Checkpoint::open(&dir).is_ok(),
        "the current format must open"
    );

    write(CONVERTED_FORMAT as f32 + 1.0);
    let msg = match Checkpoint::open(&dir) {
        Ok(_) => panic!("a newer format must be refused"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("inkling2_convert"),
        "the error should say how to fix it, got: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
