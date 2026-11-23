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
use teeny_llm_core::models::inkling2::Inkling2Config;
use teeny_llm_core::models::inkling2::runner::checkpoint_name;
use teeny_llm_core::models::inkling2::weights::{Checkpoint, Encoding};

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

    let two = HashMap::from([
        (
            "model.llm.layers.0.mlp.experts.w13_weight".to_string(),
            TensorView::new(Dtype::U8, vec![2, 16], &packed)?,
        ),
        (
            "model.llm.layers.0.mlp.experts.w13_weight_scale".to_string(),
            TensorView::new(Dtype::F8_E4M3, vec![4], &scales)?,
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
/// [`teeny_llm_core::models::inkling2`] for what that leaves unimplemented.
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
