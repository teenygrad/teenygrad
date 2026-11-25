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

//! Converts an Inkling-2 checkpoint into the form the engine actually loads, once,
//! so that every later load is a read rather than a conversion.
//!
//! Loading does three expensive things to the shipped checkpoint: it expands bf16 to
//! f32, repacks the quantised tensors four bytes to an f32 container, and folds
//! NVFP4's E4M3 block scale together with its per-expert `amax / denominator`. None
//! of that depends on anything but the checkpoint, so it can be done ahead of time.
//!
//! The result is bigger on disk than the original -- the folded scales cost four
//! bytes where the checkpoint spends one, and bf16 doubles -- and holds only the
//! tensors the text model reads, so the vision tower and the multi-token-prediction
//! head are dropped.
//!
//! Usage: `inkling2_convert --model <checkpoint> --out <dir>`

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use safetensors::tensor::{Dtype, TensorView};
use teeny_llm_core::models::inkling2::Inkling2Config;
use teeny_llm_core::models::inkling2::weights::{
    CONVERTED_FORMAT, CONVERTED_MARKER, Checkpoint, e4m3_to_f32, pack_rows,
};

#[derive(Parser, Debug)]
#[command(about = "Convert an Inkling-2 checkpoint to the engine's load-time form")]
struct Args {
    /// The checkpoint folder to read.
    #[arg(long)]
    model: PathBuf,
    /// Where to write the converted checkpoint.
    #[arg(long)]
    out: PathBuf,
}

/// Bytes of an f32 slice, little-endian, for safetensors.
fn bytes_of(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = Inkling2Config::load(&args.model.join("config.json"))?.text_config;
    let mut ck = Checkpoint::open(&args.model)?;
    std::fs::create_dir_all(&args.out)?;

    // Only what the text model reads.
    let mut names: Vec<String> = ck
        .names()
        .filter(|n| n.starts_with("model.llm."))
        .map(str::to_string)
        .collect();
    names.sort();
    println!("{} tensors to convert", names.len());

    // Group by layer so no shard needs more than one layer in memory at a time.
    let layer_of = |n: &str| -> Option<usize> {
        n.strip_prefix("model.llm.layers.")
            .and_then(|r| r.split('.').next())
            .and_then(|d| d.parse().ok())
    };
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for n in &names {
        let shard = match layer_of(n) {
            Some(l) => format!("converted-layer{l:02}.safetensors"),
            None => "converted-top.safetensors".to_string(),
        };
        groups.entry(shard).or_default().push(n.clone());
    }

    let mut index: HashMap<String, String> = HashMap::new();
    let mut total = 0u64;
    let mut shards: Vec<String> = groups.keys().cloned().collect();
    shards.sort();

    for shard in &shards {
        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for name in &groups[shard] {
            let layer = layer_of(name);
            let nvfp4 = layer.is_some_and(|l| {
                matches!(
                    teeny_llm_core::models::inkling2::runner::quant_of(&cfg, l),
                    teeny_llm_core::models::inkling2::runner::Quant::Nvfp4
                )
            });
            let is_expert_scale = name.ends_with("_weight_scale");
            let (values, shape, _) = ck.tensor(name)?;

            let (out, out_shape) = if is_expert_scale && nvfp4 {
                // Fold the block scale with the per-expert scale.
                let prefix = name.trim_end_matches("_scale");
                let s2 = ck.expert_scale2(prefix)?;
                let per = values.len() / s2.len();
                let folded: Vec<f32> = values
                    .iter()
                    .enumerate()
                    .map(|(i, b)| e4m3_to_f32(*b as u8) * s2[i / per])
                    .collect();
                let n = folded.len();
                (folded, vec![n])
            } else {
                // `tensor` has already turned bf16 and f32 into f32; a quantised
                // tensor comes back as one byte per element and is repacked here.
                let quantised = matches!(
                    ck.encoding_of(name)?,
                    teeny_llm_core::models::inkling2::weights::Encoding::Bytes
                );
                if quantised {
                    let row = *shape.last().unwrap_or(&values.len());
                    let packed = pack_rows(&values, row);
                    let n = packed.len();
                    (packed, vec![n])
                } else {
                    (values, shape)
                }
            };
            total += (out.len() * 4) as u64;
            owned.push((name.clone(), out_shape, bytes_of(&out)));
            index.insert(name.clone(), shard.clone());
        }

        let views: HashMap<String, TensorView> = owned
            .iter()
            .map(|(n, shape, bytes)| {
                Ok((
                    n.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes)
                        .map_err(|e| anyhow::anyhow!("view for {n}: {e}"))?,
                ))
            })
            .collect::<anyhow::Result<_>>()?;
        let blob = safetensors::serialize(&views, None)
            .map_err(|e| anyhow::anyhow!("serialising {shard}: {e}"))?;
        std::fs::write(args.out.join(shard), blob)?;
        println!("  {shard}: {} tensors", owned.len());
    }

    // The marker tells the loader not to pack, fold or expand again, and which
    // format it was written in, so a later change to any of that is an error rather
    // than silently wrong output.
    let version = [CONVERTED_FORMAT as f32];
    let bytes = bytes_of(&version);
    let mut marker: HashMap<String, TensorView> = HashMap::new();
    marker.insert(
        CONVERTED_MARKER.to_string(),
        TensorView::new(Dtype::F32, vec![1], &bytes)
            .map_err(|e| anyhow::anyhow!("marker: {e}"))?,
    );
    std::fs::write(
        args.out.join("converted-marker.safetensors"),
        safetensors::serialize(&marker, None).map_err(|e| anyhow::anyhow!("marker: {e}"))?,
    )?;
    index.insert(
        CONVERTED_MARKER.to_string(),
        "converted-marker.safetensors".to_string(),
    );

    let idx = serde_json::json!({ "weight_map": index });
    std::fs::write(
        args.out.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&idx)?,
    )?;
    for f in ["config.json", "tokenizer.json", "chat_template.jinja", "special_tokens_map.json"] {
        let src = args.model.join(f);
        if src.exists() {
            std::fs::copy(&src, args.out.join(f))?;
        }
    }
    println!(
        "wrote {:.1} GB (format {CONVERTED_FORMAT}) to {}",
        total as f64 / 1e9,
        args.out.display()
    );
    Ok(())
}
