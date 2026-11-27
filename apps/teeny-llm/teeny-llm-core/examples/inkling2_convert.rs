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
use teeny_llm_core::kernels::linear::{I8_LANES, int8_name, pack_bf16_halves, quantize_lanes_i8};
use teeny_llm_core::models::inkling2::Inkling2Config;
use teeny_llm_core::models::inkling2::kernels::expert_gemv_nvfp4::{QBLOCK, byte_scale_rows};
use teeny_llm_core::models::inkling2::kernels::{expert_gemv_mxfp8, expert_gemv_nvfp4};
use teeny_llm_core::models::inkling2::load::{bf16_name, byte_scale_name};
use teeny_llm_core::models::inkling2::runner::{Quant, quant_of};
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
    /// Where to write the converted checkpoint.
    #[arg(long, required_unless_present_any = ["add_int8", "add_compact"])]
    out: Option<PathBuf>,
    /// Instead of converting, add the int8 linear weights to the converted
    /// checkpoint at `--model`, in place, so loading reads them rather than
    /// quantising at every start.
    #[arg(long)]
    add_int8: bool,
    /// Instead of converting, add the compact tensors to the converted checkpoint
    /// at this path, in place, reading them from the original at `--model`: the
    /// NVFP4 experts' E4M3 block scale bytes, which conversion folded into f32.
    #[arg(long)]
    add_compact: Option<PathBuf>,
}

/// Writes `converted-compact.safetensors` into the converted checkpoint `dir`, from
/// the original at `original`: every NVFP4 projection's [`byte_scale_rows`] buffer,
/// named [`byte_scale_name`]. The existing tensors are untouched, so builds that do
/// not know these names read the checkpoint as before.
fn add_compact(original: &std::path::Path, dir: &std::path::Path) -> anyhow::Result<()> {
    let cfg = Inkling2Config::load(&original.join("config.json"))?.text_config;
    let mut orig = Checkpoint::open(original)?;
    anyhow::ensure!(!orig.is_converted(), "{} is converted; pass the original", original.display());
    let mut conv = Checkpoint::open(dir)?;
    anyhow::ensure!(conv.is_converted(), "{} is not a converted checkpoint", dir.display());
    let layer_of = |n: &str| -> Option<usize> {
        n.strip_prefix("model.llm.layers.")
            .and_then(|r| r.split('.').next())
            .and_then(|d| d.parse().ok())
    };
    let mut names: Vec<String> = orig
        .names()
        .filter(|n| n.contains(".mlp.experts.") && n.ends_with("_weight_scale"))
        .filter(|n| layer_of(n).is_some_and(|l| quant_of(&cfg, l) == Quant::Nvfp4))
        .map(str::to_string)
        .collect();
    names.sort();
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for name in &names {
        let prefix = name.trim_end_matches("_scale");
        let (bytes, shape, _) = orig.tensor(name)?;
        let [e, n, blocks] = shape[..] else {
            anyhow::bail!("{name} is {shape:?}, expected [experts, rows, blocks]")
        };
        let scale2 = orig.expert_scale2(prefix)?;
        let buf = byte_scale_rows(&bytes, &scale2, e, n, blocks * QBLOCK);
        owned.push((byte_scale_name(prefix), vec![buf.len()], bytes_of(&buf)));
    }
    // The token embedding as bf16 pairs, half its f32 bytes.
    let embed = "model.llm.embed.weight";
    let (values, shape, _) = conv.tensor(embed)?;
    let [vocab, hidden] = shape[..] else {
        anyhow::bail!("{embed} is {shape:?}, expected [vocab, hidden]")
    };
    let packed = pack_bf16_halves(&values, vocab, hidden).map_err(|e| anyhow::anyhow!("{embed}: {e}"))?;
    owned.push((bf16_name(embed), vec![vocab, hidden / 2], bytes_of(&packed)));
    let shard = "converted-compact.safetensors";
    write_extra_shard(dir, shard, &owned)?;
    let bytes: usize = owned.iter().map(|(_, _, b)| b.len()).sum();
    println!("added {} compact tensors, {:.2} GB, to {}", owned.len(), bytes as f64 / 1e9, dir.display());
    Ok(())
}

/// Writes `owned` as the shard `shard` of the checkpoint at `dir` and adds its
/// tensors to the index.
fn write_extra_shard(dir: &std::path::Path, shard: &str, owned: &[(String, Vec<usize>, Vec<u8>)]) -> anyhow::Result<()> {
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
    std::fs::write(
        dir.join(shard),
        safetensors::serialize(&views, None).map_err(|e| anyhow::anyhow!("serialising {shard}: {e}"))?,
    )?;
    let index_path = dir.join("model.safetensors.index.json");
    let mut idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
    let map = idx["weight_map"]
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{} has no weight_map", index_path.display()))?;
    for (n, _, _) in owned {
        map.insert(n.clone(), serde_json::Value::String(shard.to_string()));
    }
    std::fs::write(&index_path, serde_json::to_string_pretty(&idx)?)?;
    Ok(())
}


/// Writes `converted-int8.safetensors` into a converted checkpoint: for every 2-D
/// linear weight the lanes kernel can take, the buffer [`quantize_lanes_i8`] gives,
/// named [`int8_name`]. The existing tensors are untouched, so a build that does
/// not know these names reads the checkpoint as before.
fn add_int8(dir: &std::path::Path) -> anyhow::Result<()> {
    let mut ck = Checkpoint::open(dir)?;
    anyhow::ensure!(ck.is_converted(), "{} is not a converted checkpoint", dir.display());
    let mut names: Vec<String> = ck
        .names()
        .filter(|n| n.starts_with("model.llm.") && n.ends_with(".weight"))
        .filter(|n| !n.contains(".mlp.experts") && *n != "model.llm.embed.weight")
        .map(str::to_string)
        .collect();
    names.sort();
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for name in &names {
        let (values, shape, _) = ck.tensor(name)?;
        let [n, k] = shape[..] else { continue };
        if n % I8_LANES != 0 || k % 4 != 0 {
            continue;
        }
        let buf = quantize_lanes_i8(&values, n, k).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        owned.push((int8_name(name), vec![buf.len()], bytes_of(&buf)));
    }
    // Named by lane width, so a build reading another width keeps its own shard.
    write_extra_shard(dir, &format!("converted-int8x{I8_LANES}.safetensors"), &owned)?;
    let bytes: usize = owned.iter().map(|(_, _, b)| b.len()).sum();
    println!("added {} int8 linears, {:.2} GB, to {}", owned.len(), bytes as f64 / 1e9, dir.display());
    Ok(())
}

/// Bytes of an f32 slice, little-endian, for safetensors.
fn bytes_of(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.add_int8 {
        return add_int8(&args.model);
    }
    if let Some(dir) = &args.add_compact {
        return add_compact(&args.model, dir);
    }
    let out_dir = args.out.clone().expect("clap requires --out without --add-int8");
    let cfg = Inkling2Config::load(&args.model.join("config.json"))?.text_config;
    let mut ck = Checkpoint::open(&args.model)?;
    std::fs::create_dir_all(&out_dir)?;

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
            let quant = layer_of(name).map(|l| quant_of(&cfg, l));
            let is_expert = name.contains(".mlp.experts.");
            let is_expert_weight =
                is_expert && (name.ends_with("w13_weight") || name.ends_with("w2_weight"));
            let is_expert_scale = is_expert && name.ends_with("_weight_scale");
            let (values, shape, _) = ck.tensor(name)?;
            let quantised = matches!(
                ck.encoding_of(name)?,
                teeny_llm_core::models::inkling2::weights::Encoding::Bytes
            );
            // [experts, rows, ...]: the rows-as-lanes projections interleave by rows.
            let (experts, rows) = (shape.first().copied().unwrap_or(1), shape.get(1).copied().unwrap_or(1));
            let row_len = *shape.last().unwrap_or(&values.len());

            let out: Vec<f32> = match quant {
                // NVFP4: packed nibble bytes and folded, halved scales, row-interleaved
                // as ExpertGemvNvfp4Op reads them.
                Some(Quant::Nvfp4) if is_expert_scale => {
                    let prefix = name.trim_end_matches("_scale");
                    let s2 = ck.expert_scale2(prefix)?;
                    let per = values.len() / s2.len();
                    let folded: Vec<f32> = values
                        .iter()
                        .enumerate()
                        .map(|(i, b)| e4m3_to_f32(*b as u8) * s2[i / per] * 0.5)
                        .collect();
                    expert_gemv_nvfp4::interleave_rows(&folded, experts, rows, row_len, expert_gemv_nvfp4::ROWS)
                }
                Some(Quant::Nvfp4) if is_expert_weight => expert_gemv_nvfp4::interleave_row_bytes(
                    &pack_rows(&values, row_len),
                    experts,
                    rows,
                    row_len,
                    expert_gemv_nvfp4::ROWS,
                ),
                // MXFP8: weight and E8M0 scale bytes alike, byte-interleaved for
                // ExpertGemvMxfp8Op.
                Some(Quant::Mxfp8) if is_expert_weight || is_expert_scale => {
                    expert_gemv_nvfp4::interleave_row_bytes(
                        &pack_rows(&values, row_len),
                        experts,
                        rows,
                        row_len,
                        expert_gemv_mxfp8::ROWS,
                    )
                }
                // `tensor` has already turned bf16 and f32 into f32; any other
                // quantised tensor is repacked four bytes to an element.
                _ if quantised => pack_rows(&values, row_len),
                _ => values,
            };
            let out_shape = if quantised || is_expert_scale {
                vec![out.len()]
            } else {
                shape
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
        std::fs::write(out_dir.join(shard), blob)?;
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
        TensorView::new(Dtype::F32, vec![1], &bytes).map_err(|e| anyhow::anyhow!("marker: {e}"))?,
    );
    std::fs::write(
        out_dir.join("converted-marker.safetensors"),
        safetensors::serialize(&marker, None).map_err(|e| anyhow::anyhow!("marker: {e}"))?,
    )?;
    index.insert(
        CONVERTED_MARKER.to_string(),
        "converted-marker.safetensors".to_string(),
    );

    let idx = serde_json::json!({ "weight_map": index });
    std::fs::write(
        out_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&idx)?,
    )?;
    for f in [
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ] {
        let src = args.model.join(f);
        if src.exists() {
            std::fs::copy(&src, out_dir.join(f))?;
        }
    }
    println!(
        "wrote {:.1} GB (format {CONVERTED_FORMAT}) to {}",
        total as f64 / 1e9,
        out_dir.display()
    );
    Ok(())
}
