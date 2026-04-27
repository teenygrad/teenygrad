/*
 * Copyright (c) 2026 Teenygrad.
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

//! End-to-end MNIST inference demo.
//!
//! Steps performed:
//!   1. Download the HuggingFace MNIST parquet files (cached in TEENY_DATA_DIR).
//!   2. Read a batch of BATCH_SIZE training images from the parquet file.
//!   3. Trace the LeNet-5 (valid-conv, no-bias) graph with SymTensor.
//!   4. Compile every kernel to PTX via CudaGraphCompiler.
//!   5. Load compiled kernels + pre-allocate weight buffers on the device.
//!   6. Initialise all weights with Kaiming-uniform and upload to device.
//!   7. Copy the image batch to device memory.
//!   8. Run LoadedModel::forward().
//!   9. Copy logits back to host and print argmax predictions vs. ground truth.
//!
//! Required environment variables:
//!   TEENY_RUSTC_PATH  — path to the teenygrad rustc binary used by LlvmCompiler
//!   TEENY_DATA_DIR    — (optional) cache dir for MNIST parquet  [/tmp/teenygrad_cache]
//!   TEENY_CACHE_DIR   — (optional) cache dir for compiled PTX   [/tmp/teenygrad_rustc]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::{
    graph::{DtypeRepr, SymTensor},
    nn::Layer,
};
use teeny_core::device::context::Context as DeviceContext;
use teeny_cuda::{
    compiler::{graph::CudaGraphCompiler, target::capability_from_device_info},
    device::{context::Cuda, mem},
    model::TensorRef,
};
use teeny_kernels::graph::TritonLowering;
use teeny_vision::mnist::{MnistDataset, mnist_valid};
use tokio::io::AsyncWriteExt;

const BATCH_SIZE: usize = 32;

#[tokio::main]
async fn main() -> Result<()> {
    // ── 1. Download MNIST parquet files ──────────────────────────────────────
    let data_dir = PathBuf::from(
        env::var("TEENY_DATA_DIR").unwrap_or_else(|_| "/tmp/teenygrad_cache".to_string()),
    );
    download_data(&data_dir).await?;
    let train_path = data_dir.join("mnist/mnist-train.parquet");

    // ── 2. Read a batch of training images ───────────────────────────────────
    println!("[1/7] reading {BATCH_SIZE} training samples …");
    let dataset = MnistDataset::open(&train_path)
        .with_context(|| format!("failed to open {}", train_path.display()))?;
    println!("      dataset: {} samples", dataset.len());

    let batch = dataset.read_batch(0, BATCH_SIZE)?;
    let (pmin, pmax) = batch.images.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(lo, hi), &v| (lo.min(v), hi.max(v)),
    );
    println!("      labels[0..8]: {:?}", &batch.labels[..8]);
    println!("      pixel range : [{pmin:.4}, {pmax:.4}]");

    // ── 3. Initialise CUDA device ─────────────────────────────────────────────
    // Cuda::try_new() calls cuInit; then device() opens the context.
    println!("[2/7] initialising CUDA …");
    let cuda = Cuda::try_new()?;
    let device_infos = cuda.list_devices()?;
    let device = cuda.device(&device_infos[0].id)?;
    let capability = capability_from_device_info(&device.info)?;
    println!("      device: {} ({})", device.info.name, capability);

    // ── 4. Trace LeNet-5 (valid-conv, no-bias) ────────────────────────────────
    println!("[3/7] tracing LeNet-5 (valid-conv) graph …");
    let (input, graph) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(1), Some(28), Some(28)]);
    let _output = Layer::call(&mnist_valid::<f32>(), input);
    let graph = graph.borrow();
    println!("      {} nodes", graph.nodes.len());

    // ── 5. Compile kernels to PTX ─────────────────────────────────────────────
    println!("[4/7] compiling kernels …");
    let rustc_path = env::var("TEENY_RUSTC_PATH")
        .context("TEENY_RUSTC_PATH must point to the teenygrad rustc binary")?;
    let ptx_cache =
        env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());

    let compiler = LlvmCompiler::new(rustc_path, ptx_cache)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let target = Target::new(capability);
    let lowering = TritonLowering::new();
    let cuda_model = graph_compiler.compile_model(&graph, &lowering, &target, false)?;
    println!("      compiled {} DAG nodes", cuda_model.dag.len());

    // ── 6. Load + initialise weights ─────────────────────────────────────────
    println!("[5/7] loading model + uploading Kaiming-uniform weights …");
    let mut loaded = cuda_model.load(&device, BATCH_SIZE)?;

    // param_info() yields (node_idx, &[shape_per_slot]) for every parameterised node.
    let param_summary: Vec<(usize, Vec<Vec<usize>>)> = loaded
        .param_info()
        .map(|(idx, shapes)| (idx, shapes.to_vec()))
        .collect();

    for (node_idx, shapes) in &param_summary {
        for (param_idx, shape) in shapes.iter().enumerate() {
            let data = kaiming_uniform(shape);
            loaded.load_param_f32(*node_idx, param_idx, &data)?;
            println!(
                "      node {node_idx} param {param_idx}: shape {:?} ({} floats)",
                shape,
                data.len()
            );
        }
    }

    // ── 7. Copy input batch to device ─────────────────────────────────────────
    println!("[6/7] running forward pass …");
    let img_bytes = batch.images.len() * std::mem::size_of::<f32>();
    let img_ptr = mem::alloc(img_bytes)?;
    // SAFETY: img_ptr was just allocated with the right size.
    unsafe { mem::copy_h_to_d(img_ptr, batch.images.as_ptr(), batch.images.len()) }?;

    let input_ref = TensorRef::new(img_ptr, vec![BATCH_SIZE, 1, 28, 28]);

    let output = loaded.forward(&device, BATCH_SIZE, &[input_ref])?;

    // ── 8. Copy logits back and evaluate ─────────────────────────────────────
    println!("[7/7] copying results …");
    let n_logits = output.shape.iter().product::<usize>();
    let mut logits = vec![0.0f32; n_logits];
    // SAFETY: logits has the right capacity; output.ptr is a valid device alloc.
    unsafe { mem::copy_d_to_h(logits.as_mut_ptr(), output.ptr, n_logits) }?;

    // Free the caller-owned output buffer and the input buffer.
    mem::free(output.ptr)?;
    mem::free(img_ptr)?;

    // ── 9. Print predictions ──────────────────────────────────────────────────
    println!();
    println!("  idx  label  pred   logits (first 4 classes)");
    println!("  ───  ─────  ────   ─────────────────────────");
    let mut correct = 0usize;
    for i in 0..BATCH_SIZE {
        let row = &logits[i * 10..(i + 1) * 10];
        let pred = argmax(row);
        let label = batch.labels[i] as usize;
        if pred == label { correct += 1; }
        if i < 16 {
            println!(
                "  {:>3}  {:>5}  {:>4}   [{:.3}, {:.3}, {:.3}, {:.3}]",
                i, label, pred, row[0], row[1], row[2], row[3],
            );
        }
    }
    println!();
    println!("  accuracy: {correct}/{BATCH_SIZE} ({:.1}%)", 100.0 * correct as f32 / BATCH_SIZE as f32);
    println!();
    println!("Done — forward pass completed successfully.");
    Ok(())
}

// ── Weight initialisation ─────────────────────────────────────────────────────

/// Kaiming-uniform initialisation for a weight tensor.
///
/// `fan_in` = product of all dimensions except the first (output-channels/features):
///   - Conv weight `[C_out, C_in, KH, KW]` → fan_in = C_in * KH * KW
///   - Linear weight `[N, K]`               → fan_in = K
///
/// `bound = sqrt(6 / fan_in)`, values drawn uniformly from `[-bound, bound]`.
fn kaiming_uniform(shape: &[usize]) -> Vec<f32> {
    let fan_in: usize = if shape.len() >= 2 {
        shape[1..].iter().product()
    } else {
        shape[0]
    };
    let bound = (6.0_f32 / fan_in as f32).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = rand::rng();
    (0..n)
        .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Download helpers ──────────────────────────────────────────────────────────

async fn download_data(cache_dir: &Path) -> Result<()> {
    const TEST_URL: &str = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true";
    const TRAIN_URL: &str = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true";

    download_if_not_exists(TEST_URL,  &cache_dir.join("mnist/mnist-test.parquet")).await?;
    download_if_not_exists(TRAIN_URL, &cache_dir.join("mnist/mnist-train.parquet")).await?;
    Ok(())
}

async fn download_if_not_exists(url: &str, path: &Path) -> Result<()> {
    if path.exists() { return Ok(()); }

    let mut response = reqwest::get(url).await?;
    response.error_for_status_ref()?;

    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }

    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("download");
    let total = response.content_length().unwrap_or(0);
    let bar = if total > 0 {
        let b = ProgressBar::new(total);
        b.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )?.progress_chars("##-"),
        );
        b
    } else {
        let b = ProgressBar::new_spinner();
        b.set_style(ProgressStyle::with_template("{msg} {spinner} {bytes} downloaded")?);
        b.enable_steady_tick(std::time::Duration::from_millis(120));
        b
    };
    bar.set_message(format!("Downloading {file_name}"));

    let mut file = tokio::fs::File::create(path).await?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
        bar.inc(chunk.len() as u64);
    }
    file.flush().await?;
    bar.finish_with_message(format!("Downloaded {file_name}"));
    Ok(())
}
