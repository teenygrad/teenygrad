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

//! End-to-end MNIST training + evaluation demo.
//!
//! Architecture: 3-layer MLP (784→256→64→10, ReLU activations, no bias).
//! Loss:         cross-entropy (numerically stable log-sum-exp variant).
//! Optimiser:    AdamW with decoupled weight decay.
//!
//! Steps performed:
//!   1. Download the HuggingFace MNIST parquet files (cached in TEENY_DATA_DIR).
//!   2. Load the full train/test sets into host memory (PNG decode, one-time cost).
//!   3. Trace + compile the MLP graph to PTX via CudaGraphCompiler.
//!   4. Compile standalone CE-loss and AdamW PTX kernels.
//!   5. Initialise weights with Kaiming-uniform and upload to device.
//!   6. Train for N_EPOCHS, printing loss every 200 steps.
//!   7. Evaluate on the test set and print final accuracy.
//!
//! Required environment variables:
//!   TEENY_RUSTC_PATH  — path to the teenygrad rustc binary used by LlvmCompiler
//!   TEENY_DATA_DIR    — (optional) cache dir for MNIST parquet  [/tmp/teenygrad_cache]
//!   TEENY_CACHE_DIR   — (optional) cache dir for compiled PTX   [/tmp/teenygrad_rustc]

use std::env;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_compiler::compiler::driver::cuda::compile_kernel;
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::{
    device::{context::Context as DeviceContext, program::ArgVisitor},
    graph::{DtypeRepr, SymTensor},
    nn::Layer,
};
use teeny_cuda::{
    compiler::{graph::CudaGraphCompiler, target::capability_from_device_info},
    device::{CudaArgPacker, CudaLaunchConfig, context::Cuda, mem, program::{CudaProgram, ErasedKernel}},
    model::{AdamwKernel, TensorRef},
};
use teeny_kernels::{
    graph::TritonLowering,
    nn::{
        loss::nll::{CrossEntropyLossBackward, CrossEntropyLossForward},
        optim::adam::AdamwStep,
    },
};
use teeny_vision::mnist::{MnistDataset, mnist_mlp};
use tokio::io::AsyncWriteExt;

// ── Hyper-parameters ──────────────────────────────────────────────────────────

const BATCH_SIZE: usize = 64;
const N_EPOCHS: usize = 5;
const N_CLASSES: usize = 10;

/// `next_power_of_two(N_CLASSES)` — required by the CE loss kernel.
const CE_BLOCK_SIZE: i32 = 16;
/// Threads per CTA for the CE loss kernels (determined at PTX compile time).
const CE_PTX_THREADS: u32 = 128;
/// Block size for the AdamW kernel.
const ADAMW_BLOCK_SIZE: i32 = 1024;

const LR: f32 = 1e-3;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const WEIGHT_DECAY: f32 = 1e-2;

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // ── 1. Download MNIST parquet files ──────────────────────────────────────
    let data_dir = PathBuf::from(
        env::var("TEENY_DATA_DIR").unwrap_or_else(|_| "/tmp/teenygrad_cache".to_string()),
    );
    download_data(&data_dir).await?;
    let train_path = data_dir.join("mnist/mnist-train.parquet");
    let test_path  = data_dir.join("mnist/mnist-test.parquet");

    // ── 2. Load all data into host memory ─────────────────────────────────────
    println!("[1/8] loading MNIST (PNG decode) …");
    let train_ds = MnistDataset::open(&train_path)
        .with_context(|| format!("failed to open {}", train_path.display()))?;
    let test_ds = MnistDataset::open(&test_path)
        .with_context(|| format!("failed to open {}", test_path.display()))?;
    println!("      train: {} samples", train_ds.len());
    println!("      test : {} samples", test_ds.len());

    let train_all = train_ds.read_batch(0, train_ds.len())?;
    let test_all  = test_ds.read_batch(0, test_ds.len())?;

    // Round down to the nearest complete batch.
    let n_train_batches = train_all.batch_size / BATCH_SIZE;
    let n_test_batches  = test_all.batch_size  / BATCH_SIZE;
    let n_train = n_train_batches * BATCH_SIZE;
    let n_test  = n_test_batches  * BATCH_SIZE;
    println!("      using {} train / {} test samples ({} / {} batches)",
             n_train, n_test, n_train_batches, n_test_batches);

    // ── 3. Initialise CUDA ────────────────────────────────────────────────────
    println!("[2/8] initialising CUDA …");
    let cuda = Cuda::try_new()?;
    let device_infos = cuda.list_devices()?;
    let device = cuda.device(&device_infos[0].id)?;
    let capability = capability_from_device_info(&device.info)?;
    println!("      device: {} ({})", device.info.name, capability);

    // ── 4. Trace + compile the MLP ───────────────────────────────────────────
    println!("[3/8] tracing + compiling MLP graph …");
    let rustc_path = env::var("TEENY_RUSTC_PATH")
        .context("TEENY_RUSTC_PATH must point to the teenygrad rustc binary")?;
    let ptx_cache =
        env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());

    let (input, graph) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(1), Some(28), Some(28)]);
    let _output = Layer::call(&mnist_mlp::<f32>(), input);
    let graph = graph.borrow();
    println!("      graph: {} nodes", graph.nodes.len());

    let compiler = LlvmCompiler::new(rustc_path, ptx_cache)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let target = Target::new(capability);
    let lowering = TritonLowering::new();
    let cuda_model = graph_compiler.compile_model(&graph, &lowering, &target, false)?;
    println!("      compiled {} DAG nodes", cuda_model.dag.len());

    // ── 5. Load model + initialise weights ───────────────────────────────────
    println!("[4/8] loading model + uploading Kaiming-uniform weights …");
    let mut model = cuda_model.load(&device, BATCH_SIZE)?;
    let param_info: Vec<(usize, Vec<Vec<usize>>)> = model
        .param_info()
        .map(|(idx, shapes)| (idx, shapes.to_vec()))
        .collect();
    let mut n_params = 0usize;
    for (node_idx, shapes) in &param_info {
        for (param_idx, shape) in shapes.iter().enumerate() {
            let data = kaiming_uniform(shape);
            n_params += data.len();
            model.load_param_f32(*node_idx, param_idx, &data)?;
        }
    }
    println!("      {n_params} trainable parameters");

    // ── 6. Compile standalone CE loss + AdamW kernels ─────────────────────────
    println!("[5/8] compiling CE loss + AdamW kernels …");
    let ce_fwd_spec  = CrossEntropyLossForward::new(CE_BLOCK_SIZE);
    let ce_bwd_spec  = CrossEntropyLossBackward::new(CE_BLOCK_SIZE);
    let adamw_spec   = AdamwStep::new(ADAMW_BLOCK_SIZE);

    let ptx_ce_fwd  = std::fs::read(compile_kernel(&ce_fwd_spec,  &target, false)?)?;
    let ptx_ce_bwd  = std::fs::read(compile_kernel(&ce_bwd_spec,  &target, false)?)?;
    let ptx_adamw   = std::fs::read(compile_kernel(&adamw_spec,   &target, false)?)?;

    let ce_fwd_prog = CudaProgram::<ErasedKernel>::try_from_ptx(&ptx_ce_fwd, "entry_point")?;
    let ce_bwd_prog = CudaProgram::<ErasedKernel>::try_from_ptx(&ptx_ce_bwd, "entry_point")?;
    let adamw_kernel = AdamwKernel::from_ptx(&ptx_adamw, ADAMW_BLOCK_SIZE as u32)?;
    println!("      done.");

    // ── 7. Pre-allocate fixed device buffers ──────────────────────────────────
    // img_ptr is allocated fresh each batch (owned by the ActivationCache).
    let img_numel = BATCH_SIZE * 784; // 1 × 28 × 28
    let tgt_ptr      = mem::alloc(BATCH_SIZE * size_of::<i32>())?;
    let loss_ptr     = mem::alloc(BATCH_SIZE * size_of::<f32>())?;
    let dy_ptr       = mem::alloc(BATCH_SIZE * size_of::<f32>())?;
    let dx_logit_ptr = mem::alloc(BATCH_SIZE * N_CLASSES * size_of::<f32>())?;

    // Upstream CE gradient: 1/BATCH_SIZE for mean reduction (constant for all steps).
    let dy_host = vec![1.0_f32 / BATCH_SIZE as f32; BATCH_SIZE];
    unsafe { mem::copy_h_to_d(dy_ptr, dy_host.as_ptr(), BATCH_SIZE) }?;

    // CE loss launch config: one CTA per row (batch element).
    let ce_cfg = CudaLaunchConfig {
        grid:    [BATCH_SIZE as u32, 1, 1],
        block:   [CE_PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    };

    // ── 8. Training loop ──────────────────────────────────────────────────────
    println!("[6/8] training for {N_EPOCHS} epochs …");
    println!();

    for epoch in 0..N_EPOCHS {
        let mut epoch_loss = 0.0_f32;

        for batch_idx in 0..n_train_batches {
            let start = batch_idx * BATCH_SIZE;
            let img_slice  = &train_all.images[start * 784 .. (start + BATCH_SIZE) * 784];
            let lbl_slice  = &train_all.labels[start .. start + BATCH_SIZE];
            let labels_i32: Vec<i32> = lbl_slice.iter().map(|&l| l as i32).collect();

            // Allocate image buffer for this batch.
            // The ActivationCache takes ownership and frees it on drop.
            let img_ptr = mem::alloc(img_numel * size_of::<f32>())?;
            unsafe { mem::copy_h_to_d(img_ptr, img_slice.as_ptr(), img_numel) }?;
            unsafe { mem::copy_h_to_d(tgt_ptr, labels_i32.as_ptr(), BATCH_SIZE) }?;

            let img_ref = TensorRef::new(img_ptr, vec![BATCH_SIZE, 1, 28, 28]);
            model.zero_grad();

            // Forward pass (retain all activations for backward).
            let (logits, cache) =
                model.forward_train(&device, BATCH_SIZE, &[img_ref])?;

            // CE forward: compute per-sample loss for reporting.
            {
                let mut packer = CudaArgPacker::new();
                packer.visit_ptr(logits.ptr as *mut c_void);
                packer.visit_ptr(tgt_ptr   as *mut c_void);
                packer.visit_ptr(loss_ptr  as *mut c_void);
                packer.visit_i32(BATCH_SIZE as i32);
                packer.visit_i32(N_CLASSES  as i32);
                device.launch_with_packer(&ce_fwd_prog, &ce_cfg, &mut packer)?;
            }
            let mut loss_host = vec![0.0_f32; BATCH_SIZE];
            unsafe { mem::copy_d_to_h(loss_host.as_mut_ptr(), loss_ptr, BATCH_SIZE) }?;
            let batch_loss = loss_host.iter().sum::<f32>() / BATCH_SIZE as f32;
            epoch_loss += batch_loss;

            // CE backward: compute logit gradients (dy * (softmax(x) - one_hot(target))).
            // The result is written to dx_logit_ptr and used as model's grad_output.
            {
                let mut packer = CudaArgPacker::new();
                packer.visit_ptr(dy_ptr      as *mut c_void);
                packer.visit_ptr(logits.ptr  as *mut c_void);
                packer.visit_ptr(tgt_ptr     as *mut c_void);
                packer.visit_ptr(dx_logit_ptr as *mut c_void);
                packer.visit_i32(BATCH_SIZE as i32);
                packer.visit_i32(N_CLASSES  as i32);
                device.launch_with_packer(&ce_bwd_prog, &ce_cfg, &mut packer)?;
            }

            // Model backward: propagate gradients through the MLP layers.
            let grad_out = TensorRef::new(dx_logit_ptr, vec![BATCH_SIZE, N_CLASSES]);
            model.backward(&device, BATCH_SIZE, grad_out, &cache)?;

            // Drop the activation cache (frees all intermediate + input buffers).
            // After this point, logits.ptr is dangling — do not use logits.
            drop(cache);

            // AdamW parameter update.
            model.adamw_step(&device, &adamw_kernel, LR, BETA1, BETA2, EPS, WEIGHT_DECAY)?;

            if (batch_idx + 1) % 200 == 0 || batch_idx + 1 == n_train_batches {
                println!(
                    "  epoch {}/{N_EPOCHS}  step {:>4}/{n_train_batches}  loss={batch_loss:.4}",
                    epoch + 1, batch_idx + 1,
                );
            }
        }

        println!(
            "  ─── epoch {}/{N_EPOCHS} complete: avg_loss={:.4} ───",
            epoch + 1, epoch_loss / n_train_batches as f32,
        );
        println!();
    }

    // ── 9. Test evaluation ────────────────────────────────────────────────────
    println!("[7/8] evaluating on test set …");
    let test_img_ptr = mem::alloc(img_numel * size_of::<f32>())?;
    let mut total_correct = 0usize;

    for batch_idx in 0..n_test_batches {
        let start     = batch_idx * BATCH_SIZE;
        let img_slice = &test_all.images[start * 784 .. (start + BATCH_SIZE) * 784];
        let lbl_slice = &test_all.labels[start .. start + BATCH_SIZE];

        unsafe { mem::copy_h_to_d(test_img_ptr, img_slice.as_ptr(), img_numel) }?;
        let img_ref = TensorRef::new(test_img_ptr, vec![BATCH_SIZE, 1, 28, 28]);

        // Inference: intermediate activations are freed by forward(); caller owns result.ptr.
        let logits = model.forward(&device, BATCH_SIZE, &[img_ref])?;

        let mut logits_host = vec![0.0_f32; BATCH_SIZE * N_CLASSES];
        unsafe { mem::copy_d_to_h(logits_host.as_mut_ptr(), logits.ptr, BATCH_SIZE * N_CLASSES) }?;
        mem::free(logits.ptr)?;

        for i in 0..BATCH_SIZE {
            let row = &logits_host[i * N_CLASSES .. (i + 1) * N_CLASSES];
            if argmax(row) == lbl_slice[i] as usize {
                total_correct += 1;
            }
        }
    }

    mem::free(test_img_ptr)?;
    let total_test = n_test_batches * BATCH_SIZE;

    // ── 10. Results ───────────────────────────────────────────────────────────
    println!("[8/8] done.");
    println!();
    println!("  Test accuracy: {total_correct}/{total_test}  ({:.2}%)",
             100.0 * total_correct as f32 / total_test as f32);
    println!();

    // ── Cleanup ───────────────────────────────────────────────────────────────
    mem::free(tgt_ptr)?;
    mem::free(loss_ptr)?;
    mem::free(dy_ptr)?;
    mem::free(dx_logit_ptr)?;

    Ok(())
}

// ── Weight initialisation ─────────────────────────────────────────────────────

/// Kaiming-uniform initialisation: `bound = sqrt(6 / fan_in)`, uniform in `[-bound, bound]`.
fn kaiming_uniform(shape: &[usize]) -> Vec<f32> {
    let fan_in: usize = if shape.len() >= 2 { shape[1..].iter().product() } else { shape[0] };
    let bound = (6.0_f32 / fan_in as f32).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<f32>() * 2.0 * bound - bound).collect()
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
