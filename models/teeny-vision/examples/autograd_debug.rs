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

//! Minimal autograd correctness check.
//!
//! Network: Linear(K→N, no bias), K=4, N=2, batch B=2.
//!
//! With known x and a hand-specified grad_output (upstream gradient of the
//! loss w.r.t. the layer output), the expected weight gradient is:
//!
//!   dW = grad_output^T @ x
//!
//! We run:
//!   1. forward_train  → logits (shape [B, N])
//!   2. inject a known grad_output directly (no loss kernel)
//!   3. backward       → accumulates dW into grad_param_bufs
//!   4. read dW back   → compare with expected
//!
//! Then repeat for a 2-layer network: Linear(4→4) → ReLU → Linear(4→2).
//! This verifies that gradients chain correctly through ReLU.

use std::env;

use anyhow::{Context, Result};
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::{
    device::context::Context as DeviceContext,
    graph::{DtypeRepr, SymTensor},
    model::LoweringMode,
    nn::{
        Layer,
        activation::relu::Relu,
        linear::Linear,
    },
    sequential,
};
use teeny_cuda::{
    compiler::{graph::CudaGraphCompiler, target::capability_from_device_info},
    device::{context::Cuda, mem},
    model::TensorRef,
};
use teeny_kernels::graph::TritonLowering;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Matrix multiply A (M×K) @ B (K×N) → C (M×N), row-major, host-side.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
    c
}

/// Transpose a (rows×cols) matrix stored row-major.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

// ── network factories ─────────────────────────────────────────────────────────

fn net_linear_only<D: teeny_core::dtype::Float>() -> impl Fn(SymTensor) -> SymTensor {
    sequential![Linear::<D, _, _, 2>::new(4, 2, false)]
}

fn net_linear_relu_linear<D: teeny_core::dtype::Float>() -> impl Fn(SymTensor) -> SymTensor {
    sequential![
        Linear::<D, _, _, 2>::new(4, 4, false),
        Relu::<D, _, 2>::new(),
        Linear::<D, _, _, 2>::new(4, 2, false)
    ]
}

// ── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let rustc_path = env::var("TEENY_RUSTC_PATH")
        .context("TEENY_RUSTC_PATH must be set")?;
    let ptx_cache =
        env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());

    let cuda = Cuda::try_new()?;
    let device_infos = cuda.list_devices()?;
    let device = cuda.device(&device_infos[0].id)?;
    let capability = capability_from_device_info(&device.info)?;
    println!("device: {} ({})", device.info.name, capability);

    let compiler = LlvmCompiler::new(rustc_path, ptx_cache)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let target = Target::new(capability);
    let lowering = TritonLowering::new();

    // ─────────────────────────────────────────────────────────────────────────
    // TEST 1: Single Linear(4→2), no bias
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════");
    println!("Test 1: single Linear(4→2), no bias");
    println!("═══════════════════════════════════════");

    const B: usize = 2;  // batch size
    const K: usize = 4;  // in_features
    const N: usize = 2;  // out_features

    // x[B, K] — input activation
    let x: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,   // row 0
        5.0, 6.0, 7.0, 8.0,   // row 1
    ];
    // W[N, K] — weight matrix
    let w: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4,   // row 0
        0.5, 0.6, 0.7, 0.8,   // row 1
    ];
    // grad_output[B, N] — upstream gradient (injected directly, no loss kernel)
    let grad_out_host: Vec<f32> = vec![
        1.0, 0.0,   // row 0
        0.0, 1.0,   // row 1
    ];

    // Expected dW = grad_output^T @ x
    // grad_output^T is [N, B], x is [B, K] → dW is [N, K]
    let grad_out_t = transpose(&grad_out_host, B, N);
    let expected_dw = matmul(&grad_out_t, &x, N, B, K);
    println!("expected dW = {:?}", expected_dw);

    let (input_sym, graph) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(K)]);
    let _output = Layer::call(&net_linear_only::<f32>(), input_sym);
    let graph = graph.borrow();

    let cuda_model = graph_compiler.compile_model(&graph, &lowering, &target, LoweringMode::Inference, false)?;
    let mut model = cuda_model.load(&device, B)?;

    let param_info: Vec<(usize, Vec<Vec<usize>>)> = model
        .param_info()
        .map(|(idx, shapes)| (idx, shapes.to_vec()))
        .collect();
    assert_eq!(param_info.len(), 1, "expected exactly one node with params");
    let (weight_node, _) = &param_info[0];
    let weight_node = *weight_node;
    model.load_param_f32(weight_node, 0, &w)?;

    // Allocate x on device (owned by ActivationCache)
    let x_ptr = mem::alloc(B * K * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), B * K) }?;
    let x_ref = TensorRef::new(x_ptr, vec![B, K]);

    model.zero_grad();
    let (_logits, cache) = model.forward_train(&device, B, &[x_ref])?;

    // Allocate and fill grad_output device buffer
    let grad_out_ptr = mem::alloc(B * N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(grad_out_ptr, grad_out_host.as_ptr(), B * N) }?;
    let grad_out_ref = TensorRef::new(grad_out_ptr, vec![B, N]);

    model.backward(&device, B, grad_out_ref, &cache)?;
    drop(cache);

    let actual_dw = model.read_param_grad_f32(weight_node, 0)?;

    mem::free(grad_out_ptr)?;

    println!("actual  dW = {:?}", actual_dw);
    let err1 = max_abs_diff(&actual_dw, &expected_dw);
    println!("max |actual - expected| = {err1:.2e}");
    if err1 < 1e-4 {
        println!("✓  Test 1 PASSED");
    } else {
        println!("✗  Test 1 FAILED");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TEST 2: Linear(4→4) → ReLU → Linear(4→2), gradient chain
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════");
    println!("Test 2: Linear(4→4) → ReLU → Linear(4→2)");
    println!("═══════════════════════════════════════");

    const K2: usize = 4;
    const H: usize = 4;   // hidden
    const N2: usize = 2;
    const B2: usize = 2;

    // Known weights
    // W1[H, K2]
    let w1: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    // W2[N2, H]
    let w2: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ];

    // x[B2, K2]
    let x2: Vec<f32> = vec![
        1.0,  2.0,  3.0,  4.0,
        5.0, -6.0,  7.0, -8.0,   // some negatives to exercise ReLU masking
    ];

    // Compute forward pass analytically:
    // h_pre = x2 @ W1^T  (W1 is identity, so h_pre = x2)
    // h = ReLU(h_pre)     → zero out negatives
    // y = h @ W2^T        (W2 selects first 2 features from h)

    let h_pre = matmul(&x2, &transpose(&w1, H, K2), B2, K2, H);  // [B2, H]
    let h: Vec<f32> = h_pre.iter().map(|&v| v.max(0.0)).collect();
    let y = matmul(&h, &transpose(&w2, N2, H), B2, H, N2);        // [B2, N2]
    println!("h_pre = {h_pre:?}");
    println!("h     = {h:?}");
    println!("y     = {y:?}");

    // grad_output[B2, N2] = all-ones
    let grad_out2_host = vec![1.0_f32; B2 * N2];

    // Expected dW2 = grad_output^T @ h  (shape [N2, H])
    let grad_out2_t = transpose(&grad_out2_host, B2, N2);
    let expected_dw2 = matmul(&grad_out2_t, &h, N2, B2, H);
    println!("expected dW2 = {expected_dw2:?}");

    // dh = grad_output @ W2 (shape [B2, H])
    let dh = matmul(&grad_out2_host, &w2, B2, N2, H);
    // dh_pre = dh * (h_pre > 0)  (ReLU backward)
    let dh_pre: Vec<f32> = dh.iter().zip(h_pre.iter()).map(|(&g, &a)| if a > 0.0 { g } else { 0.0 }).collect();
    // Expected dW1 = dh_pre^T @ x2  (shape [H, K2])
    let dh_pre_t = transpose(&dh_pre, B2, H);
    let expected_dw1 = matmul(&dh_pre_t, &x2, H, B2, K2);
    println!("expected dW1 = {expected_dw1:?}");

    let (input_sym2, graph2) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(K2)]);
    let _output2 = Layer::call(&net_linear_relu_linear::<f32>(), input_sym2);
    let graph2 = graph2.borrow();

    let cuda_model2 = graph_compiler.compile_model(&graph2, &lowering, &target, LoweringMode::Inference, false)?;
    let mut model2 = cuda_model2.load(&device, B2)?;

    // Identify which node is W1 (smaller index) and W2 (larger index).
    let param_nodes: Vec<(usize, Vec<Vec<usize>>)> = model2
        .param_info()
        .map(|(idx, shapes)| (idx, shapes.to_vec()))
        .collect();
    assert_eq!(param_nodes.len(), 2, "expected 2 weight nodes");
    // Nodes are in topological order; first node = first linear layer.
    let (w1_node, _) = &param_nodes[0];
    let (w2_node, _) = &param_nodes[1];
    let (w1_node, w2_node) = (*w1_node, *w2_node);
    model2.load_param_f32(w1_node, 0, &w1)?;
    model2.load_param_f32(w2_node, 0, &w2)?;

    let x2_ptr = mem::alloc(B2 * K2 * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x2_ptr, x2.as_ptr(), B2 * K2) }?;
    let x2_ref = TensorRef::new(x2_ptr, vec![B2, K2]);

    model2.zero_grad();
    let (_logits2, cache2) = model2.forward_train(&device, B2, &[x2_ref])?;

    let grad_out2_ptr = mem::alloc(B2 * N2 * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(grad_out2_ptr, grad_out2_host.as_ptr(), B2 * N2) }?;
    let grad_out2_ref = TensorRef::new(grad_out2_ptr, vec![B2, N2]);

    model2.backward(&device, B2, grad_out2_ref, &cache2)?;
    drop(cache2);

    let actual_dw1 = model2.read_param_grad_f32(w1_node, 0)?;
    let actual_dw2 = model2.read_param_grad_f32(w2_node, 0)?;

    mem::free(grad_out2_ptr)?;

    println!("actual  dW2 = {actual_dw2:?}");
    let err2a = max_abs_diff(&actual_dw2, &expected_dw2);
    println!("max |dW2 actual - expected| = {err2a:.2e}");

    println!("actual  dW1 = {actual_dw1:?}");
    let err2b = max_abs_diff(&actual_dw1, &expected_dw1);
    println!("max |dW1 actual - expected| = {err2b:.2e}");

    if err2a < 1e-4 && err2b < 1e-4 {
        println!("✓  Test 2 PASSED");
    } else {
        println!("✗  Test 2 FAILED");
    }

    Ok(())
}
