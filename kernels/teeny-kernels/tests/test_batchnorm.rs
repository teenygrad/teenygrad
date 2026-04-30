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

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

#[cfg(all(feature = "cuda", feature = "training"))]
use {
    teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler,
    teeny_core::{
        graph::{DtypeRepr, SymTensor},
        model::LoweringMode,
        nn::{batchnorm::BatchNorm1d, Layer},
    },
    teeny_cuda::{compiler::graph::CudaGraphCompiler, device::mem},
    teeny_kernels::graph::TritonLowering,
};

const N: usize = 64;     // batch size
const C: usize = 32;     // channels
const EPS: f32 = 1e-5;
const MOMENTUM: f32 = 0.1;
const BLOCK_N: i32 = 128; // must be >= N; each CTA handles the full batch in one tile
const TOL: f32 = 1e-4;

// ─── Reference implementations ───────────────────────────────────────────────

fn ref_batch_norm_inference(
    x: &[f32], weight: &[f32], bias: &[f32],
    running_mean: &[f32], running_var: &[f32],
    n: usize, c: usize, eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n * c];
    for ch in 0..c {
        let rstd = 1.0 / (running_var[ch] + eps).sqrt();
        for batch in 0..n {
            let idx = batch * c + ch;
            y[idx] = weight[ch] * (x[idx] - running_mean[ch]) * rstd + bias[ch];
        }
    }
    y
}

#[cfg(feature = "training")]
fn ref_batch_norm_stats(x: &[f32], n: usize, c: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; c];
    let mut rstd = vec![0.0f32; c];
    for ch in 0..c {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for batch in 0..n {
            let v = x[batch * c + ch];
            sum += v;
            sum_sq += v * v;
        }
        let m = sum / n as f32;
        let var = sum_sq / n as f32 - m * m;
        mean[ch] = m;
        rstd[ch] = 1.0 / (var + EPS).sqrt();
    }
    (mean, rstd)
}

#[cfg(feature = "training")]
fn ref_batch_norm_normalize(
    x: &[f32], weight: &[f32], bias: &[f32],
    mean: &[f32], rstd: &[f32],
    n: usize, c: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n * c];
    for ch in 0..c {
        for batch in 0..n {
            let idx = batch * c + ch;
            y[idx] = weight[ch] * (x[idx] - mean[ch]) * rstd[ch] + bias[ch];
        }
    }
    y
}

#[cfg(feature = "training")]
fn ref_batch_norm_backward(
    dy: &[f32], x: &[f32], weight: &[f32],
    mean: &[f32], rstd: &[f32],
    n: usize, c: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; n * c];
    let mut dweight = vec![0.0f32; c];
    let mut dbias = vec![0.0f32; c];
    for ch in 0..c {
        let mut sum_dy = 0.0f32;
        let mut sum_dy_xhat = 0.0f32;
        for batch in 0..n {
            let idx = batch * c + ch;
            let xhat = (x[idx] - mean[ch]) * rstd[ch];
            sum_dy += dy[idx];
            sum_dy_xhat += dy[idx] * xhat;
        }
        dbias[ch] = sum_dy;
        dweight[ch] = sum_dy_xhat;
        for batch in 0..n {
            let idx = batch * c + ch;
            let xhat = (x[idx] - mean[ch]) * rstd[ch];
            dx[idx] = weight[ch] * rstd[ch]
                * (dy[idx] - sum_dy / n as f32 - xhat * sum_dy_xhat / n as f32);
        }
    }
    (dx, dweight, dbias)
}

// ─── Source snapshot tests (no CUDA required) ─────────────────────────────────

#[test]
fn test_batch_norm_inference_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability;
    let kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormForwardInference::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("batch_norm_inference_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_stats_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability;
    let kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormStatsForward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("batch_norm_stats_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_normalize_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability;
    let kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormNormalizeForward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("batch_norm_normalize_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_backward_source() -> anyhow::Result<()> {
    dotenv()?;
    use teeny_cuda::compiler::target::Capability;
    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormBackward::<f32>::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("batch_norm_backward_source", kernel.source());
    Ok(())
}

// ─── CUDA execution tests ─────────────────────────────────────────────────────

// Grid for all batchnorm kernels: one CTA per channel.
#[cfg(feature = "cuda")]
fn bn_cfg() -> CudaLaunchConfig {
    CudaLaunchConfig { grid: [C as u32, 1, 1], block: [1, 1, 1], cluster: [1, 1, 1] }
}

#[test]
#[cfg(feature = "cuda")]
fn test_batch_norm_inference_gpu() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x: Vec<f32> = (0..N * C).map(|i| (i as f32 * 0.1 - 3.2) % 5.0).collect();
    let weight: Vec<f32> = (0..C).map(|i| 0.5 + i as f32 * 0.03).collect();
    let bias: Vec<f32> = (0..C).map(|i| -0.1 + i as f32 * 0.02).collect();
    let running_mean: Vec<f32> = (0..C).map(|i| i as f32 * 0.01).collect();
    let running_var: Vec<f32> = (0..C).map(|i| 1.0 + i as f32 * 0.005).collect();

    let expected = ref_batch_norm_inference(&x, &weight, &bias, &running_mean, &running_var, N, C, EPS);

    let mut x_buf = device.buffer::<f32>(N * C)?;
    let mut w_buf = device.buffer::<f32>(C)?;
    let mut b_buf = device.buffer::<f32>(C)?;
    let mut rm_buf = device.buffer::<f32>(C)?;
    let mut rv_buf = device.buffer::<f32>(C)?;
    let y_buf = device.buffer::<f32>(N * C)?;
    let mut y_out = vec![0.0f32; N * C];

    x_buf.to_device(&x)?;
    w_buf.to_device(&weight)?;
    b_buf.to_device(&bias)?;
    rm_buf.to_device(&running_mean)?;
    rv_buf.to_device(&running_var)?;

    let kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormForwardInference::<f32>::new(BLOCK_N);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::norm::batchnorm::BatchNormForwardInference<f32>,
    >(&ptx)?;

    device.launch(&program, &bn_cfg(), (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        b_buf.as_device_ptr() as *mut f32,
        rm_buf.as_device_ptr() as *mut f32,
        rv_buf.as_device_ptr() as *mut f32,
        N as i32, C as i32, EPS,
    ))?;

    y_buf.to_host(&mut y_out)?;
    for i in 0..N * C {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "inference mismatch at i={i}: gpu={} ref={}",
            y_out[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "training"))]
fn test_batch_norm_forward_training_gpu() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x: Vec<f32> = (0..N * C).map(|i| (i as f32 * 0.07 - 2.0) % 4.0).collect();
    let weight: Vec<f32> = (0..C).map(|i| 1.0 + i as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..C).map(|i| i as f32 * 0.005).collect();
    let running_mean = vec![0.0f32; C];
    let running_var = vec![1.0f32; C];

    let (ref_mean, ref_rstd) = ref_batch_norm_stats(&x, N, C);
    let ref_y = ref_batch_norm_normalize(&x, &weight, &bias, &ref_mean, &ref_rstd, N, C);

    let mut x_buf = device.buffer::<f32>(N * C)?;
    let mut w_buf = device.buffer::<f32>(C)?;
    let mut b_buf = device.buffer::<f32>(C)?;
    let mut rm_buf = device.buffer::<f32>(C)?;
    let mut rv_buf = device.buffer::<f32>(C)?;
    let mean_buf = device.buffer::<f32>(C)?;
    let rstd_buf = device.buffer::<f32>(C)?;
    let y_buf = device.buffer::<f32>(N * C)?;
    let mut y_out = vec![0.0f32; N * C];

    x_buf.to_device(&x)?;
    w_buf.to_device(&weight)?;
    b_buf.to_device(&bias)?;
    rm_buf.to_device(&running_mean)?;
    rv_buf.to_device(&running_var)?;

    let target = Target::new(env.capability);

    // Launch 1: stats (computes mean, rstd, updates running stats).
    let stats_kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormStatsForward::<f32>::new(BLOCK_N);
    let stats_ptx = std::fs::read(compile_kernel(&stats_kernel, &target, true)?)?;
    let stats_prog = testing::load_program_from_ptx::<
        teeny_kernels::nn::norm::batchnorm::BatchNormStatsForward<f32>,
    >(&stats_ptx)?;
    device.launch(&stats_prog, &bn_cfg(), (
        x_buf.as_device_ptr() as *mut f32,
        mean_buf.as_device_ptr() as *mut f32,
        rstd_buf.as_device_ptr() as *mut f32,
        rm_buf.as_device_ptr() as *mut f32,
        rv_buf.as_device_ptr() as *mut f32,
        N as i32, C as i32, EPS, MOMENTUM,
    ))?;

    // Launch 2: normalise (reads mean/rstd written by launch 1).
    let norm_kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormNormalizeForward::<f32>::new(BLOCK_N);
    let norm_ptx = std::fs::read(compile_kernel(&norm_kernel, &target, true)?)?;
    let norm_prog = testing::load_program_from_ptx::<
        teeny_kernels::nn::norm::batchnorm::BatchNormNormalizeForward<f32>,
    >(&norm_ptx)?;
    device.launch(&norm_prog, &bn_cfg(), (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        b_buf.as_device_ptr() as *mut f32,
        mean_buf.as_device_ptr() as *mut f32,
        rstd_buf.as_device_ptr() as *mut f32,
        N as i32, C as i32,
    ))?;

    y_buf.to_host(&mut y_out)?;
    for i in 0..N * C {
        assert!(
            (y_out[i] - ref_y[i]).abs() < TOL,
            "training fwd mismatch at i={i}: gpu={} ref={}",
            y_out[i], ref_y[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "training"))]
fn test_batch_norm_backward_gpu() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x: Vec<f32> = (0..N * C).map(|i| (i as f32 * 0.07 - 2.0) % 4.0).collect();
    let dy: Vec<f32> = (0..N * C).map(|i| (i as f32 * 0.03 - 1.0) % 2.0).collect();
    let weight: Vec<f32> = (0..C).map(|i| 1.0 + i as f32 * 0.01).collect();

    let (mean, rstd) = ref_batch_norm_stats(&x, N, C);
    let (ref_dx, ref_dweight, ref_dbias) =
        ref_batch_norm_backward(&dy, &x, &weight, &mean, &rstd, N, C);

    let mut x_buf = device.buffer::<f32>(N * C)?;
    let mut dy_buf = device.buffer::<f32>(N * C)?;
    let mut w_buf = device.buffer::<f32>(C)?;
    let mut mean_buf = device.buffer::<f32>(C)?;
    let mut rstd_buf = device.buffer::<f32>(C)?;
    let dx_buf = device.buffer::<f32>(N * C)?;
    let dw_buf = device.buffer::<f32>(C)?;
    let db_buf = device.buffer::<f32>(C)?;
    let mut dx_out = vec![0.0f32; N * C];
    let mut dw_out = vec![0.0f32; C];
    let mut db_out = vec![0.0f32; C];

    x_buf.to_device(&x)?;
    dy_buf.to_device(&dy)?;
    w_buf.to_device(&weight)?;
    mean_buf.to_device(&mean)?;
    rstd_buf.to_device(&rstd)?;

    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormBackward::<f32>::new(BLOCK_N);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        teeny_kernels::nn::norm::batchnorm::BatchNormBackward<f32>,
    >(&ptx)?;

    device.launch(&program, &bn_cfg(), (
        dy_buf.as_device_ptr() as *mut f32,
        x_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        w_buf.as_device_ptr() as *mut f32,
        mean_buf.as_device_ptr() as *mut f32,
        rstd_buf.as_device_ptr() as *mut f32,
        dw_buf.as_device_ptr() as *mut f32,
        db_buf.as_device_ptr() as *mut f32,
        N as i32, C as i32,
    ))?;

    dx_buf.to_host(&mut dx_out)?;
    dw_buf.to_host(&mut dw_out)?;
    db_buf.to_host(&mut db_out)?;

    for i in 0..N * C {
        assert!(
            (dx_out[i] - ref_dx[i]).abs() < TOL,
            "dx mismatch at i={i}: gpu={} ref={}",
            dx_out[i], ref_dx[i]
        );
    }
    for ch in 0..C {
        assert!(
            (dw_out[ch] - ref_dweight[ch]).abs() < TOL,
            "dweight mismatch at ch={ch}: gpu={} ref={}",
            dw_out[ch], ref_dweight[ch]
        );
        assert!(
            (db_out[ch] - ref_dbias[ch]).abs() < TOL,
            "dbias mismatch at ch={ch}: gpu={} ref={}",
            db_out[ch], ref_dbias[ch]
        );
    }
    Ok(())
}

// ─── Graph-compiler training test ─────────────────────────────────────────────

/// Verifies that the graph compiler emits two sequential DAG nodes for
/// BatchNorm1d under `LoweringMode::Training` (stats + normalize), and that
/// running both kernels through `LoadedModel::forward` produces the correct
/// normalized output.
#[test]
#[cfg(all(feature = "cuda", feature = "training"))]
fn test_batch_norm_training_graph() -> anyhow::Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    // Trace graph: Input([None, C]) → BatchNorm1d.
    let (input, graph) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(C)]);
    let _output = Layer::call(
        &BatchNorm1d::<f32, SymTensor, SymTensor, 2>::new(C)
            .with_eps(EPS as f64)
            .with_momentum(MOMENTUM as f64),
        input,
    );
    let graph = graph.borrow();

    // Compile with LoweringMode::Training — should produce 3 DAG nodes:
    // Input, BatchNormStats, BatchNormNormalize.
    let rustc_path = std::env::var("TEENY_RUSTC_PATH")
        .expect("TEENY_RUSTC_PATH must be set");
    let cache_dir =
        std::env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());
    let compiler = LlvmCompiler::new(rustc_path, cache_dir)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let lowering = TritonLowering::new();
    let model = graph_compiler.compile_model(
        &graph, &lowering, &target, LoweringMode::Training, false,
    )?;

    // Graph has 2 nodes (Input + BatchNorm1d); training lowering expands to 3.
    assert_eq!(model.dag.len(), 3, "expected Input + Stats + Normalize nodes");

    // Load the model.
    let mut loaded = model.load(&env.device, N)?;

    // Upload parameters.
    // Node 0 = Input (no params), Node 1 = Stats (running_mean, running_var),
    // Node 2 = Normalize (weight, bias).
    let running_mean = vec![0.0f32; C];
    let running_var  = vec![1.0f32; C];
    let weight: Vec<f32> = (0..C).map(|i| 1.0 + i as f32 * 0.01).collect();
    let bias:   Vec<f32> = (0..C).map(|i| i as f32 * 0.005).collect();

    loaded.load_param_f32(1, 0, &running_mean)?;  // stats: running_mean
    loaded.load_param_f32(1, 1, &running_var)?;   // stats: running_var
    loaded.load_param_f32(2, 0, &weight)?;         // normalize: weight
    loaded.load_param_f32(2, 1, &bias)?;           // normalize: bias

    // Input data.
    let x: Vec<f32> = (0..N * C).map(|i| (i as f32 * 0.07 - 2.0) % 4.0).collect();

    // Reference: compute stats then normalize.
    let (ref_mean, ref_rstd) = ref_batch_norm_stats(&x, N, C);
    let ref_y = ref_batch_norm_normalize(&x, &weight, &bias, &ref_mean, &ref_rstd, N, C);

    // Allocate input on device.
    let x_ptr = mem::alloc(N * C * std::mem::size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N * C) }?;
    let x_tensor = teeny_cuda::model::TensorRef::new(x_ptr, vec![N, C]);

    // Run forward.
    let output = loaded.forward(&env.device, N, &[x_tensor])?;

    // Copy result back to host.
    let mut y_out = vec![0.0f32; N * C];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N * C) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    for i in 0..N * C {
        assert!(
            (y_out[i] - ref_y[i]).abs() < TOL,
            "graph training mismatch at i={i}: gpu={} ref={}",
            y_out[i], ref_y[i]
        );
    }

    Ok(())
}
