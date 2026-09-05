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

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::device::program::Kernel;

#[cfg(all(feature = "cuda", feature = "hardware", feature = "training"))]
use teeny_cuda::compiler::target::Target;

#[cfg(feature = "hardware")]
use teeny_core::device::{Device, buffer::Buffer};
#[cfg(all(feature = "cuda", feature = "hardware", feature = "training"))]
use teeny_test::cuda as testing;

#[cfg(feature = "hardware")]
use teeny_test::load_fixture;
#[cfg(all(feature = "cuda", feature = "hardware", feature = "training"))]
use {
    teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler,
    teeny_core::{
        graph::{DtypeRepr, SymTensor},
        model::LoweringMode,
        nn::{Layer, batchnorm::BatchNorm1d},
    },
    teeny_cuda::{compiler::graph::CudaGraphCompiler, device::mem},
    teeny_kernels::graph::TritonLowering,
};

#[cfg(feature = "hardware")]
const N: usize = 64;
#[cfg(feature = "hardware")]
const C: usize = 32;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-5;
#[cfg(all(feature = "hardware", feature = "training"))]
const MOMENTUM: f32 = 0.1;
const BLOCK_N: i32 = 128;
#[cfg(feature = "hardware")]
const TOL: f32 = 1e-4;

// ─── Source snapshot tests (no CUDA required) ─────────────────────────────────

#[test]
fn test_batch_norm_inference_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormForwardInference::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "batch_norm_inference_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_stats_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormStatsForward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!("batch_norm_stats_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_normalize_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormNormalizeForward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "batch_norm_normalize_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormBackward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!("batch_norm_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    Ok(())
}

// ─── CUDA execution tests ─────────────────────────────────────────────────────

#[cfg(feature = "hardware")]
fn bn_cfg() -> teeny_runtime::LaunchConfig {
    teeny_runtime::launch_config_custom([C as u32, 1, 1], [1, 1, 1], [1, 1, 1])
}

#[test]
#[cfg(feature = "hardware")]
fn test_batch_norm_inference_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/x.bin");
    let weight = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/weight.bin");
    let bias = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/bias.bin");
    let running_mean = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/running_mean.bin");
    let running_var = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/running_var.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "batchnorm/expected_forward_inference.bin",
    );

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

    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNormForwardInference::<f32>::new(BLOCK_N);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::batchnorm::BatchNormForwardInference<f32>,
    >(&ptx_path)?;

    device.launch(
        &program,
        &bn_cfg(),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            b_buf.as_device_ptr(),
            rm_buf.as_device_ptr(),
            rv_buf.as_device_ptr(),
            N as i32,
            C as i32,
            EPS,
        ),
    )?;

    y_buf.to_host(&mut y_out)?;
    for i in 0..N * C {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "inference mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "hardware", feature = "training"))]
fn test_batch_norm_forward_training_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/x.bin");
    let weight = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/weight.bin");
    let bias = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/bias.bin");
    let running_mean = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/running_mean.bin");
    let running_var = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/running_var.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "batchnorm/expected_forward_training.bin",
    );

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

    let target = teeny_runtime::default_target(&device)?;

    let stats_kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormStatsForward::<f32>::new(BLOCK_N);
    let stats_ptx_path = teeny_runtime::compile_kernel(&stats_kernel, &target, true, false)?;
    let stats_prog = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::batchnorm::BatchNormStatsForward<f32>,
    >(&stats_ptx_path)?;
    device.launch(
        &stats_prog,
        &bn_cfg(),
        (
            x_buf.as_device_ptr(),
            mean_buf.as_device_ptr(),
            rstd_buf.as_device_ptr(),
            rm_buf.as_device_ptr(),
            rv_buf.as_device_ptr(),
            N as i32,
            C as i32,
            EPS,
            MOMENTUM,
        ),
    )?;

    let norm_kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNormNormalizeForward::<f32>::new(BLOCK_N);
    let norm_ptx_path = teeny_runtime::compile_kernel(&norm_kernel, &target, true, false)?;
    let norm_prog = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::batchnorm::BatchNormNormalizeForward<f32>,
    >(&norm_ptx_path)?;
    device.launch(
        &norm_prog,
        &bn_cfg(),
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            b_buf.as_device_ptr(),
            mean_buf.as_device_ptr(),
            rstd_buf.as_device_ptr(),
            N as i32,
            C as i32,
        ),
    )?;

    y_buf.to_host(&mut y_out)?;
    for i in 0..N * C {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "training fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "hardware", feature = "training"))]
fn test_batch_norm_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/x.bin");
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/dy.bin");
    let weight = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/weight.bin");
    let mean = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/expected_mean.bin");
    let rstd = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/expected_rstd.bin");
    let expected_dx = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/expected_dx.bin");
    let expected_dweight =
        load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/expected_dweight.bin");
    let expected_dbias = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/expected_dbias.bin");

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
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::batchnorm::BatchNormBackward<f32>,
    >(&ptx_path)?;

    device.launch(
        &program,
        &bn_cfg(),
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            mean_buf.as_device_ptr(),
            rstd_buf.as_device_ptr(),
            dw_buf.as_device_ptr(),
            db_buf.as_device_ptr(),
            N as i32,
            C as i32,
        ),
    )?;

    dx_buf.to_host(&mut dx_out)?;
    dw_buf.to_host(&mut dw_out)?;
    db_buf.to_host(&mut db_out)?;

    for i in 0..N * C {
        assert!(
            (dx_out[i] - expected_dx[i]).abs() < TOL,
            "dx mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected_dx[i]
        );
    }
    for ch in 0..C {
        assert!(
            (dw_out[ch] - expected_dweight[ch]).abs() < TOL,
            "dweight mismatch at ch={ch}: gpu={} expected={}",
            dw_out[ch],
            expected_dweight[ch]
        );
        assert!(
            (db_out[ch] - expected_dbias[ch]).abs() < TOL,
            "dbias mismatch at ch={ch}: gpu={} expected={}",
            db_out[ch],
            expected_dbias[ch]
        );
    }
    Ok(())
}

// ─── NCHW backward kernel tests ──────────────────────────────────────────────

#[cfg(feature = "training")]
#[test]
fn test_batch_norm_2d_nchw_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::norm::batchnorm::BatchNorm2dNchwBackward::<f32>::new(BLOCK_N);
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    assert_debug_snapshot!(
        format!(
            "batch_norm_2d_nchw_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    Ok(())
}

// Analytically verified test: mean=2, var=0, eps=1 → rstd=1.
//   x = 5.0 everywhere, weight = 3.0 everywhere, dy = 1.0 everywhere.
//   xhat = (5 - 2) * 1 = 3.
//   dx   = 3 * 1 * 1 = 3.0 per element.
//   dweight[c] = B*H*W * (1 * 3) = 4 * 3 = 12.0.
//   dbias[c]   = B*H*W * 1       = 4.0.
#[test]
#[cfg(all(feature = "hardware", feature = "training"))]
fn test_batch_norm_2d_nchw_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    const BN_B: usize = 1;
    const BN_C: usize = 4;
    const BN_H: usize = 2;
    const BN_W: usize = 2;
    const BN_HW: usize = BN_H * BN_W;
    const BN_ELEM: usize = BN_B * BN_C * BN_HW;
    const BN_BLOCK_HW: i32 = 128;
    const BN_EPS: f32 = 1.0;

    let x_host = vec![5.0_f32; BN_ELEM];
    let dy_host = vec![1.0_f32; BN_ELEM];
    let weight_host = vec![3.0_f32; BN_C];
    let running_mean_host = vec![2.0_f32; BN_C];
    let running_var_host = vec![0.0_f32; BN_C];

    let mut x_buf = device.buffer::<f32>(BN_ELEM)?;
    let mut dy_buf = device.buffer::<f32>(BN_ELEM)?;
    let mut w_buf = device.buffer::<f32>(BN_C)?;
    let mut rm_buf = device.buffer::<f32>(BN_C)?;
    let mut rv_buf = device.buffer::<f32>(BN_C)?;
    let dx_buf = device.buffer::<f32>(BN_ELEM)?;
    let dw_buf = device.buffer::<f32>(BN_C)?;
    let db_buf = device.buffer::<f32>(BN_C)?;

    x_buf.to_device(&x_host)?;
    dy_buf.to_device(&dy_host)?;
    w_buf.to_device(&weight_host)?;
    rm_buf.to_device(&running_mean_host)?;
    rv_buf.to_device(&running_var_host)?;

    let kernel =
        teeny_kernels::nn::norm::batchnorm::BatchNorm2dNchwBackward::<f32>::new(BN_BLOCK_HW);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::norm::batchnorm::BatchNorm2dNchwBackward<f32>,
    >(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [BN_C as u32, 1, 1],
        [BN_BLOCK_HW as u32, 1, 1],
        [1, 1, 1],
    );
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            rm_buf.as_device_ptr(),
            rv_buf.as_device_ptr(),
            dw_buf.as_device_ptr(),
            db_buf.as_device_ptr(),
            BN_B as i32,
            BN_C as i32,
            BN_HW as i32,
            BN_EPS,
        ),
    )?;

    let mut dx_out = vec![0.0_f32; BN_ELEM];
    let mut dw_out = vec![0.0_f32; BN_C];
    let mut db_out = vec![0.0_f32; BN_C];
    dx_buf.to_host(&mut dx_out)?;
    dw_buf.to_host(&mut dw_out)?;
    db_buf.to_host(&mut db_out)?;

    // rstd = 1/sqrt(0 + 1.0) = 1.0; dx = weight * rstd * dy = 3 * 1 * 1 = 3.0
    for (i, &v) in dx_out.iter().enumerate() {
        assert!(
            (v - 3.0).abs() < 1e-5,
            "nchw_backward: dx[{i}] = {v}, expected 3.0",
        );
    }
    // dweight[c] = B*H*W * (dy * xhat) = 4 * (1 * 3) = 12.0
    for (c, &v) in dw_out.iter().enumerate() {
        assert!(
            (v - 12.0).abs() < 1e-5,
            "nchw_backward: dweight[{c}] = {v}, expected 12.0",
        );
    }
    // dbias[c] = B*H*W * dy = 4 * 1 = 4.0
    for (c, &v) in db_out.iter().enumerate() {
        assert!(
            (v - 4.0).abs() < 1e-5,
            "nchw_backward: dbias[{c}] = {v}, expected 4.0",
        );
    }
    Ok(())
}

// ─── Graph-compiler training test ─────────────────────────────────────────────

#[test]
#[cfg(all(feature = "cuda", feature = "hardware", feature = "training"))]
fn test_batch_norm_training_graph() -> anyhow::Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(C)]);
    let _output = Layer::call(
        &BatchNorm1d::<f32, SymTensor, SymTensor, 2>::new(C)
            .with_eps(EPS as f64)
            .with_momentum(MOMENTUM as f64),
        input,
    );
    let graph = graph.borrow();

    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let lowering = TritonLowering::new();
    let model =
        graph_compiler.compile_model(&graph, &lowering, &target, LoweringMode::Training, false)?;

    assert_eq!(
        model.dag.len(),
        3,
        "expected Input + Stats + Normalize nodes"
    );

    let mut loaded = model.load(&env.device, N)?;

    let running_mean = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/running_mean.bin");
    let running_var = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/running_var.bin");
    let weight = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/weight.bin");
    let bias = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/bias.bin");
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "batchnorm/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "batchnorm/expected_forward_training.bin",
    );

    loaded.load_param_f32(1, 0, &running_mean)?;
    loaded.load_param_f32(1, 1, &running_var)?;
    loaded.load_param_f32(2, 0, &weight)?;
    loaded.load_param_f32(2, 1, &bias)?;

    let x_ptr = mem::alloc(N * C * std::mem::size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N * C) }?;
    let x_tensor = teeny_cuda::model::TensorRef::new(x_ptr, vec![N, C]);

    let output = loaded.forward(&env.device, N, &[x_tensor])?;

    let mut y_out = vec![0.0f32; N * C];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N * C) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    for i in 0..N * C {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "graph training mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }

    Ok(())
}
