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

const N: usize = 64;
const C: usize = 32;
const EPS: f32 = 1e-5;
const MOMENTUM: f32 = 0.1;
const BLOCK_N: i32 = 128;
const TOL: f32 = 1e-4;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
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

    let x = load_fixture("batchnorm/x.bin");
    let weight = load_fixture("batchnorm/weight.bin");
    let bias = load_fixture("batchnorm/bias.bin");
    let running_mean = load_fixture("batchnorm/running_mean.bin");
    let running_var = load_fixture("batchnorm/running_var.bin");
    let expected = load_fixture("batchnorm/expected_forward_inference.bin");

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
            "inference mismatch at i={i}: gpu={} expected={}",
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

    let x = load_fixture("batchnorm/x.bin");
    let weight = load_fixture("batchnorm/weight.bin");
    let bias = load_fixture("batchnorm/bias.bin");
    let running_mean = load_fixture("batchnorm/running_mean.bin");
    let running_var = load_fixture("batchnorm/running_var.bin");
    let expected = load_fixture("batchnorm/expected_forward_training.bin");

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
            (y_out[i] - expected[i]).abs() < TOL,
            "training fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i], expected[i]
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

    let x = load_fixture("batchnorm/x.bin");
    let dy = load_fixture("batchnorm/dy.bin");
    let weight = load_fixture("batchnorm/weight.bin");
    let mean = load_fixture("batchnorm/expected_mean.bin");
    let rstd = load_fixture("batchnorm/expected_rstd.bin");
    let expected_dx = load_fixture("batchnorm/expected_dx.bin");
    let expected_dweight = load_fixture("batchnorm/expected_dweight.bin");
    let expected_dbias = load_fixture("batchnorm/expected_dbias.bin");

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
            (dx_out[i] - expected_dx[i]).abs() < TOL,
            "dx mismatch at i={i}: gpu={} expected={}",
            dx_out[i], expected_dx[i]
        );
    }
    for ch in 0..C {
        assert!(
            (dw_out[ch] - expected_dweight[ch]).abs() < TOL,
            "dweight mismatch at ch={ch}: gpu={} expected={}",
            dw_out[ch], expected_dweight[ch]
        );
        assert!(
            (db_out[ch] - expected_dbias[ch]).abs() < TOL,
            "dbias mismatch at ch={ch}: gpu={} expected={}",
            db_out[ch], expected_dbias[ch]
        );
    }
    Ok(())
}

// ─── Graph-compiler training test ─────────────────────────────────────────────

#[test]
#[cfg(all(feature = "cuda", feature = "training"))]
fn test_batch_norm_training_graph() -> anyhow::Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let (input, graph) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(C)]);
    let _output = Layer::call(
        &BatchNorm1d::<f32, SymTensor, SymTensor, 2>::new(C)
            .with_eps(EPS as f64)
            .with_momentum(MOMENTUM as f64),
        input,
    );
    let graph = graph.borrow();

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

    assert_eq!(model.dag.len(), 3, "expected Input + Stats + Normalize nodes");

    let mut loaded = model.load(&env.device, N)?;

    let running_mean = load_fixture("batchnorm/running_mean.bin");
    let running_var = load_fixture("batchnorm/running_var.bin");
    let weight = load_fixture("batchnorm/weight.bin");
    let bias = load_fixture("batchnorm/bias.bin");
    let x = load_fixture("batchnorm/x.bin");
    let expected = load_fixture("batchnorm/expected_forward_training.bin");

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
            y_out[i], expected[i]
        );
    }

    Ok(())
}
