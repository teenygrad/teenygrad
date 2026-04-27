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
use std::path::PathBuf;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

const N_ROWS: usize = 64;
const N_COLS: usize = 16;
const BLOCK_SIZE_CE: i32 = 16; // next_power_of_two(N_COLS)
const N_MLSM: usize = 1024;
const BLOCK_SIZE_MLSM: i32 = 1024;
const PTX_THREADS: u32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_fixture_i32(rel: &str) -> Vec<i32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn row_launch_cfg() -> CudaLaunchConfig {
    CudaLaunchConfig {
        grid: [N_ROWS as u32, 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    }
}

fn mlsm_launch_cfg() -> CudaLaunchConfig {
    CudaLaunchConfig {
        grid: [(N_MLSM as u32).div_ceil(BLOCK_SIZE_MLSM as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    }
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_nll_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::nll::NllLossForward::new();
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("nll_loss_forward_source", kernel.source());
    assert_debug_snapshot!("nll_loss_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_cross_entropy_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::nll::CrossEntropyLossForward::new(BLOCK_SIZE_CE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("cross_entropy_loss_forward_source", kernel.source());
    assert_debug_snapshot!("cross_entropy_loss_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_multilabel_soft_margin_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::nn::loss::nll::MultilabelSoftMarginLossForward::new(BLOCK_SIZE_MLSM);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("multilabel_soft_margin_loss_forward_source", kernel.source());
    assert_debug_snapshot!("multilabel_soft_margin_loss_forward_mlir", mlir.trim());
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_nll_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let lp_host  = load_fixture("loss_nll/nll_log_probs.bin");
    let tgt_host = load_fixture_i32("loss_nll/nll_targets.bin");
    let expected = load_fixture("loss_nll/nll_expected_forward.bin");
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut lp_buf  = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut tgt_buf = device.buffer::<i32>(N_ROWS)?;
    let out_buf     = device.buffer::<f32>(N_ROWS)?;
    lp_buf.to_device(&lp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::nll::NllLossForward::new();
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::nll::NllLossForward>(&ptx)?;

    let args = (lp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut i32,
                out_buf.as_device_ptr() as *mut f32, N_ROWS as i32, N_COLS as i32);
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "nll_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_nll_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_nll/nll_dy.bin");
    let tgt_host = load_fixture_i32("loss_nll/nll_targets.bin");
    let expected = load_fixture("loss_nll/nll_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N_ROWS * N_COLS];

    let mut dy_buf  = device.buffer::<f32>(N_ROWS)?;
    let mut tgt_buf = device.buffer::<i32>(N_ROWS)?;
    let mut dx_buf  = device.buffer::<f32>(N_ROWS * N_COLS)?;
    dy_buf.to_device(&dy_host)?;
    tgt_buf.to_device(&tgt_host)?;
    dx_buf.to_device(&vec![0.0f32; N_ROWS * N_COLS])?; // kernel only writes target column

    let kernel = teeny_kernels::nn::loss::nll::NllLossBackward::new();
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::nll::NllLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut i32,
                dx_buf.as_device_ptr() as *mut f32, N_ROWS as i32, N_COLS as i32);
    device.launch(&program, &row_launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(N_ROWS * N_COLS) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "nll_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cross_entropy_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_nll/ce_input.bin");
    let tgt_host = load_fixture_i32("loss_nll/ce_targets.bin");
    let expected = load_fixture("loss_nll/ce_expected_forward.bin");
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut inp_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut tgt_buf = device.buffer::<i32>(N_ROWS)?;
    let out_buf     = device.buffer::<f32>(N_ROWS)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::nll::CrossEntropyLossForward::new(BLOCK_SIZE_CE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::nll::CrossEntropyLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut i32,
                out_buf.as_device_ptr() as *mut f32, N_ROWS as i32, N_COLS as i32);
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "cross_entropy_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cross_entropy_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_nll/ce_dy.bin");
    let inp_host = load_fixture("loss_nll/ce_input.bin");
    let tgt_host = load_fixture_i32("loss_nll/ce_targets.bin");
    let expected = load_fixture("loss_nll/ce_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N_ROWS * N_COLS];

    let mut dy_buf  = device.buffer::<f32>(N_ROWS)?;
    let mut inp_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut tgt_buf = device.buffer::<i32>(N_ROWS)?;
    let dx_buf      = device.buffer::<f32>(N_ROWS * N_COLS)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::nll::CrossEntropyLossBackward::new(BLOCK_SIZE_CE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::nll::CrossEntropyLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut i32, dx_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32, N_COLS as i32);
    device.launch(&program, &row_launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(N_ROWS * N_COLS) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "cross_entropy_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_multilabel_soft_margin_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_nll/mlsm_input.bin");
    let tgt_host = load_fixture("loss_nll/mlsm_target.bin");
    let expected = load_fixture("loss_nll/mlsm_expected_forward.bin");
    let mut out_host = vec![0.0f32; N_MLSM];

    let mut inp_buf = device.buffer::<f32>(N_MLSM)?;
    let mut tgt_buf = device.buffer::<f32>(N_MLSM)?;
    let out_buf     = device.buffer::<f32>(N_MLSM)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::nll::MultilabelSoftMarginLossForward::new(BLOCK_SIZE_MLSM);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::nll::MultilabelSoftMarginLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                out_buf.as_device_ptr() as *mut f32, N_MLSM as i32);
    device.launch(&program, &mlsm_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_MLSM {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "multilabel_soft_margin_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_multilabel_soft_margin_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_nll/mlsm_dy.bin");
    let inp_host = load_fixture("loss_nll/mlsm_input.bin");
    let tgt_host = load_fixture("loss_nll/mlsm_target.bin");
    let expected = load_fixture("loss_nll/mlsm_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N_MLSM];

    let mut dy_buf  = device.buffer::<f32>(N_MLSM)?;
    let mut inp_buf = device.buffer::<f32>(N_MLSM)?;
    let mut tgt_buf = device.buffer::<f32>(N_MLSM)?;
    let dx_buf      = device.buffer::<f32>(N_MLSM)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::nll::MultilabelSoftMarginLossBackward::new(BLOCK_SIZE_MLSM);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::nn::loss::nll::MultilabelSoftMarginLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut f32, dx_buf.as_device_ptr() as *mut f32,
                N_MLSM as i32);
    device.launch(&program, &mlsm_launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N_MLSM {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "multilabel_soft_margin_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}
