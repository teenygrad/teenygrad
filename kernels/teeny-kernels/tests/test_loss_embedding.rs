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
const N_DIM: usize = 64;
const BLOCK_SIZE: i32 = 64; // next_power_of_two(N_DIM)
const MARGIN: f32 = 0.5;
const EPS: f32 = 1e-6;
const PTX_THREADS: u32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn row_launch_cfg() -> CudaLaunchConfig {
    CudaLaunchConfig {
        grid: [N_ROWS as u32, 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    }
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_cosine_embedding_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::embedding::CosineEmbeddingLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("cosine_embedding_loss_forward_source", kernel.source());
    assert_debug_snapshot!("cosine_embedding_loss_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_triplet_margin_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::embedding::TripletMarginLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("triplet_margin_loss_forward_source", kernel.source());
    assert_debug_snapshot!("triplet_margin_loss_forward_mlir", mlir.trim());
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_cosine_embedding_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let x1_host = load_fixture("loss_embedding/cel_x1.bin");
    let x2_host = load_fixture("loss_embedding/cel_x2.bin");
    let y_host  = load_fixture("loss_embedding/cel_y.bin");
    let expected = load_fixture("loss_embedding/cel_expected_forward.bin");
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut x1_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut x2_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut y_buf  = device.buffer::<f32>(N_ROWS)?;
    let out_buf    = device.buffer::<f32>(N_ROWS)?;
    x1_buf.to_device(&x1_host)?;
    x2_buf.to_device(&x2_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::loss::embedding::CosineEmbeddingLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::embedding::CosineEmbeddingLossForward>(&ptx)?;

    let args = (x1_buf.as_device_ptr() as *mut f32, x2_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32, out_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32, N_DIM as i32, MARGIN);
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "cosine_embedding_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cosine_embedding_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_embedding/cel_dy.bin");
    let x1_host  = load_fixture("loss_embedding/cel_x1.bin");
    let x2_host  = load_fixture("loss_embedding/cel_x2.bin");
    let y_host   = load_fixture("loss_embedding/cel_y.bin");
    let exp_dx1  = load_fixture("loss_embedding/cel_expected_dx1.bin");
    let exp_dx2  = load_fixture("loss_embedding/cel_expected_dx2.bin");
    let mut dx1_host = vec![0.0f32; N_ROWS * N_DIM];
    let mut dx2_host = vec![0.0f32; N_ROWS * N_DIM];

    let mut dy_buf  = device.buffer::<f32>(N_ROWS)?;
    let mut x1_buf  = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut x2_buf  = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut y_buf   = device.buffer::<f32>(N_ROWS)?;
    let dx1_buf     = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let dx2_buf     = device.buffer::<f32>(N_ROWS * N_DIM)?;
    dy_buf.to_device(&dy_host)?;
    x1_buf.to_device(&x1_host)?;
    x2_buf.to_device(&x2_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::loss::embedding::CosineEmbeddingLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::embedding::CosineEmbeddingLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, x1_buf.as_device_ptr() as *mut f32,
                x2_buf.as_device_ptr() as *mut f32, y_buf.as_device_ptr() as *mut f32,
                dx1_buf.as_device_ptr() as *mut f32, dx2_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32, N_DIM as i32, MARGIN);
    device.launch(&program, &row_launch_cfg(), args)?;
    dx1_buf.to_host(&mut dx1_host)?;
    dx2_buf.to_host(&mut dx2_host)?;

    for i in 0..(N_ROWS * N_DIM) {
        assert!(
            (dx1_host[i] - exp_dx1[i]).abs() < 1e-4,
            "cel_backward dx1 mismatch at {i}: gpu={}, expected={}", dx1_host[i], exp_dx1[i]
        );
        assert!(
            (dx2_host[i] - exp_dx2[i]).abs() < 1e-4,
            "cel_backward dx2 mismatch at {i}: gpu={}, expected={}", dx2_host[i], exp_dx2[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_triplet_margin_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let a_host  = load_fixture("loss_embedding/tml_anchor.bin");
    let p_host  = load_fixture("loss_embedding/tml_positive.bin");
    let n_host  = load_fixture("loss_embedding/tml_negative.bin");
    let expected = load_fixture("loss_embedding/tml_expected_forward.bin");
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut a_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut p_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut n_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let out_buf   = device.buffer::<f32>(N_ROWS)?;
    a_buf.to_device(&a_host)?;
    p_buf.to_device(&p_host)?;
    n_buf.to_device(&n_host)?;

    let kernel = teeny_kernels::loss::embedding::TripletMarginLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::embedding::TripletMarginLossForward>(&ptx)?;

    let args = (a_buf.as_device_ptr() as *mut f32, p_buf.as_device_ptr() as *mut f32,
                n_buf.as_device_ptr() as *mut f32, out_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32, N_DIM as i32, MARGIN, EPS);
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "triplet_margin_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_triplet_margin_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_embedding/tml_dy.bin");
    let a_host   = load_fixture("loss_embedding/tml_anchor.bin");
    let p_host   = load_fixture("loss_embedding/tml_positive.bin");
    let n_host   = load_fixture("loss_embedding/tml_negative.bin");
    let exp_da   = load_fixture("loss_embedding/tml_expected_da.bin");
    let exp_dp   = load_fixture("loss_embedding/tml_expected_dp.bin");
    let exp_dn   = load_fixture("loss_embedding/tml_expected_dn.bin");
    let mut da_host = vec![0.0f32; N_ROWS * N_DIM];
    let mut dp_host = vec![0.0f32; N_ROWS * N_DIM];
    let mut dn_host = vec![0.0f32; N_ROWS * N_DIM];

    let mut dy_buf = device.buffer::<f32>(N_ROWS)?;
    let mut a_buf  = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut p_buf  = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut n_buf  = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let da_buf     = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let dp_buf     = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let dn_buf     = device.buffer::<f32>(N_ROWS * N_DIM)?;
    dy_buf.to_device(&dy_host)?;
    a_buf.to_device(&a_host)?;
    p_buf.to_device(&p_host)?;
    n_buf.to_device(&n_host)?;

    let kernel = teeny_kernels::loss::embedding::TripletMarginLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::embedding::TripletMarginLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, a_buf.as_device_ptr() as *mut f32,
                p_buf.as_device_ptr() as *mut f32, n_buf.as_device_ptr() as *mut f32,
                da_buf.as_device_ptr() as *mut f32, dp_buf.as_device_ptr() as *mut f32,
                dn_buf.as_device_ptr() as *mut f32,
                N_ROWS as i32, N_DIM as i32, MARGIN, EPS);
    device.launch(&program, &row_launch_cfg(), args)?;
    da_buf.to_host(&mut da_host)?;
    dp_buf.to_host(&mut dp_host)?;
    dn_buf.to_host(&mut dn_host)?;

    for i in 0..(N_ROWS * N_DIM) {
        assert!(
            (da_host[i] - exp_da[i]).abs() < 1e-4,
            "tml_backward da mismatch at {i}: gpu={}, expected={}", da_host[i], exp_da[i]
        );
        assert!(
            (dp_host[i] - exp_dp[i]).abs() < 1e-4,
            "tml_backward dp mismatch at {i}: gpu={}, expected={}", dp_host[i], exp_dp[i]
        );
        assert!(
            (dn_host[i] - exp_dn[i]).abs() < 1e-4,
            "tml_backward dn mismatch at {i}: gpu={}, expected={}", dn_host[i], exp_dn[i]
        );
    }
    Ok(())
}
