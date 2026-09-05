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
use std::path::PathBuf;
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const N_ROWS: usize = 64;
#[cfg(feature = "hardware")]
const N_DIM: usize = 64;
const BLOCK_SIZE: i32 = 64; // next_power_of_two(N_DIM)
#[cfg(feature = "hardware")]
const MARGIN: f32 = 0.5;
#[cfg(feature = "hardware")]
const EPS: f32 = 1e-6;
#[cfg(feature = "hardware")]
const PTX_THREADS: u32 = 128;

#[cfg(feature = "hardware")]
fn row_launch_cfg() -> teeny_runtime::LaunchConfig {
    teeny_runtime::launch_config_custom([N_ROWS as u32, 1, 1], [PTX_THREADS, 1, 1], [1, 1, 1])
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_cosine_embedding_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::embedding::CosineEmbeddingLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "cosine_embedding_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "cosine_embedding_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_triplet_margin_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::embedding::TripletMarginLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "triplet_margin_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "triplet_margin_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_cosine_embedding_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x1_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_x1.bin");
    let x2_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_x2.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/cel_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut x1_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut x2_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut y_buf = device.buffer::<f32>(N_ROWS)?;
    let out_buf = device.buffer::<f32>(N_ROWS)?;
    x1_buf.to_device(&x1_host)?;
    x2_buf.to_device(&x2_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::embedding::CosineEmbeddingLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::embedding::CosineEmbeddingLossForward,
    >(&ptx_path)?;

    let args = (
        x1_buf.as_device_ptr(),
        x2_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N_ROWS as i32,
        N_DIM as i32,
        MARGIN,
    );
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "cosine_embedding_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_cosine_embedding_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_dy.bin");
    let x1_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_x1.bin");
    let x2_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_x2.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/cel_y.bin");
    let exp_dx1 = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/cel_expected_dx1.bin",
    );
    let exp_dx2 = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/cel_expected_dx2.bin",
    );
    let mut dx1_host = vec![0.0f32; N_ROWS * N_DIM];
    let mut dx2_host = vec![0.0f32; N_ROWS * N_DIM];

    let mut dy_buf = device.buffer::<f32>(N_ROWS)?;
    let mut x1_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut x2_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut y_buf = device.buffer::<f32>(N_ROWS)?;
    let dx1_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let dx2_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    dy_buf.to_device(&dy_host)?;
    x1_buf.to_device(&x1_host)?;
    x2_buf.to_device(&x2_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::embedding::CosineEmbeddingLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::embedding::CosineEmbeddingLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        x1_buf.as_device_ptr(),
        x2_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx1_buf.as_device_ptr(),
        dx2_buf.as_device_ptr(),
        N_ROWS as i32,
        N_DIM as i32,
        MARGIN,
    );
    device.launch(&program, &row_launch_cfg(), args)?;
    dx1_buf.to_host(&mut dx1_host)?;
    dx2_buf.to_host(&mut dx2_host)?;

    for i in 0..(N_ROWS * N_DIM) {
        assert!(
            (dx1_host[i] - exp_dx1[i]).abs() < 1e-4,
            "cel_backward dx1 mismatch at {i}: gpu={}, expected={}",
            dx1_host[i],
            exp_dx1[i]
        );
        assert!(
            (dx2_host[i] - exp_dx2[i]).abs() < 1e-4,
            "cel_backward dx2 mismatch at {i}: gpu={}, expected={}",
            dx2_host[i],
            exp_dx2[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_triplet_margin_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let a_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/tml_anchor.bin");
    let p_host = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_positive.bin",
    );
    let n_host = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_negative.bin",
    );
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut a_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut p_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut n_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let out_buf = device.buffer::<f32>(N_ROWS)?;
    a_buf.to_device(&a_host)?;
    p_buf.to_device(&p_host)?;
    n_buf.to_device(&n_host)?;

    let kernel = teeny_kernels::nn::loss::embedding::TripletMarginLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::embedding::TripletMarginLossForward,
    >(&ptx_path)?;

    let args = (
        a_buf.as_device_ptr(),
        p_buf.as_device_ptr(),
        n_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N_ROWS as i32,
        N_DIM as i32,
        MARGIN,
        EPS,
    );
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "triplet_margin_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_triplet_margin_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/tml_dy.bin");
    let a_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_embedding/tml_anchor.bin");
    let p_host = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_positive.bin",
    );
    let n_host = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_negative.bin",
    );
    let exp_da = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_expected_da.bin",
    );
    let exp_dp = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_expected_dp.bin",
    );
    let exp_dn = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_embedding/tml_expected_dn.bin",
    );
    let mut da_host = vec![0.0f32; N_ROWS * N_DIM];
    let mut dp_host = vec![0.0f32; N_ROWS * N_DIM];
    let mut dn_host = vec![0.0f32; N_ROWS * N_DIM];

    let mut dy_buf = device.buffer::<f32>(N_ROWS)?;
    let mut a_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut p_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let mut n_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let da_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let dp_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    let dn_buf = device.buffer::<f32>(N_ROWS * N_DIM)?;
    dy_buf.to_device(&dy_host)?;
    a_buf.to_device(&a_host)?;
    p_buf.to_device(&p_host)?;
    n_buf.to_device(&n_host)?;

    let kernel = teeny_kernels::nn::loss::embedding::TripletMarginLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::embedding::TripletMarginLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        a_buf.as_device_ptr(),
        p_buf.as_device_ptr(),
        n_buf.as_device_ptr(),
        da_buf.as_device_ptr(),
        dp_buf.as_device_ptr(),
        dn_buf.as_device_ptr(),
        N_ROWS as i32,
        N_DIM as i32,
        MARGIN,
        EPS,
    );
    device.launch(&program, &row_launch_cfg(), args)?;
    da_buf.to_host(&mut da_host)?;
    dp_buf.to_host(&mut dp_host)?;
    dn_buf.to_host(&mut dn_host)?;

    for i in 0..(N_ROWS * N_DIM) {
        assert!(
            (da_host[i] - exp_da[i]).abs() < 1e-4,
            "tml_backward da mismatch at {i}: gpu={}, expected={}",
            da_host[i],
            exp_da[i]
        );
        assert!(
            (dp_host[i] - exp_dp[i]).abs() < 1e-4,
            "tml_backward dp mismatch at {i}: gpu={}, expected={}",
            dp_host[i],
            exp_dp[i]
        );
        assert!(
            (dn_host[i] - exp_dn[i]).abs() < 1e-4,
            "tml_backward dn mismatch at {i}: gpu={}, expected={}",
            dn_host[i],
            exp_dn[i]
        );
    }
    Ok(())
}
