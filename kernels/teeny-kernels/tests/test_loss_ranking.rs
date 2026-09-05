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
use teeny_test::{load_fixture, load_fixture_i32};

#[cfg(feature = "hardware")]
const N: usize = 1024;
const BLOCK_SIZE: i32 = 1024;
#[cfg(feature = "hardware")]
const N_ROWS: usize = 64;
#[cfg(feature = "hardware")]
const N_COLS: usize = 16;
const BLOCK_SIZE_MM: i32 = 16; // next_power_of_two(N_COLS)
#[cfg(feature = "hardware")]
const MARGIN: f32 = 1.0;
#[cfg(feature = "hardware")]
const PTX_THREADS: u32 = 128;

#[cfg(feature = "hardware")]
fn elem_launch_cfg() -> teeny_runtime::LaunchConfig {
    teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    )
}

#[cfg(feature = "hardware")]
fn row_launch_cfg() -> teeny_runtime::LaunchConfig {
    teeny_runtime::launch_config_custom([N_ROWS as u32, 1, 1], [PTX_THREADS, 1, 1], [1, 1, 1])
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_margin_ranking_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::ranking::MarginRankingLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "margin_ranking_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "margin_ranking_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_hinge_embedding_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::ranking::HingeEmbeddingLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "hinge_embedding_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "hinge_embedding_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_multi_margin_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::ranking::MultiMarginLossForward::new(BLOCK_SIZE_MM);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "multi_margin_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "multi_margin_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_margin_ranking_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let x1_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_x1.bin");
    let x2_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_x2.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/mrl_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut x1_buf = device.buffer::<f32>(N)?;
    let mut x2_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    x1_buf.to_device(&x1_host)?;
    x2_buf.to_device(&x2_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::ranking::MarginRankingLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::ranking::MarginRankingLossForward,
    >(&ptx_path)?;

    let args = (
        x1_buf.as_device_ptr(),
        x2_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
        MARGIN,
    );
    device.launch(&program, &elem_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "margin_ranking_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_margin_ranking_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_dy.bin");
    let x1_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_x1.bin");
    let x2_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_x2.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mrl_y.bin");
    let exp_dx1 = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/mrl_expected_dx1.bin",
    );
    let exp_dx2 = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/mrl_expected_dx2.bin",
    );
    let mut dx1_host = vec![0.0f32; N];
    let mut dx2_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut x1_buf = device.buffer::<f32>(N)?;
    let mut x2_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx1_buf = device.buffer::<f32>(N)?;
    let dx2_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    x1_buf.to_device(&x1_host)?;
    x2_buf.to_device(&x2_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::ranking::MarginRankingLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::ranking::MarginRankingLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        x1_buf.as_device_ptr(),
        x2_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx1_buf.as_device_ptr(),
        dx2_buf.as_device_ptr(),
        N as i32,
        MARGIN,
    );
    device.launch(&program, &elem_launch_cfg(), args)?;
    dx1_buf.to_host(&mut dx1_host)?;
    dx2_buf.to_host(&mut dx2_host)?;

    for i in 0..N {
        assert!(
            (dx1_host[i] - exp_dx1[i]).abs() < 1e-5,
            "mrl_backward dx1 mismatch at {i}: gpu={}, expected={}",
            dx1_host[i],
            exp_dx1[i]
        );
        assert!(
            (dx2_host[i] - exp_dx2[i]).abs() < 1e-5,
            "mrl_backward dx2 mismatch at {i}: gpu={}, expected={}",
            dx2_host[i],
            exp_dx2[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_hinge_embedding_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/hel_input.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/hel_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/hel_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::ranking::HingeEmbeddingLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::ranking::HingeEmbeddingLossForward,
    >(&ptx_path)?;

    let args = (
        inp_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
        MARGIN,
    );
    device.launch(&program, &elem_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "hinge_embedding_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_hinge_embedding_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/hel_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/hel_input.bin");
    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/hel_y.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/hel_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::loss::ranking::HingeEmbeddingLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::ranking::HingeEmbeddingLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
        MARGIN,
    );
    device.launch(&program, &elem_launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "hinge_embedding_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_multi_margin_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mm_input.bin");
    let tgt_host = load_fixture_i32(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mm_targets.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/mm_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N_ROWS];

    let mut inp_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut tgt_buf = device.buffer::<i32>(N_ROWS)?;
    let out_buf = device.buffer::<f32>(N_ROWS)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::ranking::MultiMarginLossForward::new(BLOCK_SIZE_MM);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::ranking::MultiMarginLossForward,
    >(&ptx_path)?;

    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N_ROWS as i32,
        N_COLS as i32,
        MARGIN,
    );
    device.launch(&program, &row_launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N_ROWS {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "multi_margin_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_multi_margin_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mm_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mm_input.bin");
    let tgt_host = load_fixture_i32(env!("CARGO_MANIFEST_DIR"), "loss_ranking/mm_targets.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_ranking/mm_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N_ROWS * N_COLS];

    let mut dy_buf = device.buffer::<f32>(N_ROWS)?;
    let mut inp_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    let mut tgt_buf = device.buffer::<i32>(N_ROWS)?;
    let mut dx_buf = device.buffer::<f32>(N_ROWS * N_COLS)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;
    dx_buf.to_device(&vec![0.0f32; N_ROWS * N_COLS])?;

    let kernel = teeny_kernels::nn::loss::ranking::MultiMarginLossBackward::new(BLOCK_SIZE_MM);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::ranking::MultiMarginLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N_ROWS as i32,
        N_COLS as i32,
        MARGIN,
    );
    device.launch(&program, &row_launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..(N_ROWS * N_COLS) {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "multi_margin_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
