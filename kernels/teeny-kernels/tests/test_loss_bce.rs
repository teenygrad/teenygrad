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
const N: usize = 1024;
const BLOCK_SIZE: i32 = 1024;
#[cfg(feature = "hardware")]
const PTX_THREADS: u32 = 128;

#[cfg(feature = "hardware")]
fn launch_cfg() -> teeny_runtime::LaunchConfig {
    teeny_runtime::launch_config_custom(
        [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        [PTX_THREADS, 1, 1],
        [1, 1, 1],
    )
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_bce_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::bce::BceLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("bce_loss_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("bce_loss_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_bce_with_logits_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::bce::BceWithLogitsLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "bce_with_logits_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "bce_with_logits_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_soft_margin_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::bce::SoftMarginLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "soft_margin_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "soft_margin_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_kl_div_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::bce::KlDivLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("kl_div_loss_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("kl_div_loss_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_poisson_nll_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::bce::PoissonNllLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "poisson_nll_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "poisson_nll_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_gaussian_nll_loss_mlir() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::loss::bce::GaussianNllLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "gaussian_nll_loss_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "gaussian_nll_loss_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "hardware")]
fn test_bce_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bce_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bce_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/bce_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::BceLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::loss::bce::BceLossForward>(&ptx_path)?;

    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "bce_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_bce_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bce_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bce_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bce_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/bce_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::BceLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::loss::bce::BceLossBackward>(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "bce_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_bce_with_logits_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bwl_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bwl_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/bwl_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::BceWithLogitsLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::bce::BceWithLogitsLossForward,
    >(&ptx_path)?;

    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "bce_with_logits_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_bce_with_logits_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bwl_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bwl_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/bwl_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/bwl_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::BceWithLogitsLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::bce::BceWithLogitsLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "bce_with_logits_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_soft_margin_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/sml_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/sml_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/sml_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::SoftMarginLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<teeny_kernels::nn::loss::bce::SoftMarginLossForward>(
        &ptx_path,
    )?;

    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "soft_margin_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_soft_margin_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/sml_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/sml_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/sml_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/sml_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::SoftMarginLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::bce::SoftMarginLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "soft_margin_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_kl_div_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/kl_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/kl_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/kl_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::KlDivLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::loss::bce::KlDivLossForward>(&ptx_path)?;

    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "kl_div_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_kl_div_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/kl_dy.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/kl_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/kl_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::KlDivLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program =
        teeny_runtime::load_program::<teeny_kernels::nn::loss::bce::KlDivLossBackward>(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "kl_div_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_poisson_nll_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/pnll_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/pnll_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/pnll_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::PoissonNllLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<teeny_kernels::nn::loss::bce::PoissonNllLossForward>(
        &ptx_path,
    )?;

    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "poisson_nll_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_poisson_nll_loss_backward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/pnll_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/pnll_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/pnll_target.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/pnll_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::nn::loss::bce::PoissonNllLossBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::bce::PoissonNllLossBackward,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "poisson_nll_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_gaussian_nll_loss_forward_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_target.bin");
    let var_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_var.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/gnll_expected_forward.bin",
    );
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let mut var_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;
    var_buf.to_device(&var_host)?;

    let kernel = teeny_kernels::nn::loss::bce::GaussianNllLossForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::bce::GaussianNllLossForward,
    >(&ptx_path)?;

    let cfg = launch_cfg();
    let args = (
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        var_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        N as i32,
        1e-6_f32,
    );
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "gaussian_nll_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "hardware")]
fn test_gaussian_nll_loss_backward_input_cuda() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_dy.bin");
    let inp_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_input.bin");
    let tgt_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_target.bin");
    let var_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "loss_bce/gnll_var.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "loss_bce/gnll_expected_backward.bin",
    );
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let mut var_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;
    var_buf.to_device(&var_host)?;

    let kernel = teeny_kernels::nn::loss::bce::GaussianNllLossBackwardInput::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::loss::bce::GaussianNllLossBackwardInput,
    >(&ptx_path)?;

    let args = (
        dy_buf.as_device_ptr(),
        inp_buf.as_device_ptr(),
        tgt_buf.as_device_ptr(),
        var_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
        N as i32,
        1e-6_f32,
    );
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "gaussian_nll_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i],
            expected[i]
        );
    }
    Ok(())
}
