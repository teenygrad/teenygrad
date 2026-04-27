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

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, device::CudaLaunchConfig, errors::Result, testing};

const N: usize = 1024;
const BLOCK_SIZE: i32 = 1024;
const PTX_THREADS: u32 = 128;

fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn launch_cfg() -> CudaLaunchConfig {
    CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [PTX_THREADS, 1, 1],
        cluster: [1, 1, 1],
    }
}

// ── MLIR snapshot tests ───────────────────────────────────────────────────────

#[test]
fn test_bce_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::bce::BceLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("bce_loss_forward_source", kernel.source());
    assert_debug_snapshot!("bce_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_bce_with_logits_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::bce::BceWithLogitsLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("bce_with_logits_loss_forward_source", kernel.source());
    assert_debug_snapshot!("bce_with_logits_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_soft_margin_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::bce::SoftMarginLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("soft_margin_loss_forward_source", kernel.source());
    assert_debug_snapshot!("soft_margin_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_kl_div_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::bce::KlDivLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("kl_div_loss_forward_source", kernel.source());
    assert_debug_snapshot!("kl_div_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_poisson_nll_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::bce::PoissonNllLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("poisson_nll_loss_forward_source", kernel.source());
    assert_debug_snapshot!("poisson_nll_loss_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_gaussian_nll_loss_mlir() -> anyhow::Result<()> {
    dotenv()?;
    let kernel = teeny_kernels::loss::bce::GaussianNllLossForward::new(BLOCK_SIZE);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("gaussian_nll_loss_forward_source", kernel.source());
    assert_debug_snapshot!("gaussian_nll_loss_forward_mlir",   mlir.trim());
    Ok(())
}

// ── CUDA integration tests ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_bce_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_bce/bce_input.bin");
    let tgt_host = load_fixture("loss_bce/bce_target.bin");
    let expected  = load_fixture("loss_bce/bce_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::BceLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::BceLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                out_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "bce_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_bce_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_bce/bce_dy.bin");
    let inp_host = load_fixture("loss_bce/bce_input.bin");
    let tgt_host = load_fixture("loss_bce/bce_target.bin");
    let expected  = load_fixture("loss_bce/bce_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf  = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::BceLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::BceLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut f32, dx_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "bce_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_bce_with_logits_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_bce/bwl_input.bin");
    let tgt_host = load_fixture("loss_bce/bwl_target.bin");
    let expected  = load_fixture("loss_bce/bwl_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::BceWithLogitsLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::BceWithLogitsLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                out_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "bce_with_logits_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_bce_with_logits_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_bce/bwl_dy.bin");
    let inp_host = load_fixture("loss_bce/bwl_input.bin");
    let tgt_host = load_fixture("loss_bce/bwl_target.bin");
    let expected  = load_fixture("loss_bce/bwl_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf  = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::BceWithLogitsLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::BceWithLogitsLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut f32, dx_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "bce_with_logits_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_soft_margin_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_bce/sml_input.bin");
    let tgt_host = load_fixture("loss_bce/sml_target.bin");
    let expected  = load_fixture("loss_bce/sml_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::SoftMarginLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::SoftMarginLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                out_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-5,
            "soft_margin_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_soft_margin_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_bce/sml_dy.bin");
    let inp_host = load_fixture("loss_bce/sml_input.bin");
    let tgt_host = load_fixture("loss_bce/sml_target.bin");
    let expected  = load_fixture("loss_bce/sml_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf  = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::SoftMarginLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::SoftMarginLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut f32, dx_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "soft_margin_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_kl_div_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_bce/kl_input.bin");
    let tgt_host = load_fixture("loss_bce/kl_target.bin");
    let expected  = load_fixture("loss_bce/kl_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::KlDivLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::KlDivLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                out_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "kl_div_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_kl_div_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_bce/kl_dy.bin");
    let tgt_host = load_fixture("loss_bce/kl_target.bin");
    let expected  = load_fixture("loss_bce/kl_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf  = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::KlDivLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::KlDivLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                dx_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-5,
            "kl_div_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_poisson_nll_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_bce/pnll_input.bin");
    let tgt_host = load_fixture("loss_bce/pnll_target.bin");
    let expected  = load_fixture("loss_bce/pnll_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::PoissonNllLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::PoissonNllLossForward>(&ptx)?;

    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                out_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "poisson_nll_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_poisson_nll_loss_backward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_bce/pnll_dy.bin");
    let inp_host = load_fixture("loss_bce/pnll_input.bin");
    let tgt_host = load_fixture("loss_bce/pnll_target.bin");
    let expected  = load_fixture("loss_bce/pnll_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf  = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;

    let kernel = teeny_kernels::loss::bce::PoissonNllLossBackward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::PoissonNllLossBackward>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut f32, dx_buf.as_device_ptr() as *mut f32, N as i32);
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "poisson_nll_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_gaussian_nll_loss_forward_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp_host = load_fixture("loss_bce/gnll_input.bin");
    let tgt_host = load_fixture("loss_bce/gnll_target.bin");
    let var_host = load_fixture("loss_bce/gnll_var.bin");
    let expected  = load_fixture("loss_bce/gnll_expected_forward.bin");
    let mut out_host = vec![0.0f32; N];

    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let mut var_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;
    var_buf.to_device(&var_host)?;

    let kernel = teeny_kernels::loss::bce::GaussianNllLossForward::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::GaussianNllLossForward>(&ptx)?;

    let cfg = launch_cfg();
    let args = (inp_buf.as_device_ptr() as *mut f32, tgt_buf.as_device_ptr() as *mut f32,
                var_buf.as_device_ptr() as *mut f32, out_buf.as_device_ptr() as *mut f32,
                N as i32, 1e-6_f32);
    device.launch(&program, &cfg, args)?;
    out_buf.to_host(&mut out_host)?;

    for i in 0..N {
        assert!(
            (out_host[i] - expected[i]).abs() < 1e-4,
            "gaussian_nll_loss_forward mismatch at {i}: gpu={}, expected={}",
            out_host[i], expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_gaussian_nll_loss_backward_input_cuda() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let dy_host  = load_fixture("loss_bce/gnll_dy.bin");
    let inp_host = load_fixture("loss_bce/gnll_input.bin");
    let tgt_host = load_fixture("loss_bce/gnll_target.bin");
    let var_host = load_fixture("loss_bce/gnll_var.bin");
    let expected  = load_fixture("loss_bce/gnll_expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf  = device.buffer::<f32>(N)?;
    let mut inp_buf = device.buffer::<f32>(N)?;
    let mut tgt_buf = device.buffer::<f32>(N)?;
    let mut var_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;
    dy_buf.to_device(&dy_host)?;
    inp_buf.to_device(&inp_host)?;
    tgt_buf.to_device(&tgt_host)?;
    var_buf.to_device(&var_host)?;

    let kernel = teeny_kernels::loss::bce::GaussianNllLossBackwardInput::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<teeny_kernels::loss::bce::GaussianNllLossBackwardInput>(&ptx)?;

    let args = (dy_buf.as_device_ptr() as *mut f32, inp_buf.as_device_ptr() as *mut f32,
                tgt_buf.as_device_ptr() as *mut f32, var_buf.as_device_ptr() as *mut f32,
                dx_buf.as_device_ptr() as *mut f32, N as i32, 1e-6_f32);
    device.launch(&program, &launch_cfg(), args)?;
    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert!(
            (dx_host[i] - expected[i]).abs() < 1e-4,
            "gaussian_nll_loss_backward mismatch at {i}: gpu={}, expected={}",
            dx_host[i], expected[i]
        );
    }
    Ok(())
}
