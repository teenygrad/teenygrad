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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::device::program::Kernel;

#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;

use teeny_kernels::nn::tensor::elemwise_unary::{
    ElemwiseAbsBackward, ElemwiseAbsForward, ElemwiseAcosBackward, ElemwiseAcosForward,
    ElemwiseAcoshBackward, ElemwiseAcoshForward, ElemwiseAsinBackward, ElemwiseAsinForward,
    ElemwiseAsinhBackward, ElemwiseAsinhForward, ElemwiseAtanBackward, ElemwiseAtanForward,
    ElemwiseAtanhBackward, ElemwiseAtanhForward, ElemwiseCeilForward, ElemwiseCosBackward,
    ElemwiseCosForward, ElemwiseCoshBackward, ElemwiseCoshForward, ElemwiseErfBackward,
    ElemwiseErfForward, ElemwiseExpBackward, ElemwiseExpForward, ElemwiseFloorForward,
    ElemwiseIsnanForward, ElemwiseLogBackward, ElemwiseLogForward, ElemwiseNegBackward,
    ElemwiseNegForward, ElemwiseReciprocalBackward, ElemwiseReciprocalForward, ElemwiseSignForward,
    ElemwiseSinBackward, ElemwiseSinForward, ElemwiseSinhBackward, ElemwiseSinhForward,
    ElemwiseSqrtBackward, ElemwiseSqrtForward, ElemwiseTanBackward, ElemwiseTanForward,
};
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

const BLOCK_SIZE: i32 = 1024;
#[cfg(feature = "hardware")]
const TOL: f32 = 1e-4;

// ── Macro: source + MLIR snapshot (no CUDA required) ─────────────────────────

macro_rules! source_test {
    ($test_name:ident, $kernel_ty:ty, $snap_prefix:literal) => {
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let kernel = <$kernel_ty>::new(BLOCK_SIZE);
            let target = teeny_runtime::reference_target();
            let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
                &kernel, &target, true, false,
            )?);
            let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
            assert_debug_snapshot!(
                format!(
                    "{}_{}",
                    concat!($snap_prefix, "_source"),
                    teeny_runtime::BACKEND_NAME
                ),
                kernel.source()
            );
            assert_debug_snapshot!(
                format!(
                    "{}_{}",
                    concat!($snap_prefix, "_mlir"),
                    teeny_runtime::BACKEND_NAME
                ),
                mlir.trim()
            );
            Ok(())
        }
    };
}

// ── Macro: GPU forward test (x_ptr, y_ptr, n) ─────────────────────────────────

macro_rules! gpu_forward_test {
    ($test_name:ident, $kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "hardware")]
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let device = teeny_runtime::open()?;
            let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/x.bin");
            let expected = load_fixture(
                env!("CARGO_MANIFEST_DIR"),
                concat!("elemwise_unary/expected_", $fixture_op, ".bin"),
            );
            let n = x.len();
            let mut x_buf = device.buffer::<f32>(n)?;
            let y_buf = device.buffer::<f32>(n)?;
            let mut y_out = vec![0.0f32; n];
            x_buf.to_device(&x)?;
            let kernel = <$kernel_ty>::new(BLOCK_SIZE);
            let target = teeny_runtime::default_target(&device)?;
            let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
            let program = teeny_runtime::load_program::<$kernel_ty>(&ptx_path)?;
            let cfg = teeny_runtime::launch_config(n, &program);
            device.launch(
                &program,
                &cfg,
                (x_buf.as_device_ptr(), y_buf.as_device_ptr(), n as i32),
            )?;
            y_buf.to_host(&mut y_out)?;
            for i in 0..n {
                assert!(
                    (y_out[i] - expected[i]).abs() < TOL,
                    "{} fwd mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    y_out[i],
                    expected[i]
                );
            }
            Ok(())
        }
    };
}

// ── Macro: GPU backward test (dy_ptr, x_ptr, dx_ptr, n) ─────────────────────

macro_rules! gpu_backward_test {
    ($test_name:ident, $bwd_kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "hardware")]
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let device = teeny_runtime::open()?;
            let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/x.bin");
            let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/dy.bin");
            let expected = load_fixture(
                env!("CARGO_MANIFEST_DIR"),
                concat!("elemwise_unary/expected_", $fixture_op, "_backward.bin"),
            );
            let n = x.len();
            let mut x_buf = device.buffer::<f32>(n)?;
            let mut dy_buf = device.buffer::<f32>(n)?;
            let dx_buf = device.buffer::<f32>(n)?;
            let mut dx_out = vec![0.0f32; n];
            x_buf.to_device(&x)?;
            dy_buf.to_device(&dy)?;
            let kernel = <$bwd_kernel_ty>::new(BLOCK_SIZE);
            let target = teeny_runtime::default_target(&device)?;
            let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
            let program = teeny_runtime::load_program::<$bwd_kernel_ty>(&ptx_path)?;
            let cfg = teeny_runtime::launch_config(n, &program);
            device.launch(
                &program,
                &cfg,
                (
                    dy_buf.as_device_ptr(),
                    x_buf.as_device_ptr(),
                    dx_buf.as_device_ptr(),
                    n as i32,
                ),
            )?;
            dx_buf.to_host(&mut dx_out)?;
            for i in 0..n {
                assert!(
                    (dx_out[i] - expected[i]).abs() < TOL,
                    "{} bwd mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    dx_out[i],
                    expected[i]
                );
            }
            Ok(())
        }
    };
}

// ── Source + MLIR snapshots ───────────────────────────────────────────────────

source_test!(
    test_abs_source,
    ElemwiseAbsForward::<f32>,
    "elemwise_abs_forward"
);
source_test!(
    test_neg_source,
    ElemwiseNegForward::<f32>,
    "elemwise_neg_forward"
);
source_test!(
    test_sign_source,
    ElemwiseSignForward::<f32>,
    "elemwise_sign_forward"
);
source_test!(
    test_ceil_source,
    ElemwiseCeilForward::<f32>,
    "elemwise_ceil_forward"
);
source_test!(
    test_floor_source,
    ElemwiseFloorForward::<f32>,
    "elemwise_floor_forward"
);
source_test!(
    test_sqrt_source,
    ElemwiseSqrtForward::<f32>,
    "elemwise_sqrt_forward"
);
source_test!(
    test_reciprocal_source,
    ElemwiseReciprocalForward::<f32>,
    "elemwise_reciprocal_forward"
);
source_test!(
    test_exp_source,
    ElemwiseExpForward::<f32>,
    "elemwise_exp_forward"
);
source_test!(
    test_log_source,
    ElemwiseLogForward::<f32>,
    "elemwise_log_forward"
);
source_test!(
    test_erf_source,
    ElemwiseErfForward::<f32>,
    "elemwise_erf_forward"
);
source_test!(
    test_sin_source,
    ElemwiseSinForward::<f32>,
    "elemwise_sin_forward"
);
source_test!(
    test_cos_source,
    ElemwiseCosForward::<f32>,
    "elemwise_cos_forward"
);
source_test!(
    test_tan_source,
    ElemwiseTanForward::<f32>,
    "elemwise_tan_forward"
);
source_test!(
    test_asin_source,
    ElemwiseAsinForward::<f32>,
    "elemwise_asin_forward"
);
source_test!(
    test_acos_source,
    ElemwiseAcosForward::<f32>,
    "elemwise_acos_forward"
);
source_test!(
    test_atan_source,
    ElemwiseAtanForward::<f32>,
    "elemwise_atan_forward"
);
source_test!(
    test_sinh_source,
    ElemwiseSinhForward::<f32>,
    "elemwise_sinh_forward"
);
source_test!(
    test_cosh_source,
    ElemwiseCoshForward::<f32>,
    "elemwise_cosh_forward"
);
source_test!(
    test_asinh_source,
    ElemwiseAsinhForward::<f32>,
    "elemwise_asinh_forward"
);
source_test!(
    test_acosh_source,
    ElemwiseAcoshForward::<f32>,
    "elemwise_acosh_forward"
);
source_test!(
    test_atanh_source,
    ElemwiseAtanhForward::<f32>,
    "elemwise_atanh_forward"
);

// IsNaN uses a non-generic struct — write it out by hand
#[test]
fn test_isnan_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = ElemwiseIsnanForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "elemwise_isnan_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "elemwise_isnan_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

// Backward source + MLIR snapshots
source_test!(
    test_abs_backward_source,
    ElemwiseAbsBackward::<f32>,
    "elemwise_abs_backward"
);
source_test!(
    test_neg_backward_source,
    ElemwiseNegBackward::<f32>,
    "elemwise_neg_backward"
);
source_test!(
    test_sqrt_backward_source,
    ElemwiseSqrtBackward::<f32>,
    "elemwise_sqrt_backward"
);
source_test!(
    test_reciprocal_backward_source,
    ElemwiseReciprocalBackward::<f32>,
    "elemwise_reciprocal_backward"
);
source_test!(
    test_exp_backward_source,
    ElemwiseExpBackward::<f32>,
    "elemwise_exp_backward"
);
source_test!(
    test_log_backward_source,
    ElemwiseLogBackward::<f32>,
    "elemwise_log_backward"
);
source_test!(
    test_erf_backward_source,
    ElemwiseErfBackward::<f32>,
    "elemwise_erf_backward"
);
source_test!(
    test_sin_backward_source,
    ElemwiseSinBackward::<f32>,
    "elemwise_sin_backward"
);
source_test!(
    test_cos_backward_source,
    ElemwiseCosBackward::<f32>,
    "elemwise_cos_backward"
);
source_test!(
    test_tan_backward_source,
    ElemwiseTanBackward::<f32>,
    "elemwise_tan_backward"
);
source_test!(
    test_asin_backward_source,
    ElemwiseAsinBackward::<f32>,
    "elemwise_asin_backward"
);
source_test!(
    test_acos_backward_source,
    ElemwiseAcosBackward::<f32>,
    "elemwise_acos_backward"
);
source_test!(
    test_atan_backward_source,
    ElemwiseAtanBackward::<f32>,
    "elemwise_atan_backward"
);
source_test!(
    test_sinh_backward_source,
    ElemwiseSinhBackward::<f32>,
    "elemwise_sinh_backward"
);
source_test!(
    test_cosh_backward_source,
    ElemwiseCoshBackward::<f32>,
    "elemwise_cosh_backward"
);
source_test!(
    test_asinh_backward_source,
    ElemwiseAsinhBackward::<f32>,
    "elemwise_asinh_backward"
);
source_test!(
    test_acosh_backward_source,
    ElemwiseAcoshBackward::<f32>,
    "elemwise_acosh_backward"
);
source_test!(
    test_atanh_backward_source,
    ElemwiseAtanhBackward::<f32>,
    "elemwise_atanh_backward"
);

// ── GPU forward tests ─────────────────────────────────────────────────────────

gpu_forward_test!(
    test_abs_forward_gpu,
    ElemwiseAbsForward::<f32>,
    "abs",
    "abs"
);
gpu_forward_test!(
    test_neg_forward_gpu,
    ElemwiseNegForward::<f32>,
    "neg",
    "neg"
);
gpu_forward_test!(
    test_sign_forward_gpu,
    ElemwiseSignForward::<f32>,
    "sign",
    "sign"
);
gpu_forward_test!(
    test_ceil_forward_gpu,
    ElemwiseCeilForward::<f32>,
    "ceil",
    "ceil"
);
gpu_forward_test!(
    test_floor_forward_gpu,
    ElemwiseFloorForward::<f32>,
    "floor",
    "floor"
);
gpu_forward_test!(
    test_sqrt_forward_gpu,
    ElemwiseSqrtForward::<f32>,
    "sqrt",
    "sqrt"
);
gpu_forward_test!(
    test_reciprocal_forward_gpu,
    ElemwiseReciprocalForward::<f32>,
    "reciprocal",
    "reciprocal"
);
gpu_forward_test!(
    test_exp_forward_gpu,
    ElemwiseExpForward::<f32>,
    "exp",
    "exp"
);
gpu_forward_test!(
    test_log_forward_gpu,
    ElemwiseLogForward::<f32>,
    "log",
    "log"
);
gpu_forward_test!(
    test_erf_forward_gpu,
    ElemwiseErfForward::<f32>,
    "erf",
    "erf"
);
gpu_forward_test!(
    test_sin_forward_gpu,
    ElemwiseSinForward::<f32>,
    "sin",
    "sin"
);
gpu_forward_test!(
    test_cos_forward_gpu,
    ElemwiseCosForward::<f32>,
    "cos",
    "cos"
);
gpu_forward_test!(
    test_tan_forward_gpu,
    ElemwiseTanForward::<f32>,
    "tan",
    "tan"
);
gpu_forward_test!(
    test_asin_forward_gpu,
    ElemwiseAsinForward::<f32>,
    "asin",
    "asin"
);
gpu_forward_test!(
    test_acos_forward_gpu,
    ElemwiseAcosForward::<f32>,
    "acos",
    "acos"
);
gpu_forward_test!(
    test_atan_forward_gpu,
    ElemwiseAtanForward::<f32>,
    "atan",
    "atan"
);
gpu_forward_test!(
    test_sinh_forward_gpu,
    ElemwiseSinhForward::<f32>,
    "sinh",
    "sinh"
);
gpu_forward_test!(
    test_cosh_forward_gpu,
    ElemwiseCoshForward::<f32>,
    "cosh",
    "cosh"
);
gpu_forward_test!(
    test_asinh_forward_gpu,
    ElemwiseAsinhForward::<f32>,
    "asinh",
    "asinh"
);
// acosh requires x >= 1 — uses x_acosh.bin not x.bin
#[cfg(feature = "hardware")]
#[test]
fn test_acosh_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/x_acosh.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "elemwise_unary/expected_acosh.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    let kernel = ElemwiseAcoshForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ElemwiseAcoshForward<f32>>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), n as i32),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "acosh fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}
gpu_forward_test!(
    test_atanh_forward_gpu,
    ElemwiseAtanhForward::<f32>,
    "atanh",
    "atanh"
);

// IsNaN uses x_with_nan.bin (contains some NaNs)
#[cfg(feature = "hardware")]
#[test]
fn test_isnan_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/x_with_nan.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "elemwise_unary/expected_isnan.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    let kernel = ElemwiseIsnanForward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ElemwiseIsnanForward<f32>>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (x_buf.as_device_ptr(), y_buf.as_device_ptr(), n as i32),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "isnan fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

// ── GPU backward tests ────────────────────────────────────────────────────────

gpu_backward_test!(
    test_abs_backward_gpu,
    ElemwiseAbsBackward::<f32>,
    "abs",
    "abs"
);
gpu_backward_test!(
    test_sqrt_backward_gpu,
    ElemwiseSqrtBackward::<f32>,
    "sqrt",
    "sqrt"
);
gpu_backward_test!(
    test_reciprocal_backward_gpu,
    ElemwiseReciprocalBackward::<f32>,
    "reciprocal",
    "reciprocal"
);
gpu_backward_test!(
    test_exp_backward_gpu,
    ElemwiseExpBackward::<f32>,
    "exp",
    "exp"
);
gpu_backward_test!(
    test_log_backward_gpu,
    ElemwiseLogBackward::<f32>,
    "log",
    "log"
);
gpu_backward_test!(
    test_erf_backward_gpu,
    ElemwiseErfBackward::<f32>,
    "erf",
    "erf"
);
gpu_backward_test!(
    test_sin_backward_gpu,
    ElemwiseSinBackward::<f32>,
    "sin",
    "sin"
);
gpu_backward_test!(
    test_cos_backward_gpu,
    ElemwiseCosBackward::<f32>,
    "cos",
    "cos"
);
gpu_backward_test!(
    test_tan_backward_gpu,
    ElemwiseTanBackward::<f32>,
    "tan",
    "tan"
);
gpu_backward_test!(
    test_asin_backward_gpu,
    ElemwiseAsinBackward::<f32>,
    "asin",
    "asin"
);
gpu_backward_test!(
    test_acos_backward_gpu,
    ElemwiseAcosBackward::<f32>,
    "acos",
    "acos"
);
gpu_backward_test!(
    test_atan_backward_gpu,
    ElemwiseAtanBackward::<f32>,
    "atan",
    "atan"
);
gpu_backward_test!(
    test_sinh_backward_gpu,
    ElemwiseSinhBackward::<f32>,
    "sinh",
    "sinh"
);
gpu_backward_test!(
    test_cosh_backward_gpu,
    ElemwiseCoshBackward::<f32>,
    "cosh",
    "cosh"
);
gpu_backward_test!(
    test_asinh_backward_gpu,
    ElemwiseAsinhBackward::<f32>,
    "asinh",
    "asinh"
);
// acosh backward: uses x_acosh.bin (domain x >= 1)
#[cfg(feature = "hardware")]
#[test]
fn test_acosh_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/x_acosh.bin");
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "elemwise_unary/expected_acosh_backward.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut dy_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    dy_buf.to_device(&dy)?;
    let kernel = ElemwiseAcoshBackward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ElemwiseAcoshBackward<f32>>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            n as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "acosh bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}
gpu_backward_test!(
    test_atanh_backward_gpu,
    ElemwiseAtanhBackward::<f32>,
    "atanh",
    "atanh"
);

// Neg backward: signature is (dy_ptr, dx_ptr, n) — no x saved
#[cfg(feature = "hardware")]
#[test]
fn test_neg_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "elemwise_unary/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "elemwise_unary/expected_neg_backward.bin",
    );
    let n = dy.len();
    let mut dy_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    dy_buf.to_device(&dy)?;
    let kernel = ElemwiseNegBackward::<f32>::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ElemwiseNegBackward<f32>>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (dy_buf.as_device_ptr(), dx_buf.as_device_ptr(), n as i32),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "neg bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}
