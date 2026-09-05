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

use teeny_kernels::nn::activation::extra::{
    LogSoftmaxBackward, LogSoftmaxForward, PreluBackward, PreluForward, ShrinkBackward,
    ShrinkForward, SwishBackward, SwishForward, ThresholdedReluBackward, ThresholdedReluForward,
};
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

const BLOCK_SIZE: i32 = 1024;
#[cfg(feature = "hardware")]
const TOL: f32 = 1e-4;

// ── Source + MLIR snapshots ───────────────────────────────────────────────────

#[test]
fn test_swish_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = SwishForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("swish_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("swish_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_swish_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = SwishBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("swish_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("swish_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_prelu_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = PreluForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("prelu_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("prelu_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_prelu_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = PreluBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("prelu_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("prelu_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// LogSoftmax uses n_cols=128 as BLOCK_SIZE (power of 2 matching input width)
const LOG_SOFTMAX_COLS: i32 = 128;

#[test]
fn test_log_softmax_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = LogSoftmaxForward::new(LOG_SOFTMAX_COLS);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("log_softmax_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("log_softmax_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_log_softmax_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = LogSoftmaxBackward::new(LOG_SOFTMAX_COLS);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "log_softmax_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("log_softmax_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_thresholded_relu_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = ThresholdedReluForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "thresholded_relu_forward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "thresholded_relu_forward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_thresholded_relu_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = ThresholdedReluBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!(
            "thresholded_relu_backward_source_{}",
            teeny_runtime::BACKEND_NAME
        ),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!(
            "thresholded_relu_backward_mlir_{}",
            teeny_runtime::BACKEND_NAME
        ),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_shrink_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = ShrinkForward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("shrink_forward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("shrink_forward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

#[test]
fn test_shrink_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = ShrinkBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::reference_target();
    let ptx_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!(
        format!("shrink_backward_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("shrink_backward_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );
    Ok(())
}

// ── GPU forward tests ─────────────────────────────────────────────────────────

// Swish forward: (x_ptr, y_ptr, n)
#[cfg(feature = "hardware")]
#[test]
fn test_swish_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_swish.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    let kernel = SwishForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<SwishForward>(&ptx_path)?;
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
            "swish fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

// Swish backward: (dy_ptr, x_ptr, dx_ptr, n)
#[cfg(feature = "hardware")]
#[test]
fn test_swish_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_swish_backward.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut dy_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    dy_buf.to_device(&dy)?;
    let kernel = SwishBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<SwishBackward>(&ptx_path)?;
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
            "swish bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}

// PRelu forward: (x_ptr, slope_ptr, y_ptr, n)
#[cfg(feature = "hardware")]
#[test]
fn test_prelu_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let slope = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/slope.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_prelu.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut slope_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    slope_buf.to_device(&slope)?;
    let kernel = PreluForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<PreluForward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            slope_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            n as i32,
        ),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "prelu fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

// PRelu backward: (dy_ptr, x_ptr, slope_ptr, dx_ptr, dslope_ptr, n)
#[cfg(feature = "hardware")]
#[test]
fn test_prelu_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let slope = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/slope.bin");
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/dy.bin");
    let expected_dx = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_prelu_dx.bin",
    );
    let expected_dslope = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_prelu_dslope.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut slope_buf = device.buffer::<f32>(n)?;
    let mut dy_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let dslope_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    let mut dslope_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    slope_buf.to_device(&slope)?;
    dy_buf.to_device(&dy)?;
    let kernel = PreluBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<PreluBackward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            slope_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            dslope_buf.as_device_ptr(),
            n as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    dslope_buf.to_host(&mut dslope_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected_dx[i]).abs() < TOL,
            "prelu bwd dx mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected_dx[i]
        );
        assert!(
            (dslope_out[i] - expected_dslope[i]).abs() < TOL,
            "prelu bwd dslope mismatch at i={i}: gpu={} expected={}",
            dslope_out[i],
            expected_dslope[i]
        );
    }
    Ok(())
}

// LogSoftmax forward: (x_ptr, y_ptr, n_rows, n_cols)
#[cfg(feature = "hardware")]
const LOG_SOFTMAX_ROWS: usize = 32;
#[cfg(feature = "hardware")]
const LOG_SOFTMAX_COLS_VAL: usize = 128;

#[cfg(feature = "hardware")]
#[test]
fn test_log_softmax_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x_2d.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_log_softmax.bin",
    );
    let n_total = x.len();
    let n_rows = LOG_SOFTMAX_ROWS;
    let n_cols = LOG_SOFTMAX_COLS_VAL;
    let mut x_buf = device.buffer::<f32>(n_total)?;
    let y_buf = device.buffer::<f32>(n_total)?;
    let mut y_out = vec![0.0f32; n_total];
    x_buf.to_device(&x)?;
    let kernel = LogSoftmaxForward::new(LOG_SOFTMAX_COLS);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<LogSoftmaxForward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config_custom(
        [n_rows as u32, 1, 1],
        [n_cols as u32, 1, 1],
        [1, 1, 1],
    );
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            n_rows as i32,
            n_cols as i32,
        ),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n_total {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "log_softmax fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

// LogSoftmax backward: (dy_ptr, y_ptr, dx_ptr, n_rows, n_cols)
#[cfg(feature = "hardware")]
#[test]
fn test_log_softmax_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let y = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_log_softmax.bin",
    );
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/dy_2d.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_log_softmax_backward.bin",
    );
    let n_total = y.len();
    let n_rows = LOG_SOFTMAX_ROWS;
    let n_cols = LOG_SOFTMAX_COLS_VAL;
    let mut y_buf = device.buffer::<f32>(n_total)?;
    let mut dy_buf = device.buffer::<f32>(n_total)?;
    let dx_buf = device.buffer::<f32>(n_total)?;
    let mut dx_out = vec![0.0f32; n_total];
    y_buf.to_device(&y)?;
    dy_buf.to_device(&dy)?;
    let kernel = LogSoftmaxBackward::new(LOG_SOFTMAX_COLS);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<LogSoftmaxBackward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config_custom(
        [n_rows as u32, 1, 1],
        [n_cols as u32, 1, 1],
        [1, 1, 1],
    );
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            n_rows as i32,
            n_cols as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    for i in 0..n_total {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "log_softmax bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}

// ThresholdedRelu forward: (x_ptr, y_ptr, n, alpha)
#[cfg(feature = "hardware")]
const THRELU_ALPHA: f32 = 1.0;

#[cfg(feature = "hardware")]
#[test]
fn test_thresholded_relu_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_thresholded_relu.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    let kernel = ThresholdedReluForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ThresholdedReluForward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            n as i32,
            THRELU_ALPHA,
        ),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "thresholded_relu fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

// ThresholdedRelu backward: (dy_ptr, x_ptr, dx_ptr, n, alpha)
#[cfg(feature = "hardware")]
#[test]
fn test_thresholded_relu_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_thresholded_relu_backward.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut dy_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    dy_buf.to_device(&dy)?;
    let kernel = ThresholdedReluBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ThresholdedReluBackward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            n as i32,
            THRELU_ALPHA,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "thresholded_relu bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}

// Shrink forward: (x_ptr, y_ptr, n, lambd, bias)
#[cfg(feature = "hardware")]
const SHRINK_LAMBD: f32 = 0.5;
#[cfg(feature = "hardware")]
const SHRINK_BIAS: f32 = 0.0;

#[cfg(feature = "hardware")]
#[test]
fn test_shrink_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_shrink.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let y_buf = device.buffer::<f32>(n)?;
    let mut y_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    let kernel = ShrinkForward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ShrinkForward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            n as i32,
            SHRINK_LAMBD,
            SHRINK_BIAS,
        ),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "shrink fwd mismatch at i={i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

// Shrink backward: (dy_ptr, x_ptr, dx_ptr, n, lambd)
#[cfg(feature = "hardware")]
#[test]
fn test_shrink_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/x.bin");
    let dy = load_fixture(env!("CARGO_MANIFEST_DIR"), "extra_activations/dy.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "extra_activations/expected_shrink_backward.bin",
    );
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut dy_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    dy_buf.to_device(&dy)?;
    let kernel = ShrinkBackward::new(BLOCK_SIZE);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ShrinkBackward>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr(),
            x_buf.as_device_ptr(),
            dx_buf.as_device_ptr(),
            n as i32,
            SHRINK_LAMBD,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "shrink bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}
