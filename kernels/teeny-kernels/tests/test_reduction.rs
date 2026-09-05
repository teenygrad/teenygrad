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

use teeny_kernels::nn::tensor::reduction::{
    CumProdForward, CumSumForward, GlobalAvgPoolForward, GlobalMaxPoolForward, ReduceL1Forward,
    ReduceL2Forward, ReduceLogSumExpForward, ReduceLogSumForward, ReduceMaxForward,
    ReduceMeanForward, ReduceMinForward, ReduceProdForward, ReduceSumForward,
    ReduceSumSquareForward,
};
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

// Reduction tests use a 2-D input: OUTER rows of INNER elements.
#[cfg(feature = "hardware")]
const INNER: usize = 64;
const BLOCK_INNER: i32 = 64;
#[cfg(feature = "hardware")]
const TOL: f32 = 1e-4;

// ── Macro: source + MLIR snapshot ────────────────────────────────────────────

macro_rules! source_test {
    ($test_name:ident, $kernel_ty:ty, $snap_prefix:literal) => {
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let kernel = <$kernel_ty>::new(BLOCK_INNER);
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

// ── Macro: GPU forward for row-reduction kernels ───────────────────────────────
// Signature: (x_ptr, y_ptr, n_inner, n_outer)

macro_rules! gpu_reduce_test {
    ($test_name:ident, $kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "hardware")]
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let device = teeny_runtime::open()?;
            let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "reduction/x.bin");
            let expected = load_fixture(
                env!("CARGO_MANIFEST_DIR"),
                concat!("reduction/expected_", $fixture_op, ".bin"),
            );
            let n_total = x.len();
            let n_outer = expected.len();
            let n_inner = n_total / n_outer;
            let mut x_buf = device.buffer::<f32>(n_total)?;
            let y_buf = device.buffer::<f32>(n_outer)?;
            let mut y_out = vec![0.0f32; n_outer];
            x_buf.to_device(&x)?;
            let kernel = <$kernel_ty>::new(BLOCK_INNER);
            let target = teeny_runtime::default_target(&device)?;
            let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
            let program = teeny_runtime::load_program::<$kernel_ty>(&ptx_path)?;
            // threads-per-block comes from PTX metadata via launch_config_with_grid --
            // Triton may choose a different thread count (e.g. 128) than BLOCK_INNER (64).
            let cfg = teeny_runtime::launch_config_with_grid(n_outer, &program);
            device.launch(
                &program,
                &cfg,
                (
                    x_buf.as_device_ptr(),
                    y_buf.as_device_ptr(),
                    n_inner as i32,
                    n_outer as i32,
                ),
            )?;
            y_buf.to_host(&mut y_out)?;
            for i in 0..n_outer {
                assert!(
                    (y_out[i] - expected[i]).abs() < TOL,
                    "{} mismatch at row={i}: gpu={} expected={}",
                    $op_name,
                    y_out[i],
                    expected[i]
                );
            }
            Ok(())
        }
    };
}

// ── Macro: GPU forward for cumulative ops (output same shape as input) ────────

macro_rules! gpu_cum_test {
    ($test_name:ident, $kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "hardware")]
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let device = teeny_runtime::open()?;
            let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "reduction/x.bin");
            let expected = load_fixture(
                env!("CARGO_MANIFEST_DIR"),
                concat!("reduction/expected_", $fixture_op, ".bin"),
            );
            let n_total = x.len();
            let n_inner = INNER;
            let n_outer = n_total / n_inner;
            let mut x_buf = device.buffer::<f32>(n_total)?;
            let y_buf = device.buffer::<f32>(n_total)?;
            let mut y_out = vec![0.0f32; n_total];
            x_buf.to_device(&x)?;
            let kernel = <$kernel_ty>::new(BLOCK_INNER);
            let target = teeny_runtime::default_target(&device)?;
            let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
            let program = teeny_runtime::load_program::<$kernel_ty>(&ptx_path)?;
            let cfg = teeny_runtime::launch_config_with_grid(n_outer, &program);
            device.launch(
                &program,
                &cfg,
                (
                    x_buf.as_device_ptr(),
                    y_buf.as_device_ptr(),
                    n_inner as i32,
                    n_outer as i32,
                ),
            )?;
            y_buf.to_host(&mut y_out)?;
            for i in 0..n_total {
                assert!(
                    (y_out[i] - expected[i]).abs() < TOL,
                    "{} mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    y_out[i],
                    expected[i]
                );
            }
            Ok(())
        }
    };
}

// ── Source + MLIR snapshots ───────────────────────────────────────────────────

source_test!(
    test_reduce_sum_source,
    ReduceSumForward::<f32>,
    "reduce_sum_forward"
);
source_test!(
    test_reduce_mean_source,
    ReduceMeanForward::<f32>,
    "reduce_mean_forward"
);
source_test!(
    test_reduce_max_source,
    ReduceMaxForward::<f32>,
    "reduce_max_forward"
);
source_test!(
    test_reduce_min_source,
    ReduceMinForward::<f32>,
    "reduce_min_forward"
);
source_test!(
    test_reduce_prod_source,
    ReduceProdForward::<f32>,
    "reduce_prod_forward"
);
source_test!(
    test_reduce_l1_source,
    ReduceL1Forward::<f32>,
    "reduce_l1_forward"
);
source_test!(
    test_reduce_l2_source,
    ReduceL2Forward::<f32>,
    "reduce_l2_forward"
);
source_test!(
    test_reduce_log_sum_source,
    ReduceLogSumForward::<f32>,
    "reduce_log_sum_forward"
);
source_test!(
    test_reduce_log_sum_exp_source,
    ReduceLogSumExpForward::<f32>,
    "reduce_log_sum_exp_forward"
);
source_test!(
    test_reduce_sum_square_source,
    ReduceSumSquareForward::<f32>,
    "reduce_sum_square_forward"
);
source_test!(test_cum_sum_source, CumSumForward::<f32>, "cum_sum_forward");
source_test!(
    test_cum_prod_source,
    CumProdForward::<f32>,
    "cum_prod_forward"
);
source_test!(
    test_global_avg_pool_source,
    GlobalAvgPoolForward::<f32>,
    "global_avg_pool_forward"
);
source_test!(
    test_global_max_pool_source,
    GlobalMaxPoolForward::<f32>,
    "global_max_pool_forward"
);

// ── GPU forward tests ─────────────────────────────────────────────────────────

gpu_reduce_test!(
    test_reduce_sum_gpu,
    ReduceSumForward::<f32>,
    "reduce_sum",
    "reduce_sum"
);
gpu_reduce_test!(
    test_reduce_mean_gpu,
    ReduceMeanForward::<f32>,
    "reduce_mean",
    "reduce_mean"
);
gpu_reduce_test!(
    test_reduce_max_gpu,
    ReduceMaxForward::<f32>,
    "reduce_max",
    "reduce_max"
);
gpu_reduce_test!(
    test_reduce_min_gpu,
    ReduceMinForward::<f32>,
    "reduce_min",
    "reduce_min"
);
// reduce_prod uses exp(sum(log)) which accumulates fp error; use relative tolerance
#[cfg(feature = "hardware")]
#[test]
fn test_reduce_prod_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "reduction/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "reduction/expected_reduce_prod.bin",
    );
    let n_total = x.len();
    let n_outer = expected.len();
    let n_inner = n_total / n_outer;
    let mut x_buf = device.buffer::<f32>(n_total)?;
    let y_buf = device.buffer::<f32>(n_outer)?;
    let mut y_out = vec![0.0f32; n_outer];
    x_buf.to_device(&x)?;
    let kernel = ReduceProdForward::<f32>::new(BLOCK_INNER);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<ReduceProdForward<f32>>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config_with_grid(n_outer, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            n_inner as i32,
            n_outer as i32,
        ),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n_outer {
        let rel_err = (y_out[i] - expected[i]).abs() / expected[i].abs().max(1e-6);
        assert!(
            rel_err < 1e-4,
            "reduce_prod mismatch at row={i}: gpu={} expected={} rel_err={}",
            y_out[i],
            expected[i],
            rel_err
        );
    }
    Ok(())
}
gpu_reduce_test!(
    test_reduce_l1_gpu,
    ReduceL1Forward::<f32>,
    "reduce_l1",
    "reduce_l1"
);
gpu_reduce_test!(
    test_reduce_l2_gpu,
    ReduceL2Forward::<f32>,
    "reduce_l2",
    "reduce_l2"
);
gpu_reduce_test!(
    test_reduce_log_sum_gpu,
    ReduceLogSumForward::<f32>,
    "reduce_log_sum",
    "reduce_log_sum"
);
gpu_reduce_test!(
    test_reduce_log_sum_exp_gpu,
    ReduceLogSumExpForward::<f32>,
    "reduce_log_sum_exp",
    "reduce_log_sum_exp"
);
gpu_reduce_test!(
    test_reduce_sum_square_gpu,
    ReduceSumSquareForward::<f32>,
    "reduce_sum_square",
    "reduce_sum_square"
);
gpu_reduce_test!(
    test_global_avg_pool_gpu,
    GlobalAvgPoolForward::<f32>,
    "global_avg_pool",
    "global_avg_pool"
);
gpu_reduce_test!(
    test_global_max_pool_gpu,
    GlobalMaxPoolForward::<f32>,
    "global_max_pool",
    "global_max_pool"
);

gpu_cum_test!(test_cum_sum_gpu, CumSumForward::<f32>, "cum_sum", "cum_sum");
// cum_prod accumulates floating-point error for large products; use relative tolerance
#[cfg(feature = "hardware")]
#[test]
fn test_cum_prod_gpu() -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;
    let x = load_fixture(env!("CARGO_MANIFEST_DIR"), "reduction/x.bin");
    let expected = load_fixture(
        env!("CARGO_MANIFEST_DIR"),
        "reduction/expected_cum_prod.bin",
    );
    let n_total = x.len();
    let n_inner = INNER;
    let n_outer = n_total / n_inner;
    let mut x_buf = device.buffer::<f32>(n_total)?;
    let y_buf = device.buffer::<f32>(n_total)?;
    let mut y_out = vec![0.0f32; n_total];
    x_buf.to_device(&x)?;
    let kernel = CumProdForward::<f32>::new(BLOCK_INNER);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<CumProdForward<f32>>(&ptx_path)?;
    let cfg = teeny_runtime::launch_config_with_grid(n_outer, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            y_buf.as_device_ptr(),
            n_inner as i32,
            n_outer as i32,
        ),
    )?;
    y_buf.to_host(&mut y_out)?;
    for i in 0..n_total {
        let rel_err = (y_out[i] - expected[i]).abs() / expected[i].abs().max(1e-6);
        assert!(
            rel_err < 1e-3,
            "cum_prod mismatch at i={i}: gpu={} expected={} rel_err={}",
            y_out[i],
            expected[i],
            rel_err
        );
    }
    Ok(())
}
