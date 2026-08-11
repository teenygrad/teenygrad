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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::{compile_kernel, target::Target};

#[cfg(feature = "cuda")]
use teeny_core::device::Device;
#[cfg(feature = "cuda")]
use teeny_core::device::buffer::Buffer;
#[cfg(feature = "cuda")]
use teeny_cuda::{errors::Result, testing};

use teeny_kernels::testing::load_fixture;
use teeny_kernels::nn::tensor::elemwise_binary::{
    ElemwiseClipBackward, ElemwiseClipForward, ElemwiseDivBackward, ElemwiseDivForward,
    ElemwiseEqualForward, ElemwiseFmodForward, ElemwiseGreaterEqualForward, ElemwiseGreaterForward,
    ElemwiseLessEqualForward, ElemwiseLessForward, ElemwiseMaxBackward, ElemwiseMaxForward,
    ElemwiseMeanBackward, ElemwiseMeanForward, ElemwiseMinBackward, ElemwiseMinForward,
    ElemwiseMulBackward, ElemwiseMulForward, ElemwisePowBackward, ElemwisePowForward,
    ElemwiseSubBackward, ElemwiseSubForward, ElemwiseSumBackward, ElemwiseSumForward,
    ElemwiseWhereBackward, ElemwiseWhereForward,
};

const BLOCK_SIZE: i32 = 1024;
const TOL: f32 = 1e-4;

// ── Macro: source + MLIR snapshot ────────────────────────────────────────────

macro_rules! source_test {
    ($test_name:ident, $kernel_ty:ty, $snap_prefix:literal) => {
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let kernel = <$kernel_ty>::new(BLOCK_SIZE);
            let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
            let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
            let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
            assert_debug_snapshot!(concat!($snap_prefix, "_source"), kernel.source());
            assert_debug_snapshot!(concat!($snap_prefix, "_mlir"), mlir.trim());
            Ok(())
        }
    };
}

// ── Macro: GPU forward test (a_ptr, b_ptr, out_ptr, n) ────────────────────────

macro_rules! gpu_forward_test_2 {
    ($test_name:ident, $kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "cuda")]
        #[test]
        fn $test_name() -> Result<()> {
            dotenv().ok();
            let env = testing::setup_cuda_env()?;
            let device = env.device;
            let a = load_fixture("elemwise_binary/a.bin");
            let b = load_fixture("elemwise_binary/b.bin");
            let expected = load_fixture(concat!("elemwise_binary/expected_", $fixture_op, ".bin"));
            let n = a.len();
            let mut a_buf = device.buffer::<f32>(n)?;
            let mut b_buf = device.buffer::<f32>(n)?;
            let out_buf = device.buffer::<f32>(n)?;
            let mut out = vec![0.0f32; n];
            a_buf.to_device(&a)?;
            b_buf.to_device(&b)?;
            let kernel = <$kernel_ty>::new(BLOCK_SIZE);
            let target = Target::new(env.capability);
            let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
            let program = testing::load_program_from_ptx::<$kernel_ty>(&ptx)?;
            let cfg = testing::launch_config_from_program(n, &program);
            device.launch(
                &program,
                &cfg,
                (
                    a_buf.as_device_ptr() as *mut f32,
                    b_buf.as_device_ptr() as *mut f32,
                    out_buf.as_device_ptr() as *mut f32,
                    n as i32,
                ),
            )?;
            out_buf.to_host(&mut out)?;
            for i in 0..n {
                assert!(
                    (out[i] - expected[i]).abs() < TOL,
                    "{} fwd mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    out[i],
                    expected[i]
                );
            }
            Ok(())
        }
    };
}

// ── Macro: GPU backward (dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr, n) ─────────────

macro_rules! gpu_backward_test_2 {
    ($test_name:ident, $bwd_kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "cuda")]
        #[test]
        fn $test_name() -> Result<()> {
            dotenv().ok();
            let env = testing::setup_cuda_env()?;
            let device = env.device;
            let a = load_fixture("elemwise_binary/a.bin");
            let b = load_fixture("elemwise_binary/b.bin");
            let dy = load_fixture("elemwise_binary/dy.bin");
            let expected_da =
                load_fixture(concat!("elemwise_binary/expected_", $fixture_op, "_da.bin"));
            let expected_db =
                load_fixture(concat!("elemwise_binary/expected_", $fixture_op, "_db.bin"));
            let n = a.len();
            let mut a_buf = device.buffer::<f32>(n)?;
            let mut b_buf = device.buffer::<f32>(n)?;
            let mut dy_buf = device.buffer::<f32>(n)?;
            let da_buf = device.buffer::<f32>(n)?;
            let db_buf = device.buffer::<f32>(n)?;
            let mut da_out = vec![0.0f32; n];
            let mut db_out = vec![0.0f32; n];
            a_buf.to_device(&a)?;
            b_buf.to_device(&b)?;
            dy_buf.to_device(&dy)?;
            let kernel = <$bwd_kernel_ty>::new(BLOCK_SIZE);
            let target = Target::new(env.capability);
            let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
            let program = testing::load_program_from_ptx::<$bwd_kernel_ty>(&ptx)?;
            let cfg = testing::launch_config_from_program(n, &program);
            device.launch(
                &program,
                &cfg,
                (
                    dy_buf.as_device_ptr() as *mut f32,
                    a_buf.as_device_ptr() as *mut f32,
                    b_buf.as_device_ptr() as *mut f32,
                    da_buf.as_device_ptr() as *mut f32,
                    db_buf.as_device_ptr() as *mut f32,
                    n as i32,
                ),
            )?;
            da_buf.to_host(&mut da_out)?;
            db_buf.to_host(&mut db_out)?;
            for i in 0..n {
                assert!(
                    (da_out[i] - expected_da[i]).abs() < TOL,
                    "{} bwd da mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    da_out[i],
                    expected_da[i]
                );
                assert!(
                    (db_out[i] - expected_db[i]).abs() < TOL,
                    "{} bwd db mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    db_out[i],
                    expected_db[i]
                );
            }
            Ok(())
        }
    };
}

// ── Macro: GPU backward (dy_ptr, da_ptr, db_ptr, n) — no a,b saved ───────────
// Used by Sub, Sum, Mean backward kernels

macro_rules! gpu_backward_test_dyonly {
    ($test_name:ident, $bwd_kernel_ty:ty, $fixture_op:literal, $op_name:literal) => {
        #[cfg(feature = "cuda")]
        #[test]
        fn $test_name() -> Result<()> {
            dotenv().ok();
            let env = testing::setup_cuda_env()?;
            let device = env.device;
            let dy = load_fixture("elemwise_binary/dy.bin");
            let expected_da =
                load_fixture(concat!("elemwise_binary/expected_", $fixture_op, "_da.bin"));
            let expected_db =
                load_fixture(concat!("elemwise_binary/expected_", $fixture_op, "_db.bin"));
            let n = dy.len();
            let mut dy_buf = device.buffer::<f32>(n)?;
            let da_buf = device.buffer::<f32>(n)?;
            let db_buf = device.buffer::<f32>(n)?;
            let mut da_out = vec![0.0f32; n];
            let mut db_out = vec![0.0f32; n];
            dy_buf.to_device(&dy)?;
            let kernel = <$bwd_kernel_ty>::new(BLOCK_SIZE);
            let target = Target::new(env.capability);
            let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
            let program = testing::load_program_from_ptx::<$bwd_kernel_ty>(&ptx)?;
            let cfg = testing::launch_config_from_program(n, &program);
            device.launch(
                &program,
                &cfg,
                (
                    dy_buf.as_device_ptr() as *mut f32,
                    da_buf.as_device_ptr() as *mut f32,
                    db_buf.as_device_ptr() as *mut f32,
                    n as i32,
                ),
            )?;
            da_buf.to_host(&mut da_out)?;
            db_buf.to_host(&mut db_out)?;
            for i in 0..n {
                assert!(
                    (da_out[i] - expected_da[i]).abs() < TOL,
                    "{} bwd da mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    da_out[i],
                    expected_da[i]
                );
                assert!(
                    (db_out[i] - expected_db[i]).abs() < TOL,
                    "{} bwd db mismatch at i={i}: gpu={} expected={}",
                    $op_name,
                    db_out[i],
                    expected_db[i]
                );
            }
            Ok(())
        }
    };
}

// ── Source + MLIR snapshots ───────────────────────────────────────────────────

source_test!(
    test_mul_source,
    ElemwiseMulForward::<f32>,
    "elemwise_mul_forward"
);
source_test!(
    test_sub_source,
    ElemwiseSubForward::<f32>,
    "elemwise_sub_forward"
);
source_test!(
    test_div_source,
    ElemwiseDivForward::<f32>,
    "elemwise_div_forward"
);
source_test!(
    test_pow_source,
    ElemwisePowForward::<f32>,
    "elemwise_pow_forward"
);
source_test!(
    test_fmod_source,
    ElemwiseFmodForward::<f32>,
    "elemwise_fmod_forward"
);
source_test!(
    test_min_source,
    ElemwiseMinForward::<f32>,
    "elemwise_min_forward"
);
source_test!(
    test_max_source,
    ElemwiseMaxForward::<f32>,
    "elemwise_max_forward"
);
source_test!(
    test_mean_source,
    ElemwiseMeanForward::<f32>,
    "elemwise_mean_forward"
);
source_test!(
    test_sum_source,
    ElemwiseSumForward::<f32>,
    "elemwise_sum_forward"
);
source_test!(
    test_equal_source,
    ElemwiseEqualForward::<f32>,
    "elemwise_equal_forward"
);
source_test!(
    test_greater_source,
    ElemwiseGreaterForward::<f32>,
    "elemwise_greater_forward"
);
source_test!(
    test_greater_equal_source,
    ElemwiseGreaterEqualForward::<f32>,
    "elemwise_greater_equal_forward"
);
source_test!(
    test_less_source,
    ElemwiseLessForward::<f32>,
    "elemwise_less_forward"
);
source_test!(
    test_less_equal_source,
    ElemwiseLessEqualForward::<f32>,
    "elemwise_less_equal_forward"
);
source_test!(
    test_where_source,
    ElemwiseWhereForward::<f32>,
    "elemwise_where_forward"
);
source_test!(
    test_clip_source,
    ElemwiseClipForward::<f32>,
    "elemwise_clip_forward"
);

// Backward source snapshots
source_test!(
    test_mul_backward_source,
    ElemwiseMulBackward::<f32>,
    "elemwise_mul_backward"
);
source_test!(
    test_sub_backward_source,
    ElemwiseSubBackward::<f32>,
    "elemwise_sub_backward"
);
source_test!(
    test_div_backward_source,
    ElemwiseDivBackward::<f32>,
    "elemwise_div_backward"
);
source_test!(
    test_pow_backward_source,
    ElemwisePowBackward::<f32>,
    "elemwise_pow_backward"
);
source_test!(
    test_min_backward_source,
    ElemwiseMinBackward::<f32>,
    "elemwise_min_backward"
);
source_test!(
    test_max_backward_source,
    ElemwiseMaxBackward::<f32>,
    "elemwise_max_backward"
);
source_test!(
    test_mean_backward_source,
    ElemwiseMeanBackward::<f32>,
    "elemwise_mean_backward"
);
source_test!(
    test_sum_backward_source,
    ElemwiseSumBackward::<f32>,
    "elemwise_sum_backward"
);
source_test!(
    test_where_backward_source,
    ElemwiseWhereBackward::<f32>,
    "elemwise_where_backward"
);
source_test!(
    test_clip_backward_source,
    ElemwiseClipBackward::<f32>,
    "elemwise_clip_backward"
);

// ── GPU forward tests ─────────────────────────────────────────────────────────

gpu_forward_test_2!(
    test_mul_forward_gpu,
    ElemwiseMulForward::<f32>,
    "mul",
    "mul"
);
gpu_forward_test_2!(
    test_sub_forward_gpu,
    ElemwiseSubForward::<f32>,
    "sub",
    "sub"
);
gpu_forward_test_2!(
    test_div_forward_gpu,
    ElemwiseDivForward::<f32>,
    "div",
    "div"
);
gpu_forward_test_2!(
    test_pow_forward_gpu,
    ElemwisePowForward::<f32>,
    "pow",
    "pow"
);
gpu_forward_test_2!(
    test_fmod_forward_gpu,
    ElemwiseFmodForward::<f32>,
    "fmod",
    "fmod"
);
gpu_forward_test_2!(
    test_min_forward_gpu,
    ElemwiseMinForward::<f32>,
    "min",
    "min"
);
gpu_forward_test_2!(
    test_max_forward_gpu,
    ElemwiseMaxForward::<f32>,
    "max",
    "max"
);
gpu_forward_test_2!(
    test_mean_forward_gpu,
    ElemwiseMeanForward::<f32>,
    "mean",
    "mean"
);
gpu_forward_test_2!(
    test_sum_forward_gpu,
    ElemwiseSumForward::<f32>,
    "sum",
    "sum"
);
gpu_forward_test_2!(
    test_equal_forward_gpu,
    ElemwiseEqualForward::<f32>,
    "equal",
    "equal"
);
gpu_forward_test_2!(
    test_greater_forward_gpu,
    ElemwiseGreaterForward::<f32>,
    "greater",
    "greater"
);
gpu_forward_test_2!(
    test_greater_equal_forward_gpu,
    ElemwiseGreaterEqualForward::<f32>,
    "greater_equal",
    "greater_equal"
);
gpu_forward_test_2!(
    test_less_forward_gpu,
    ElemwiseLessForward::<f32>,
    "less",
    "less"
);
gpu_forward_test_2!(
    test_less_equal_forward_gpu,
    ElemwiseLessEqualForward::<f32>,
    "less_equal",
    "less_equal"
);

// Where forward: (cond_ptr, x_ptr, y_ptr, out_ptr, n)
#[cfg(feature = "cuda")]
#[test]
fn test_where_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let cond = load_fixture("elemwise_binary/cond.bin");
    let x = load_fixture("elemwise_binary/a.bin");
    let y = load_fixture("elemwise_binary/b.bin");
    let expected = load_fixture("elemwise_binary/expected_where.bin");
    let n = cond.len();
    let mut cond_buf = device.buffer::<f32>(n)?;
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut y_buf = device.buffer::<f32>(n)?;
    let out_buf = device.buffer::<f32>(n)?;
    let mut out = vec![0.0f32; n];
    cond_buf.to_device(&cond)?;
    x_buf.to_device(&x)?;
    y_buf.to_device(&y)?;
    let kernel = ElemwiseWhereForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<ElemwiseWhereForward<f32>>(&ptx)?;
    let cfg = testing::launch_config_from_program(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            cond_buf.as_device_ptr() as *mut f32,
            x_buf.as_device_ptr() as *mut f32,
            y_buf.as_device_ptr() as *mut f32,
            out_buf.as_device_ptr() as *mut f32,
            n as i32,
        ),
    )?;
    out_buf.to_host(&mut out)?;
    for i in 0..n {
        assert!(
            (out[i] - expected[i]).abs() < TOL,
            "where fwd mismatch at i={i}: gpu={} expected={}",
            out[i],
            expected[i]
        );
    }
    Ok(())
}

// Clip forward: (x_ptr, out_ptr, n, min_val, max_val)
#[cfg(feature = "cuda")]
#[test]
fn test_clip_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let x = load_fixture("elemwise_binary/a.bin");
    let expected = load_fixture("elemwise_binary/expected_clip.bin");
    let n = x.len();
    let mut x_buf = device.buffer::<f32>(n)?;
    let out_buf = device.buffer::<f32>(n)?;
    let mut out = vec![0.0f32; n];
    x_buf.to_device(&x)?;
    let kernel = ElemwiseClipForward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<ElemwiseClipForward<f32>>(&ptx)?;
    let cfg = testing::launch_config_from_program(n, &program);
    // min_val=-1.0, max_val=1.0 (must match fixture)
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr() as *mut f32,
            out_buf.as_device_ptr() as *mut f32,
            n as i32,
            -1.0f32,
            1.0f32,
        ),
    )?;
    out_buf.to_host(&mut out)?;
    for i in 0..n {
        assert!(
            (out[i] - expected[i]).abs() < TOL,
            "clip fwd mismatch at i={i}: gpu={} expected={}",
            out[i],
            expected[i]
        );
    }
    Ok(())
}

// ── GPU backward tests ────────────────────────────────────────────────────────

gpu_backward_test_2!(
    test_mul_backward_gpu,
    ElemwiseMulBackward::<f32>,
    "mul",
    "mul"
);
gpu_backward_test_2!(
    test_div_backward_gpu,
    ElemwiseDivBackward::<f32>,
    "div",
    "div"
);
gpu_backward_test_2!(
    test_pow_backward_gpu,
    ElemwisePowBackward::<f32>,
    "pow",
    "pow"
);
gpu_backward_test_2!(
    test_min_backward_gpu,
    ElemwiseMinBackward::<f32>,
    "min",
    "min"
);
gpu_backward_test_2!(
    test_max_backward_gpu,
    ElemwiseMaxBackward::<f32>,
    "max",
    "max"
);

// Sub/Sum/Mean backward: (dy_ptr, da_ptr, db_ptr, n) — no a,b saved
gpu_backward_test_dyonly!(
    test_sub_backward_gpu,
    ElemwiseSubBackward::<f32>,
    "sub",
    "sub"
);
gpu_backward_test_dyonly!(
    test_sum_backward_gpu,
    ElemwiseSumBackward::<f32>,
    "sum",
    "sum"
);
gpu_backward_test_dyonly!(
    test_mean_backward_gpu,
    ElemwiseMeanBackward::<f32>,
    "mean",
    "mean"
);

// Where backward: (dy_ptr, cond_ptr, dx_ptr, dy_in_ptr, n)
#[cfg(feature = "cuda")]
#[test]
fn test_where_backward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let dy = load_fixture("elemwise_binary/dy.bin");
    let cond = load_fixture("elemwise_binary/cond.bin");
    let expected_dx = load_fixture("elemwise_binary/expected_where_dx.bin");
    let expected_dy_in = load_fixture("elemwise_binary/expected_where_dy_in.bin");
    let n = dy.len();
    let mut dy_buf = device.buffer::<f32>(n)?;
    let mut cond_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let dy_in_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    let mut dy_in_out = vec![0.0f32; n];
    dy_buf.to_device(&dy)?;
    cond_buf.to_device(&cond)?;
    let kernel = ElemwiseWhereBackward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<ElemwiseWhereBackward<f32>>(&ptx)?;
    let cfg = testing::launch_config_from_program(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr() as *mut f32,
            cond_buf.as_device_ptr() as *mut f32,
            dx_buf.as_device_ptr() as *mut f32,
            dy_in_buf.as_device_ptr() as *mut f32,
            n as i32,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    dy_in_buf.to_host(&mut dy_in_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected_dx[i]).abs() < TOL,
            "where bwd dx mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected_dx[i]
        );
        assert!(
            (dy_in_out[i] - expected_dy_in[i]).abs() < TOL,
            "where bwd dy_in mismatch at i={i}: gpu={} expected={}",
            dy_in_out[i],
            expected_dy_in[i]
        );
    }
    Ok(())
}

// Clip backward: (dy_ptr, x_ptr, dx_ptr, n, min_val, max_val)
#[cfg(feature = "cuda")]
#[test]
fn test_clip_backward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let dy = load_fixture("elemwise_binary/dy.bin");
    let x = load_fixture("elemwise_binary/a.bin");
    let expected = load_fixture("elemwise_binary/expected_clip_backward.bin");
    let n = dy.len();
    let mut dy_buf = device.buffer::<f32>(n)?;
    let mut x_buf = device.buffer::<f32>(n)?;
    let dx_buf = device.buffer::<f32>(n)?;
    let mut dx_out = vec![0.0f32; n];
    dy_buf.to_device(&dy)?;
    x_buf.to_device(&x)?;
    let kernel = ElemwiseClipBackward::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
    let program = testing::load_program_from_ptx::<ElemwiseClipBackward<f32>>(&ptx)?;
    let cfg = testing::launch_config_from_program(n, &program);
    device.launch(
        &program,
        &cfg,
        (
            dy_buf.as_device_ptr() as *mut f32,
            x_buf.as_device_ptr() as *mut f32,
            dx_buf.as_device_ptr() as *mut f32,
            n as i32,
            -1.0f32,
            1.0f32,
        ),
    )?;
    dx_buf.to_host(&mut dx_out)?;
    for i in 0..n {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "clip bwd mismatch at i={i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}
