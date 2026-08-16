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
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_core::device::Device;
#[cfg(feature = "cuda")]
use teeny_core::device::buffer::Buffer;
#[cfg(feature = "cuda")]
use teeny_cuda::{errors::Result, testing};

use teeny_kernels::math::gemm::{MatmulBackwardDa, MatmulBackwardDb, MatmulForward};

const BLOCK_M: i32 = 32;
const BLOCK_N: i32 = 32;
const BLOCK_K: i32 = 32;
const GROUP_M: i32 = 8;
// TF32 tensor-core precision, not full f32: error scales with the magnitude
// of the accumulated sum (K=64 here), so a flat absolute tolerance like
// test_conv2d_bn_silu_gemm.rs's 1e-2 isn't quite enough once values grow
// past ~10. atol + rtol * |expected| scales with it instead.
const ATOL: f32 = 1e-2;
const RTOL: f32 = 1e-3;

#[cfg(feature = "cuda")]
fn tf32_close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() < ATOL + RTOL * expected.abs()
}

/// Must match `.reqntid` in the generated PTX (see linear_forward/backward).
#[cfg(feature = "cuda")]
const PTX_LAUNCH_THREADS_X: u32 = 128;

// ── Source + MLIR snapshots ───────────────────────────────────────────────────

#[test]
fn test_matmul_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = MatmulForward::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("matmul_forward_source", kernel.source());
    assert_debug_snapshot!("matmul_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_matmul_backward_da_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = MatmulBackwardDa::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("matmul_backward_da_source", kernel.source());
    assert_debug_snapshot!("matmul_backward_da_mlir", mlir.trim());
    Ok(())
}

#[test]
fn test_matmul_backward_db_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = MatmulBackwardDb::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("matmul_backward_db_source", kernel.source());
    assert_debug_snapshot!("matmul_backward_db_mlir", mlir.trim());
    Ok(())
}

// ── GPU forward test ──────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_forward_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // Deliberately not tile-aligned (M, N, K aren't multiples of BLOCK_M/N/K),
    // so this also exercises the tensor descriptors' zero-padding on the last
    // partial tile in every dimension.
    let m = 64usize;
    let n = 48usize;
    let k = 64usize;

    let a: Vec<f32> = (0..m * k).map(|i| i as f32 / (m * k) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 / (k * n) as f32).collect();

    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[mi * k + ki] * b[ki * n + ni];
            }
            expected[mi * n + ni] = sum;
        }
    }

    let mut a_buf = device.buffer::<f32>(m * k)?;
    let mut b_buf = device.buffer::<f32>(k * n)?;
    let c_buf = device.buffer::<f32>(m * n)?;
    let mut c_out = vec![0.0f32; m * n];

    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;

    let kernel = MatmulForward::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<MatmulForward<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let pm = (m as u32).div_ceil(BLOCK_M as u32);
    let pn = (n as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [pm * pn, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            a_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            c_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    c_buf.to_host(&mut c_out)?;
    for i in 0..m * n {
        assert!(
            tf32_close(c_out[i], expected[i]),
            "matmul fwd mismatch at i={i}: gpu={} expected={}",
            c_out[i],
            expected[i]
        );
    }
    Ok(())
}

// ── Inline data + pipeline-stage logging ─────────────────────────────────────
//
// Same MatmulForward kernel as above (a tensor-core-eligible tl.dot call over
// K-tiles — see MatmulForward's single T::dot call in the accumulation loop),
// but with data generated inline instead of computed alongside the launch,
// and compiled with LlvmCompiler::with_log_level(Debug) so teenyc's
// ttir/ttgpuir/llir/llvmir/ptx pipeline stages are logged to stderr as the
// compile runs. Run with `--nocapture` (and redirect stderr) to capture
// them, e.g.:
//
//   cargo test -p teeny-kernels --test test_gemm --features cuda \
//     test_matmul_forward_logs_pipeline_stages -- --nocapture 2>pipeline.log

/// Naive host-side reference for `MatmulForward`: `C = A @ B`. `a` is (m, k)
/// row-major, `b` is (k, n) row-major, `c` is (m, n) row-major.
#[cfg(feature = "cuda")]
fn matmul_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

#[test]
#[cfg(feature = "cuda")]
fn test_matmul_forward_logs_pipeline_stages() -> Result<()> {
    use teeny_compiler::compiler::backend::llvm::compiler::{LlvmCompiler, LogLevel};
    use teeny_core::compiler::Compiler as _;

    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // Deterministic inline data, independent of the other GPU tests above, so
    // this test is fully self-contained.
    let m = 64usize;
    let n = 48usize;
    let k = 64usize;

    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 % 13.0 - 6.0) * 0.05).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| (i as f32 % 11.0 - 5.0) * 0.05).collect();
    let expected = matmul_reference(&a_host, &b_host, m, n, k);
    let mut c_out = vec![0.0f32; m * n];

    let mut a_buf = device.buffer::<f32>(m * k)?;
    let mut b_buf = device.buffer::<f32>(k * n)?;
    let c_buf = device.buffer::<f32>(m * n)?;

    a_buf.to_device(&a_host)?;
    b_buf.to_device(&b_host)?;

    let kernel = MatmulForward::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(env.capability);

    // Build the compiler by hand (rather than the `compile_kernel` helper the
    // other tests use) so we can turn on pipeline-stage logging.
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir = teeny_compiler::compiler::default_cache_dir();
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?
        .with_target_cpu(target.capability.to_string())
        .with_log_level(LogLevel::Debug);

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .finish();

    // Scoped to this thread only, so it doesn't clobber a subscriber another
    // test running concurrently in this binary may have installed. `force:
    // true` guarantees `teenyc` actually runs (a cache hit would emit nothing).
    let ptx_path =
        tracing::subscriber::with_default(subscriber, || compiler.compile(&kernel, &target, true))?;
    println!("compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;

    let program = testing::load_program_from_ptx::<MatmulForward<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let pm = (m as u32).div_ceil(BLOCK_M as u32);
    let pn = (n as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [pm * pn, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };

    device.launch(
        &program,
        &cfg,
        (
            a_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            c_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;
    c_buf.to_host(&mut c_out)?;

    for i in 0..m * n {
        assert!(
            tf32_close(c_out[i], expected[i]),
            "matmul (inline data) mismatch at index {i}: gpu={}, expected={}",
            c_out[i],
            expected[i]
        );
    }

    Ok(())
}

// ── GPU backward: dA ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_backward_da_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // dA[M,K] = dC[M,N] @ B^T[N,K]  =>  dA[m,k] = sum_n dC[m,n] * B[k,n]
    let m = 64usize;
    let n = 48usize;
    let k = 64usize;

    let b: Vec<f32> = (0..k * n).map(|i| i as f32 / (k * n) as f32).collect();
    let dc: Vec<f32> = (0..m * n).map(|i| (i as f32 % 7.0) * 0.1).collect();

    let mut expected_da = vec![0.0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut sum = 0.0f32;
            for ni in 0..n {
                sum += dc[mi * n + ni] * b[ki * n + ni];
            }
            expected_da[mi * k + ki] = sum;
        }
    }

    let mut dc_buf = device.buffer::<f32>(m * n)?;
    let mut b_buf = device.buffer::<f32>(k * n)?;
    let da_buf = device.buffer::<f32>(m * k)?;
    let mut da_out = vec![0.0f32; m * k];

    dc_buf.to_device(&dc)?;
    b_buf.to_device(&b)?;

    let kernel = MatmulBackwardDa::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<MatmulBackwardDa<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let pm = (m as u32).div_ceil(BLOCK_M as u32);
    let pk = (k as u32).div_ceil(BLOCK_K as u32);
    let cfg = CudaLaunchConfig {
        grid: [pm * pk, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            dc_buf.as_device_ptr() as *mut f32,
            b_buf.as_device_ptr() as *mut f32,
            da_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    da_buf.to_host(&mut da_out)?;
    for i in 0..m * k {
        assert!(
            tf32_close(da_out[i], expected_da[i]),
            "matmul bwd_da mismatch at i={i}: gpu={} expected={}",
            da_out[i],
            expected_da[i]
        );
    }
    Ok(())
}

// ── GPU backward: dB ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_backward_db_gpu() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    // dB[K,N] = A^T[K,M] @ dC[M,N]  =>  dB[k,n] = sum_m A[m,k] * dC[m,n]
    let m = 64usize;
    let n = 48usize;
    let k = 64usize;

    let a: Vec<f32> = (0..m * k).map(|i| i as f32 / (m * k) as f32).collect();
    let dc: Vec<f32> = (0..m * n).map(|i| (i as f32 % 7.0) * 0.1).collect();

    let mut expected_db = vec![0.0f32; k * n];
    for ki in 0..k {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for mi in 0..m {
                sum += a[mi * k + ki] * dc[mi * n + ni];
            }
            expected_db[ki * n + ni] = sum;
        }
    }

    let mut dc_buf = device.buffer::<f32>(m * n)?;
    let mut a_buf = device.buffer::<f32>(m * k)?;
    let db_buf = device.buffer::<f32>(k * n)?;
    let mut db_out = vec![0.0f32; k * n];

    dc_buf.to_device(&dc)?;
    a_buf.to_device(&a)?;

    let kernel = MatmulBackwardDb::<f32>::new(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
    let target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<MatmulBackwardDb<f32>>(&ptx)?;

    use teeny_cuda::device::CudaLaunchConfig;
    let pk = (k as u32).div_ceil(BLOCK_K as u32);
    let pn = (n as u32).div_ceil(BLOCK_N as u32);
    let cfg = CudaLaunchConfig {
        grid: [pk * pn, 1, 1],
        block: [PTX_LAUNCH_THREADS_X, 1, 1],
        cluster: [1, 1, 1],
    };
    device.launch(
        &program,
        &cfg,
        (
            dc_buf.as_device_ptr() as *mut f32,
            a_buf.as_device_ptr() as *mut f32,
            db_buf.as_device_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;

    db_buf.to_host(&mut db_out)?;
    for i in 0..k * n {
        assert!(
            tf32_close(db_out[i], expected_db[i]),
            "matmul bwd_db mismatch at i={i}: gpu={} expected={}",
            db_out[i],
            expected_db[i]
        );
    }
    Ok(())
}

// ── Per-shape tile selection (spinorml-4gx.2) ─────────────────────────────────
//
// MatmulForward/MatmulBackwardDa/MatmulBackwardDb above are always exercised at a
// fixed BLOCK_M/N/K, independent of graph lowering. These two tests instead go
// through TritonLowering, the same path a real model uses, to confirm
// Op::MatMul's dispatch actually asks pick_gemm_tile_sizes for a shape-appropriate
// tile size rather than a single hardcoded one.

fn build_matmul_graph(m: usize, k: usize, n: usize) -> teeny_core::graph::Graph {
    use teeny_core::graph::{DtypeRepr, Op, SymTensor};

    let (a, graph_rc) = SymTensor::input(DtypeRepr::F32, vec![Some(m), Some(k)]);
    let b_id =
        a.graph
            .borrow_mut()
            .add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(k), Some(n)]);
    let _ = a.graph.borrow_mut().add_node(
        Op::MatMul,
        vec![a.node_id, b_id],
        DtypeRepr::F32,
        vec![Some(m), Some(n)],
    );
    drop(a);
    std::rc::Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

fn lowered_matmul_source(m: usize, k: usize, n: usize) -> String {
    use teeny_core::model::LoweringMode;
    use teeny_kernels::graph::TritonLowering;

    let graph = build_matmul_graph(m, k, n);
    let lowering = TritonLowering::new();
    let (dag, _) = lowering
        .lower_with_mapping(&graph, LoweringMode::Inference)
        .expect("lowering");
    // Nodes: [a (Input), b (Input), matmul] -> matmul is index 2.
    dag.node(2).value.forward_kernel_source().to_string()
}

#[test]
fn test_matmul_lowering_picks_small_tile_for_small_shape() {
    let src = lowered_matmul_source(64, 16, 64);
    assert!(
        src.contains("LlvmTriton, f32, 64, 64, 8, 8"),
        "expected a 64x64x8 tile for a small (64,16,64) shape, got: {src}"
    );
}

#[test]
fn test_matmul_lowering_picks_larger_tile_for_large_shape() {
    let src = lowered_matmul_source(512, 256, 256);
    assert!(
        src.contains("LlvmTriton, f32, 256, 128, 32"),
        "expected a 256x128x32 tile for a large (512,256,256) shape, got: {src}"
    );
}

#[test]
fn test_matmul_lowering_differs_by_shape() {
    // The actual point of this task: two different shapes must not collapse to the
    // same hardcoded tile size.
    let small = lowered_matmul_source(64, 16, 64);
    let large = lowered_matmul_source(512, 256, 256);
    assert_ne!(small, large);
}
