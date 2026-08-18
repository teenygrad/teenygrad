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

//! Anduin transpose-terminated fusion: `transpose(relu(x))` →
//! [`SharedTransposeFuse`] (teenygrad-3w0.10's `SetConnect` demonstration).
//!
//! This is the shared-memory tier: `T::trans` genuinely needs cross-thread
//! data movement (it already stages through shared memory transparently,
//! same as `tt.reduce`), so the chain ahead of it can't be spliced at the
//! register level the way `ReduceFuse` splices into a reduction — but
//! fusing it directly into `transpose_2d_forward`'s own kernel still avoids
//! materializing the chain's output to global memory first, which is
//! exactly the global-tier baseline this module's GPU test compares
//! against.

use std::rc::Rc;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_kernels::graph::{Anduin, GraphOptimizer, SharedTransposeFuse};

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use std::mem::size_of;
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use teeny_core::device::Device;
#[cfg(feature = "cuda")]
use teeny_core::device::buffer::Buffer;
#[cfg(feature = "cuda")]
use teeny_core::device::program::Kernel;
#[cfg(feature = "cuda")]
use teeny_core::graph::CustomOp;
#[cfg(feature = "cuda")]
use teeny_core::model::LoweringMode;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::target::Target;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::{compile_cuda_graph, compile_kernel};
#[cfg(feature = "cuda")]
use teeny_cuda::device::CudaLaunchConfig;
#[cfg(feature = "cuda")]
use teeny_cuda::device::mem;
#[cfg(feature = "cuda")]
use teeny_cuda::errors::Result;
#[cfg(feature = "cuda")]
use teeny_cuda::model::TensorRef;
#[cfg(feature = "cuda")]
use teeny_cuda::testing;
#[cfg(feature = "cuda")]
use teeny_kernels::graph::TritonLowering;
#[cfg(feature = "cuda")]
use teeny_kernels::nn::activation::relu::ReluForward;
#[cfg(feature = "cuda")]
use teeny_kernels::nn::tensor::transpose::Transpose2dForward;

const M: usize = 128;
const N: usize = 128;
#[cfg(feature = "cuda")]
const TOL: f32 = 1e-4;

fn shape_2d(m: usize, n: usize) -> Vec<Option<usize>> {
    vec![Some(m), Some(n)]
}

/// `y = transpose(relu(x))`.
fn build_relu_transpose_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_2d(M, N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_2d(M, N));
    let _ = x.graph.borrow_mut().add_node(
        Op::Transpose { perm: vec![] },
        vec![relu],
        DtypeRepr::F32,
        shape_2d(N, M),
    );
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = transpose(x)` (no unary chain) — must not fuse: there's no chain
/// member for the pass to find.
fn build_plain_transpose_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_2d(M, N));
    let _ = x.graph.borrow_mut().add_node(
        Op::Transpose { perm: vec![] },
        vec![x.node_id],
        DtypeRepr::F32,
        shape_2d(N, M),
    );
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = transpose(relu(x))`, but `relu(x)` also feeds a second consumer
/// (`sigmoid`) — must not fuse: the chain input isn't single-consumer.
fn build_relu_transpose_with_extra_consumer_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_2d(M, N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_2d(M, N));
    let _transpose = x.graph.borrow_mut().add_node(
        Op::Transpose { perm: vec![] },
        vec![relu],
        DtypeRepr::F32,
        shape_2d(N, M),
    );
    let _ = x
        .graph
        .borrow_mut()
        .add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape_2d(M, N));
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_anduin_fuses_relu_transpose_into_shared_transpose_fuse() {
    let graph = build_relu_transpose_graph();
    assert_eq!(graph.nodes.len(), 3, "x, relu, transpose");

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "x Input, SharedTransposeFuse");

    let fused_idx = opt.nodes.len() - 1;
    match &opt.nodes[fused_idx].op {
        Op::Custom { data } => {
            let stf = data
                .downcast_ref::<SharedTransposeFuse>()
                .expect("expected SharedTransposeFuse");
            assert_eq!(stf.chain.len(), 1);
            assert!(matches!(stf.chain[0], Op::Relu));
            // M=N=128 is covered by the {16,32,64,128} candidate ladder.
            assert!(128 % stf.block_m == 0);
            assert!(128 % stf.block_n == 0);
        }
        other => panic!("expected Custom(SharedTransposeFuse), got {other:?}"),
    }
    assert_eq!(opt.nodes[fused_idx].inputs.len(), 1);
}

#[test]
fn test_anduin_does_not_fuse_plain_transpose() {
    let graph = build_plain_transpose_graph();
    assert_eq!(graph.nodes.len(), 2, "x, transpose");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "no fusion: no unary chain to splice");
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<SharedTransposeFuse>().is_none());
        }
    }
}

#[test]
fn test_anduin_does_not_fuse_relu_transpose_with_extra_consumer() {
    let graph = build_relu_transpose_with_extra_consumer_graph();
    assert_eq!(graph.nodes.len(), 4, "x, relu, transpose, sigmoid");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(
        opt.nodes.len(),
        graph.nodes.len(),
        "expected no SharedTransposeFuse fusion (relu has 2 consumers), got {opt:?}"
    );
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<SharedTransposeFuse>().is_none());
        }
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_shared_transpose_fuse_relu_transpose_matches_reference() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_transpose_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
        false,
    )?;

    assert_eq!(model.dag.len(), 2, "x Input, SharedTransposeFuse");
    let loaded = model.load(&env.device, 1)?;

    let x_data: Vec<f32> = (0..M * N)
        .map(|i| (i as f32) - (M * N) as f32 / 2.0)
        .collect();
    let x_ptr = mem::alloc(M * N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x_data.as_ptr(), M * N) }?;
    let x_tensor = TensorRef::new(x_ptr, vec![M, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor])?;
    let mut y_out = vec![0.0f32; N * M];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N * M) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    for m in 0..M {
        for n in 0..N {
            let expected = x_data[m * N + n].max(0.0);
            let got = y_out[n * M + m];
            assert!(
                (expected - got).abs() < TOL,
                "mismatch at (m={m}, n={n}): expected {expected}, got {got}"
            );
        }
    }
    Ok(())
}

/// Minimal [`Kernel`] wrapper around [`SharedTransposeFuse::lower`]'s raw
/// `(name, source)` output — same precedent as
/// `test_anduin_reduce_fuse.rs`'s `RawKernel`.
#[cfg(feature = "cuda")]
struct RawKernel {
    name: String,
    source: String,
}

#[cfg(feature = "cuda")]
impl Kernel for RawKernel {
    type Args<'a> = (*mut f32, *mut f32, i32, i32);

    fn name(&self) -> &str {
        &self.name
    }

    fn source(&self) -> &str {
        &self.source
    }

    fn kernel_source(&self) -> &str {
        &self.source
    }

    fn entry_point_source(&self) -> &str {
        &self.source
    }
}

/// teenygrad-3w0.10's `SetConnect` acceptance test: the fused kernel (one
/// launch, chain computed in registers, transpose reads it straight through
/// shared memory) against the unfused/global-tier baseline (two launches: a
/// plain `relu_forward` materializing its output to a scratch buffer in
/// global memory, then `transpose_2d_forward` reading it back) — the
/// concrete case the shared-memory tier exists for.
#[test]
#[cfg(feature = "cuda")]
fn test_shared_transpose_fuse_beats_unfused_global_round_trip() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    const BIG_M: usize = 8192;
    const BIG_N: usize = 4096;
    const BLOCK_M: i32 = 64;
    const BLOCK_N: i32 = 64;
    const RELU_BLOCK_SIZE: i32 = 1024;
    const WARMUP_ITERS: usize = 5;
    const TIMED_ITERS: usize = 25;

    let mut x_buf = env.device.buffer::<f32>(BIG_M * BIG_N)?;
    let scratch_buf = env.device.buffer::<f32>(BIG_M * BIG_N)?;
    let y_buf = env.device.buffer::<f32>(BIG_M * BIG_N)?;
    let x_data: Vec<f32> = (0..BIG_M * BIG_N)
        .map(|i| (i as f32) - (BIG_M * BIG_N) as f32 / 2.0)
        .collect();
    x_buf.to_device(&x_data)?;

    // Fused: one kernel, x -> y directly.
    let run_fused = || -> Result<(f64, Vec<f32>)> {
        let fused = SharedTransposeFuse::new(vec![Op::Relu], DtypeRepr::F32, BLOCK_M, BLOCK_N);
        let (name, source, _entry_point, _rop) = fused
            .lower()
            .expect("SharedTransposeFuse::lower should succeed");
        let kernel = RawKernel { name, source };
        let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
        let program = testing::load_program_from_ptx::<RawKernel>(&ptx)?;
        let pm = (BIG_M as u32).div_ceil(BLOCK_M as u32);
        let pn = (BIG_N as u32).div_ceil(BLOCK_N as u32);
        let cfg = CudaLaunchConfig {
            grid: [pm * pn, 1, 1],
            block: [program.threads_per_block(), 1, 1],
            cluster: [program.num_ctas().max(1), 1, 1],
        };
        let launch = || {
            env.device.launch(
                &program,
                &cfg,
                (
                    x_buf.as_device_ptr() as *mut f32,
                    y_buf.as_device_ptr() as *mut f32,
                    BIG_M as i32,
                    BIG_N as i32,
                ),
            )
        };
        for _ in 0..WARMUP_ITERS {
            launch()?;
        }
        let mut samples = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            let start = Instant::now();
            launch()?;
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut y_out = vec![0.0f32; BIG_M * BIG_N];
        y_buf.to_host(&mut y_out)?;
        Ok((samples[samples.len() / 2], y_out))
    };

    // Unfused: relu_forward (x -> scratch, global round-trip) then
    // transpose_2d_forward (scratch -> y), timed together per iteration.
    let run_unfused = || -> Result<(f64, Vec<f32>)> {
        let relu_kernel = ReluForward::<f32>::new(RELU_BLOCK_SIZE);
        let relu_ptx = std::fs::read(compile_kernel(&relu_kernel, &target, true, false)?)?;
        let relu_program = testing::load_program_from_ptx::<ReluForward<f32>>(&relu_ptx)?;
        let n_elements = (BIG_M * BIG_N) as i32;
        let relu_cfg = CudaLaunchConfig {
            grid: [(n_elements as u32).div_ceil(RELU_BLOCK_SIZE as u32), 1, 1],
            block: [relu_program.threads_per_block(), 1, 1],
            cluster: [relu_program.num_ctas().max(1), 1, 1],
        };

        let transpose_kernel = Transpose2dForward::<f32>::new(BLOCK_M, BLOCK_N);
        let transpose_ptx =
            std::fs::read(compile_kernel(&transpose_kernel, &target, true, false)?)?;
        let transpose_program =
            testing::load_program_from_ptx::<Transpose2dForward<f32>>(&transpose_ptx)?;
        let pm = (BIG_M as u32).div_ceil(BLOCK_M as u32);
        let pn = (BIG_N as u32).div_ceil(BLOCK_N as u32);
        let transpose_cfg = CudaLaunchConfig {
            grid: [pm * pn, 1, 1],
            block: [transpose_program.threads_per_block(), 1, 1],
            cluster: [transpose_program.num_ctas().max(1), 1, 1],
        };

        let launch = || {
            env.device.launch(
                &relu_program,
                &relu_cfg,
                (
                    x_buf.as_device_ptr() as *mut f32,
                    scratch_buf.as_device_ptr() as *mut f32,
                    n_elements,
                ),
            )?;
            env.device.launch(
                &transpose_program,
                &transpose_cfg,
                (
                    scratch_buf.as_device_ptr() as *mut f32,
                    y_buf.as_device_ptr() as *mut f32,
                    BIG_M as i32,
                    BIG_N as i32,
                ),
            )
        };
        for _ in 0..WARMUP_ITERS {
            launch()?;
        }
        let mut samples = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            let start = Instant::now();
            launch()?;
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut y_out = vec![0.0f32; BIG_M * BIG_N];
        y_buf.to_host(&mut y_out)?;
        Ok((samples[samples.len() / 2], y_out))
    };

    let (unfused_s, y_unfused) = run_unfused()?;
    let (fused_s, y_fused) = run_fused()?;

    for (i, (a, b)) in y_unfused.iter().zip(y_fused.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "index {i}: unfused gave {a}, fused gave {b}"
        );
    }
    for m in 0..BIG_M {
        for n in 0..BIG_N {
            let expected = x_data[m * BIG_N + n].max(0.0);
            let got = y_fused[n * BIG_M + m];
            assert!(
                (expected - got).abs() < 1e-4,
                "fused mismatch at (m={m}, n={n}): expected {expected}, got {got}"
            );
        }
    }

    eprintln!(
        "unfused (2 kernels, global round-trip): {unfused_s:.6}s median | fused (1 kernel): {fused_s:.6}s median"
    );
    assert!(
        fused_s < unfused_s,
        "fused SharedTransposeFuse should beat the unfused global round-trip \
         (got {fused_s:.6}s vs {unfused_s:.6}s)"
    );
    Ok(())
}
