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

//! Anduin reduction-terminated fusion: `reduce_sum(relu(x))` → [`ReduceFuse`]
//! (teenygrad-3w0.9).
//!
//! This is the fusion shape neither `PointwiseFuse` nor `TileFuse` can
//! reach: reduction kernels use a structurally different grid (`BLOCK_INNER`/
//! `n_inner`/`n_outer` row-per-CTA, not `PointwiseFuse`'s flat `BLOCK_SIZE`/
//! `n_elements`), so `Op::Relu`'s chain member has to be re-spliced into the
//! reduction's own load/reduce/store shape rather than concatenated as a
//! whole separate kernel body — teenygrad-1bf's fusion case 4.

use std::rc::Rc;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_kernels::graph::{Anduin, GraphOptimizer, ReduceFuse};

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
use teeny_kernels::testing::load_fixture;

const N: usize = 64;
#[cfg(feature = "cuda")]
const TOL: f32 = 1e-3;

fn shape_1d(n: usize) -> Vec<Option<usize>> {
    vec![None, Some(n)]
}

/// `y = reduce_sum(relu(x))` — single row (n_outer=1 at forward time), so
/// `keepdims=false`'s "reduce all -> scalar" static shape (`[Some(1)]`)
/// matches the runtime shape exactly.
fn build_relu_reduce_sum_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_1d(N));
    let _ = x.graph.borrow_mut().add_node(
        Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: false,
        },
        vec![relu],
        DtypeRepr::F32,
        vec![Some(1)],
    );
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = reduce_sum(x)` (no unary chain) — must not fuse into `ReduceFuse`:
/// there's no chain member for `reduce_chain_parts` to find.
fn build_plain_reduce_sum_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let _ = x.graph.borrow_mut().add_node(
        Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: false,
        },
        vec![x.node_id],
        DtypeRepr::F32,
        vec![Some(1)],
    );
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = reduce_log_sum_exp(relu(x))` — must not fuse: `ReduceLogSumExp`
/// loads its input twice (max-pass then exp-sum-pass), outside `ReduceFuse`
/// v1's supported reduce-op set (see `reduce_fuse.rs`'s module doc).
fn build_relu_reduce_log_sum_exp_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_1d(N));
    let _ = x.graph.borrow_mut().add_node(
        Op::ReduceLogSumExp {
            keepdims: false,
            noop_with_empty_axes: false,
        },
        vec![relu],
        DtypeRepr::F32,
        vec![Some(1)],
    );
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = reduce_sum(sigmoid(relu(x)))` — a 2-op chain member (`Relu` →
/// `Sigmoid`) spliced into the reduction, not just 1. Exercises
/// `ReduceFuse`'s chain-splicing over a genuinely composed chain — the
/// shape-logic case this graph exists to stress, ahead of the Welder-style
/// Tile/shared-memory composition this fusion shape is meant to move to.
fn build_relu_sigmoid_reduce_sum_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_1d(N));
    let sigmoid =
        x.graph
            .borrow_mut()
            .add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape_1d(N));
    let _ = x.graph.borrow_mut().add_node(
        Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: false,
        },
        vec![sigmoid],
        DtypeRepr::F32,
        vec![Some(1)],
    );
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = reduce_sum(relu(x))`, but `relu(x)` also feeds a second consumer
/// (`sigmoid`) — must not fuse: the chain input isn't single-consumer.
fn build_relu_reduce_sum_with_extra_consumer_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_1d(N));
    let _reduce = x.graph.borrow_mut().add_node(
        Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: false,
        },
        vec![relu],
        DtypeRepr::F32,
        vec![Some(1)],
    );
    let _ = x
        .graph
        .borrow_mut()
        .add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape_1d(N));
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_anduin_fuses_relu_reduce_sum_into_reduce_fuse() {
    let graph = build_relu_reduce_sum_graph();
    assert_eq!(graph.nodes.len(), 3, "x, relu, reduce_sum");

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "x Input, ReduceFuse");

    let fused_idx = opt.nodes.len() - 1;
    match &opt.nodes[fused_idx].op {
        Op::Custom { data } => {
            let rf = data
                .downcast_ref::<ReduceFuse>()
                .expect("expected ReduceFuse");
            assert_eq!(rf.chain.len(), 1);
            assert!(matches!(rf.chain[0], Op::Relu));
            assert!(matches!(rf.reduce_op, Op::ReduceSum { .. }));
            // teenygrad-3w0.11: shape-driven, not the old fixed 1024 -- N=64
            // is already a power of two, so the search's answer is exactly
            // N itself (zero padding waste), a 16x smaller BLOCK_INNER than
            // the pre-.11 fixed constant would have picked.
            assert_eq!(
                rf.block_inner, 64,
                "search should pick next_pow2(n_inner) for N=64"
            );
        }
        other => panic!("expected Custom(ReduceFuse), got {other:?}"),
    }
    assert_eq!(opt.nodes[fused_idx].inputs.len(), 1);
}

#[test]
fn test_anduin_fuses_relu_sigmoid_reduce_sum_into_reduce_fuse() {
    let graph = build_relu_sigmoid_reduce_sum_graph();
    assert_eq!(graph.nodes.len(), 4, "x, relu, sigmoid, reduce_sum");

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "x Input, ReduceFuse");

    let fused_idx = opt.nodes.len() - 1;
    match &opt.nodes[fused_idx].op {
        Op::Custom { data } => {
            let rf = data
                .downcast_ref::<ReduceFuse>()
                .expect("expected ReduceFuse");
            assert_eq!(rf.chain.len(), 2, "Relu, Sigmoid");
            assert!(matches!(rf.chain[0], Op::Relu));
            assert!(matches!(rf.chain[1], Op::Sigmoid));
            assert!(matches!(rf.reduce_op, Op::ReduceSum { .. }));
        }
        other => panic!("expected Custom(ReduceFuse), got {other:?}"),
    }
    assert_eq!(opt.nodes[fused_idx].inputs.len(), 1);
}

#[test]
fn test_anduin_does_not_fuse_plain_reduce_sum() {
    let graph = build_plain_reduce_sum_graph();
    assert_eq!(graph.nodes.len(), 2, "x, reduce_sum");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "no fusion: no unary chain to splice");
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<ReduceFuse>().is_none());
        }
    }
}

#[test]
fn test_anduin_does_not_fuse_reduce_log_sum_exp() {
    let graph = build_relu_reduce_log_sum_exp_graph();
    assert_eq!(graph.nodes.len(), 3, "x, relu, reduce_log_sum_exp");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(
        opt.nodes.len(),
        graph.nodes.len(),
        "expected no ReduceFuse fusion (ReduceLogSumExp unsupported), got {opt:?}"
    );
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<ReduceFuse>().is_none());
        }
    }
}

#[test]
fn test_anduin_does_not_fuse_relu_reduce_sum_with_extra_consumer() {
    let graph = build_relu_reduce_sum_with_extra_consumer_graph();
    assert_eq!(graph.nodes.len(), 4, "x, relu, reduce_sum, sigmoid");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(
        opt.nodes.len(),
        graph.nodes.len(),
        "expected no ReduceFuse fusion (relu has 2 consumers), got {opt:?}"
    );
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<ReduceFuse>().is_none());
        }
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_reduce_fuse_relu_sum_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_reduce_sum_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
        false,
    )?;

    assert_eq!(model.dag.len(), 2, "x Input, ReduceFuse");
    let loaded = model.load(&env.device, 1)?;

    let x = load_fixture("anduin_reduce_fuse_relu_sum/x.bin");
    let expected = load_fixture("anduin_reduce_fuse_relu_sum/expected_forward.bin");

    let x_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N) }?;
    // Batch dim dynamic in graph shape `[None, N]` → concrete `[1, N]`.
    let x_tensor = TensorRef::new(x_ptr, vec![1, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor])?;
    let mut y_out = vec![0.0f32; 1];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, 1) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    assert!(
        (y_out[0] - expected[0]).abs() < TOL,
        "mismatch: got {}, expected {}",
        y_out[0],
        expected[0]
    );
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_reduce_fuse_relu_sigmoid_sum_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_sigmoid_reduce_sum_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
        false,
    )?;

    assert_eq!(model.dag.len(), 2, "x Input, ReduceFuse");
    let loaded = model.load(&env.device, 1)?;

    let x = load_fixture("anduin_reduce_fuse_relu_sigmoid_sum/x.bin");
    let expected = load_fixture("anduin_reduce_fuse_relu_sigmoid_sum/expected_forward.bin");

    let x_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N) }?;
    // Batch dim dynamic in graph shape `[None, N]` → concrete `[1, N]`.
    let x_tensor = TensorRef::new(x_ptr, vec![1, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor])?;
    let mut y_out = vec![0.0f32; 1];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, 1) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    assert!(
        (y_out[0] - expected[0]).abs() < TOL,
        "mismatch: got {}, expected {}",
        y_out[0],
        expected[0]
    );
    Ok(())
}

/// Minimal [`Kernel`] wrapper around [`ReduceFuse::lower`]'s raw
/// `(name, source)` output, so it can go through the same `compile_kernel`
/// and `device.launch` path `test_mem_traffic_calibration.rs` established
/// for kernel-level micro-benchmarks — bypassing the graph/model pipeline
/// entirely, since `Op::ReduceSum { keepdims: false, .. }`'s declared
/// "reduce to a scalar" shape doesn't let `n_outer` scale with batch size
/// (a separate, pre-existing shape-resolution gap, out of this issue's
/// scope), which this test needs to control directly to get a measurable
/// aggregate bandwidth difference.
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

/// teenygrad-3w0.11: confirms the search's shape-driven `BLOCK_INNER` choice
/// beats the pre-.11 fixed 1024 on real hardware, for a reduction axis
/// (`n_inner = 100`) that isn't already power-of-two-adjacent to 1024 — the
/// search picks 128 (`next_pow2(100)`), a real 8x reduction in `x_ptr`
/// padding waste. This also doubles as the first real-hardware calibration
/// data point for this specific effect (distinct from `under_parallel_penalty`,
/// see `choose_reduce_fuse_block_inner`'s doc comment) — not yet independently
/// calibrated, same "document honestly" convention `CostModel::window_penalty`
/// already uses.
#[test]
#[cfg(feature = "cuda")]
fn test_reduce_fuse_search_beats_fixed_1024_block_inner() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    const N_INNER: usize = 100;
    const N_OUTER: usize = 100_000;
    // next_pow2(100), what teenygrad-3w0.11's search actually picks --
    // hardcoded here (not called from `choose_reduce_fuse_block_inner`
    // directly) since that function is `pub(crate)` to `teeny-kernels`, not
    // reachable from this external integration-test crate.
    const SEARCH_BLOCK_INNER: i32 = 128;
    const BASELINE_BLOCK_INNER: i32 = 1024;
    const WARMUP_ITERS: usize = 3;
    const TIMED_ITERS: usize = 15;

    let mut x_buf = env.device.buffer::<f32>(N_INNER * N_OUTER)?;
    let y_buf = env.device.buffer::<f32>(N_OUTER)?;
    x_buf.to_device(&vec![1.0f32; N_INNER * N_OUTER])?;

    let run = |block_inner: i32| -> Result<(f64, Vec<f32>)> {
        let fused = ReduceFuse::new(
            vec![Op::Relu],
            Op::ReduceSum {
                keepdims: false,
                noop_with_empty_axes: false,
            },
            DtypeRepr::F32,
            block_inner,
        );
        let (name, source, _entry_point, _rop) =
            fused.lower().expect("ReduceFuse::lower should succeed");
        let kernel = RawKernel { name, source };
        let ptx = std::fs::read(compile_kernel(&kernel, &target, true, false)?)?;
        let program = testing::load_program_from_ptx::<RawKernel>(&ptx)?;
        let cfg = CudaLaunchConfig {
            grid: [N_OUTER as u32, 1, 1],
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
                    N_INNER as i32,
                    N_OUTER as i32,
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

        let mut y_out = vec![0.0f32; N_OUTER];
        y_buf.to_host(&mut y_out)?;
        Ok((samples[samples.len() / 2], y_out))
    };

    let (baseline_s, y_baseline) = run(BASELINE_BLOCK_INNER)?;
    let (search_s, y_search) = run(SEARCH_BLOCK_INNER)?;

    // Correctness: padding past n_inner must never change the reduction --
    // both block sizes should agree exactly (up to fp rounding).
    for (i, (a, b)) in y_baseline.iter().zip(y_search.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "row {i}: block_inner=1024 gave {a}, block_inner={SEARCH_BLOCK_INNER} gave {b}"
        );
    }

    eprintln!(
        "block_inner={BASELINE_BLOCK_INNER}: {baseline_s:.6}s median | \
         block_inner={SEARCH_BLOCK_INNER}: {search_s:.6}s median"
    );
    assert!(
        search_s < baseline_s,
        "search's chosen block_inner={SEARCH_BLOCK_INNER} should be faster than \
         the fixed baseline {BASELINE_BLOCK_INNER} (got {search_s:.6}s vs {baseline_s:.6}s)"
    );
    Ok(())
}
