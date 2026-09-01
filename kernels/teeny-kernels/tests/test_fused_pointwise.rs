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

//! `input -> relu -> silu`, built as a real [`Graph`], lowered through the
//! real [`TritonLowering`], compiled and run as a real
//! [`CudaModel`](teeny_cuda::model::CudaModel) -- not individually
//! hand-picked kernels launched by hand -- and checked against a
//! PyTorch-computed `F.silu(F.relu(x))` fixture
//! (`tests/fixtures/fused_pointwise/`, `generate.py`).
//!
//! Then, on that same lowered graph, applies `Anduin`'s real
//! `GraphOptimizer::optimize` (using a [`HardwareProfile`] queried straight
//! off the real device via [`DeviceInfo::hardware_profile`], not a
//! hand-calibrated test profile) and expects it to panic: `optimize`
//! replays its winning schedule through `codegen`'s `DagCodegen` to
//! materialize a fused custom-op kernel, and every `DagCodegen` method is
//! still a `todo!()` stub (teenygrad-1nr.1 -- `#[tiled_kernel]`'s `Tile<D>`-
//! composition rework hasn't landed yet). See `codegen`'s and this crate's
//! `anduin` module doc comments, which say so plainly. The numeric checks
//! above run and gate the test *before* this, so a real regression there
//! still fails loudly instead of being masked by the expected panic.
//!
//! `N` is deliberately larger than any real GPU's default CUDA
//! shared-memory budget (see `N`'s own doc comment) so that
//! `Anduin::optimize`'s `SubGraphTiling` search must actually tile --
//! `enumerate_subtiles` has to find a subtile that fits, not just return
//! the whole tensor untiled -- rather than the search trivially having
//! nothing to do.

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
#[cfg(feature = "cuda")]
use teeny_core::device::context::DeviceInfo;
#[cfg(feature = "cuda")]
use teeny_core::graph::{DtypeRepr, Graph, Op};
#[cfg(feature = "cuda")]
use teeny_core::model::LoweringMode;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::graph::CudaGraphCompiler;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::target::Target;
#[cfg(feature = "cuda")]
use teeny_cuda::{model::TensorRef, testing};
#[cfg(feature = "cuda")]
use teeny_kernels::graph::{Anduin, GraphOptimizer, TritonLowering};
#[cfg(feature = "cuda")]
use teeny_kernels::testing::{load_fixture, teenyc_cache_dir};

// Sized to force real tiling in Anduin's SubGraphTiling search
// (teenygrad-1nr), not just trivially fit in one shot. Measured on a real
// RTX 5070 (`DeviceInfo::hardware_profile`): SharedMemory capacity is
// 49152 bytes (48KB, `cudaDeviceProp.sharedMemPerBlock` -- essentially
// every CUDA architecture's default, unchanged since Kepler), Register
// capacity (`regs_per_block * 4`) is 262144 bytes. `N * 4` bytes
// (1048576, f32) is 4x the Register capacity and >21x the SharedMemory
// capacity, so the whole relu/silu output tile cannot fit untiled at
// either level -- `enumerate_subtiles` must actually search for a
// smaller subtile instead of trivially returning the full shape as its
// own best candidate. Matches `tests/fixtures/generate.py`'s `N_FUSED`,
// which the fixtures below were regenerated from.
#[cfg(feature = "cuda")]
const N: usize = 262144;
#[cfg(feature = "cuda")]
const BATCH: usize = 1;
#[cfg(feature = "cuda")]
const TOL: f32 = 1e-5;

#[test]
#[cfg(feature = "cuda")]
#[should_panic(
    expected = "teenygrad-1nr: begin generating a custom op for this virtual node's group"
)]
fn test_fused_pointwise_relu_then_silu() {
    dotenv().ok();
    let env = testing::setup_cuda_env().expect("cuda env setup should not fail here");
    let target = Target::new(env.capability);

    // ── Build graph: input -> relu -> silu ──────────────────────────────────
    let shape = vec![Some(BATCH), Some(N)];
    let mut graph = Graph::new();
    let input = graph.add_node(Op::Input, vec![], DtypeRepr::F32, shape.clone());
    let relu = graph.add_node(Op::Relu, vec![input], DtypeRepr::F32, shape.clone());
    graph.add_node(Op::Silu, vec![relu], DtypeRepr::F32, shape);

    // ── Lower ────────────────────────────────────────────────────────────────
    let lowering = TritonLowering::new();
    let (op_dag, graph_to_dag, lowered_graph) = lowering
        .lower_with_mapping(&graph, LoweringMode::Inference)
        .expect("lowering should not fail here");
    let hardware = env.device.info().hardware_profile();
    let (op_dag, graph_to_dag) = Anduin
        .optimize(op_dag, graph_to_dag, &hardware)
        .expect("optimize should not fail here");

    // ── Compile the lowered DAG and run it for real ─────────────────────────
    let compiler = LlvmCompiler::new(
        teeny_compiler::compiler::find_teenyc().expect("teenyc should be found"),
        teenyc_cache_dir(),
    )
    .expect("compiler construction should not fail here");
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let model = graph_compiler
        .compile_lowered(
            op_dag,
            graph_to_dag,
            &lowered_graph,
            &lowering,
            &target,
            false,
        )
        .expect("compile should not fail here");
    let loaded = model
        .load(&env.device, BATCH)
        .expect("load should not fail here");

    let x_host = load_fixture("fused_pointwise/x.bin");
    let expected = load_fixture("fused_pointwise/expected_forward.bin");
    assert_eq!(x_host.len(), BATCH * N);

    let x_tensor =
        TensorRef::from_host_f32(&x_host, vec![BATCH, N]).expect("tensor upload should not fail");
    let output = loaded
        .forward(&env.device, BATCH, &[x_tensor])
        .expect("forward should not fail here");
    let y_host = output.to_host_f32().expect("readback should not fail");
    output.free().expect("free should not fail");

    for i in 0..N {
        assert!(
            (y_host[i] - expected[i]).abs() < TOL,
            "fused relu->silu mismatch at {i}: x={}, got={}, expected={}",
            x_host[i],
            y_host[i],
            expected[i]
        );
    }
}
