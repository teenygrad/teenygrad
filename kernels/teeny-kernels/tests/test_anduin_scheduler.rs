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

//! `input -> conv2d -> batchnorm2d -> silu`, built as a real [`Graph`],
//! lowered through the real [`TritonLowering`], compiled and run as a real
//! [`CudaModel`](teeny_cuda::model::CudaModel) -- not individually
//! hand-picked kernels launched by hand -- and checked against a
//! PyTorch-computed `F.silu(F.batch_norm(F.conv2d(x, w), ...))` fixture
//! (`tests/fixtures/anduin_conv2d_bn_silu/`, `generate.py`).
//!
//! Then, on a second lowering of that same graph, applies `Anduin`'s real
//! `GraphOptimizer::optimize` (using a [`HardwareProfile`] queried straight
//! off the real device via [`DeviceInfo::hardware_profile`], not a
//! hand-calibrated test profile) and expects it to panic in one of its
//! `teenygrad-1nr` stubs: `Anduin::schedule` is still a `todo!()`, and once
//! it lands `optimize` replays its winning schedule through `codegen`'s
//! `DagCodegen`, whose every method is a `todo!()` too (teenygrad-1nr.1 --
//! `#[tiled_kernel]`'s `Tile<D>`-composition rework hasn't landed yet). See
//! `codegen`'s and this crate's `anduin` module doc comments. The numeric
//! checks run on the unoptimized lowering *before* `optimize`, so a real
//! regression there still fails loudly instead of being masked by the
//! expected panic.
//!
//! The output shape is deliberately larger than any real GPU's default CUDA
//! shared-memory budget (see the comment on the test's constants) so that `Anduin::optimize`'s
//! `SubGraphTiling` search must actually tile -- `enumerate_subtiles` has
//! to find a subtile that fits, not just return the whole tensor untiled --
//! rather than the search trivially having nothing to do.

#[test]
#[cfg(all(feature = "cuda", feature = "hardware"))]
#[should_panic(expected = "teenygrad-1nr:")]
fn test_conv2d_batchnorm_silu_matches_pytorch_then_reaches_anduin_stub() {
    use dotenv::dotenv;
    use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
    use teeny_core::device::context::DeviceInfo;
    use teeny_core::model::LoweringMode;
    use teeny_cuda::compiler::graph::CudaGraphCompiler;
    use teeny_cuda::compiler::target::Target;
    use teeny_cuda::model::TensorRef;
    use teeny_kernels::graph::{Anduin, GraphOptimizer, TritonLowering};
    use teeny_test::cuda as testing;
    use teeny_test::graph::Conv2dBnSiluBuilder;
    use teeny_test::{load_fixture, teenyc_cache_dir};

    // B = 2 (not 1) so BatchNorm2d's 2-D `[C, B]` grid has a real second
    // dimension.
    //
    // Together with C_OUT/HW, sized to force real tiling in Anduin's
    // SubGraphTiling search (teenygrad-1nr), not just trivially fit in one
    // shot. Measured on a real RTX 5070 (`DeviceInfo::hardware_profile`):
    // SharedMemory capacity is 49152 bytes (48KB,
    // `cudaDeviceProp.sharedMemPerBlock` -- essentially every CUDA
    // architecture's default, unchanged since Kepler), Register capacity
    // (`regs_per_block * 4`) is 262144 bytes. The `[B, C_OUT, HW, HW]` f32
    // output (and each intermediate) is 1048576 bytes -- 4x the Register
    // capacity and >21x the SharedMemory capacity -- so no tile in the chain
    // can fit untiled at either level. Matches `tests/fixtures/generate.py`'s
    // `anduin_conv2d_bn_silu` section, which the fixtures below come from.
    const B: usize = 2;
    const C_IN: usize = 3;
    const C_OUT: usize = 8;
    // Input height and width; the builder's default 3x3 kernel, stride 1 and
    // padding 1 keep the output the same.
    const HW: usize = 128;
    // Looser than a pointwise chain's 1e-5: each output accumulates a
    // `C_IN * 3 * 3` = 27-term dot product before the BN affine.
    const TOL: f32 = 1e-4;

    dotenv().ok();
    let env = testing::setup_cuda_env().expect("cuda env setup should not fail here");
    let target = Target::new(env.capability);

    // ── Build graph: input -> conv2d -> batchnorm2d -> silu ─────────────────
    let block = Conv2dBnSiluBuilder::default()
        .batch(B)
        .in_channels(C_IN)
        .out_channels(C_OUT)
        .height(HW)
        .width(HW)
        .build()
        .expect("every required field is set")
        .graph()
        .expect("graph construction should not fail here");
    let graph = &block.graph;

    // ── Lower, compile and run the unoptimized graph for real ───────────────
    let lowering = TritonLowering::new();
    let (op_dag, graph_to_dag, lowered_graph) = lowering
        .lower_with_mapping(graph, LoweringMode::Inference)
        .expect("lowering should not fail here");
    let (conv_idx, bn_idx) = (graph_to_dag[block.conv], graph_to_dag[block.bn]);

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
    let mut loaded = model
        .load(&env.device, B)
        .expect("load should not fail here");

    let fixture = |name: &str| {
        load_fixture(
            env!("CARGO_MANIFEST_DIR"),
            &format!("anduin_conv2d_bn_silu/{name}.bin"),
        )
    };
    let x_host = fixture("x");
    let expected = fixture("expected_forward");
    let n_out = B * C_OUT * HW * HW;
    assert_eq!(x_host.len(), B * C_IN * HW * HW);
    assert_eq!(expected.len(), n_out);

    // Conv2dForward's one slot is its weight; BatchNorm2d's slots are
    // `weight, bias, running_mean, running_var` (its `param_names`).
    loaded
        .load_param_f32(conv_idx, 0, &fixture("w"))
        .expect("conv weight upload should not fail");
    for (slot, name) in ["bn_weight", "bn_bias", "bn_running_mean", "bn_running_var"]
        .into_iter()
        .enumerate()
    {
        loaded
            .load_param_f32(bn_idx, slot, &fixture(name))
            .unwrap_or_else(|e| panic!("{name} upload should not fail: {e}"));
    }

    let x_tensor = TensorRef::from_host_f32(&x_host, vec![B, C_IN, HW, HW])
        .expect("tensor upload should not fail");
    let output = loaded
        .forward(&env.device, B, &[x_tensor])
        .expect("forward should not fail here");
    let y_host = output.to_host_f32().expect("readback should not fail");
    output.free().expect("free should not fail");

    assert_eq!(y_host.len(), n_out);
    for i in 0..n_out {
        assert!(
            (y_host[i] - expected[i]).abs() < TOL,
            "conv2d->batchnorm2d->silu mismatch at {i}: got={}, expected={}",
            y_host[i],
            expected[i]
        );
    }

    // ── Schedule the same graph with Anduin ─────────────────────────────────
    let (op_dag, graph_to_dag, _) = lowering
        .lower_with_mapping(graph, LoweringMode::Inference)
        .expect("lowering should not fail here");
    let hardware = env.device.info().hardware_profile();
    let _ = Anduin.optimize(op_dag, graph_to_dag, &hardware);
}
