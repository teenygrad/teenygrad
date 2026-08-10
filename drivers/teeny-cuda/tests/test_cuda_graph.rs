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

//! Tests for CUDA graph capture and replay via [`LoadedModel::capture_graph`].
//!
//! The test builds a minimal Input → SiLU model, runs forward to get a
//! reference output, then captures a CUDA graph and verifies that repeated
//! [`CudaGraphModel::run`] calls produce identical results.

use std::mem::size_of;

use dotenv::dotenv;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_cuda::compiler::target::Target;
use teeny_core::{
    graph::{DtypeRepr, SymTensor},
    model::LoweringMode,
    nn::{Layer, activation::sigmoid::Silu},
};
use teeny_cuda::{
    compiler::graph::CudaGraphCompiler, device::mem, errors::Result, model::TensorRef, testing,
};
use teeny_kernels::graph::TritonLowering;

const BATCH: usize = 4;
const FEATURES: usize = 128;
const TOL: f32 = 1e-6;

#[test]
fn test_cuda_graph_silu_matches_forward() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    // ── Build graph: Input → SiLU ──────────────────────────────────────────
    let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(FEATURES)]);
    let _output = Layer::call(&Silu::<f32, SymTensor, 2>::default(), input);
    let graph = graph.borrow();

    // ── Compile ────────────────────────────────────────────────────────────
    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let lowering = TritonLowering::new();
    let model =
        graph_compiler.compile_model(&graph, &lowering, &target, LoweringMode::Inference, false)?;

    // ── Load ───────────────────────────────────────────────────────────────
    let loaded = model.load(&env.device, BATCH)?;

    // ── Input data ─────────────────────────────────────────────────────────
    let x: Vec<f32> = (0..BATCH * FEATURES)
        .map(|i| (i as f32 - (BATCH * FEATURES / 2) as f32) * 0.1)
        .collect();

    // ── Reference: LoadedModel::forward ────────────────────────────────────
    let x_ptr = mem::alloc(BATCH * FEATURES * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), BATCH * FEATURES) }?;
    let x_tensor = TensorRef::new(x_ptr, vec![BATCH, FEATURES]);

    let fwd_out = loaded.forward(&env.device, BATCH, &[x_tensor])?;
    let mut reference = vec![0.0_f32; BATCH * FEATURES];
    unsafe { mem::copy_d_to_h(reference.as_mut_ptr(), fwd_out.ptr, BATCH * FEATURES) }?;
    mem::free(fwd_out.ptr)?;
    mem::free(x_ptr)?;

    // ── Capture CUDA graph ──────────────────────────────────────────────────
    let terminals = loaded.terminal_node_indices_sorted_by_size();
    let graph_model =
        loaded.capture_graph(&env.device, BATCH, &[vec![BATCH, FEATURES]], &terminals)?;
    assert_eq!(graph_model.output_shapes()[0], &[BATCH, FEATURES]);

    // ── First run: compare against forward reference ───────────────────────
    let run1 = graph_model.run(&[x.as_slice()])?;
    assert_eq!(run1[0].len(), BATCH * FEATURES);
    for i in 0..BATCH * FEATURES {
        assert!(
            (run1[0][i] - reference[i]).abs() < TOL,
            "run1[{i}]: graph={} forward={}",
            run1[0][i],
            reference[i]
        );
    }

    // ── Second run: verify deterministic replay ───────────────────────────
    let run2 = graph_model.run(&[x.as_slice()])?;
    for i in 0..BATCH * FEATURES {
        assert_eq!(
            run1[0][i], run2[0][i],
            "run2 differs from run1 at index {i}"
        );
    }

    // ── Third run with different input ────────────────────────────────────
    let x2: Vec<f32> = (0..BATCH * FEATURES).map(|i| i as f32 * 0.01).collect();
    let run3 = graph_model.run(&[x2.as_slice()])?;

    // Verify run3 differs from run1 (different input → different output).
    let any_different = run3[0]
        .iter()
        .zip(run1[0].iter())
        .any(|(a, b)| (a - b).abs() > TOL);
    assert!(
        any_different,
        "run3 should differ from run1 since input changed"
    );

    Ok(())
}
