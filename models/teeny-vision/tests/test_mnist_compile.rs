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

//! Integration test: trace the MNIST LeNet-5 model, lower it, and compile
//! every kernel to PTX using the CUDA graph compiler.

use dotenv::dotenv;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::{
    graph::{DtypeRepr, SymTensor},
    nn::Layer,
};
use teeny_cuda::compiler::graph::CudaGraphCompiler;
use teeny_cuda::testing;
use teeny_kernels::graph::TritonLowering;
use teeny_vision::mnist;

#[test]
fn test_mnist_graph_compiles() -> anyhow::Result<()> {
    dotenv().ok();

    // ── CUDA environment ──────────────────────────────────────────────────────
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    // ── Trace the MNIST (LeNet-5) graph ───────────────────────────────────────
    // Input: batch of single-channel 28×28 images (batch dim is dynamic).
    let (input, graph) =
        SymTensor::input(DtypeRepr::F32, vec![None, Some(1), Some(28), Some(28)]);
    let _output = Layer::call(&mnist::mnist::<f32>(), input);
    let graph = graph.borrow();

    // LeNet-5 has 14 graph nodes: 1 Input + 13 ops.
    assert_eq!(graph.nodes.len(), 14, "unexpected number of graph nodes");
    println!("[1/3] traced MNIST graph: {} nodes", graph.nodes.len());

    // ── Build the compiler ────────────────────────────────────────────────────
    let rustc_path = std::env::var("TEENY_RUSTC_PATH")
        .expect("TEENY_RUSTC_PATH must be set to run this test");
    let cache_dir =
        std::env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());
    let compiler = LlvmCompiler::new(rustc_path, cache_dir)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    println!("[2/3] built graph compiler (target: {})", env.capability);

    // ── Compile ───────────────────────────────────────────────────────────────
    let lowering = TritonLowering::new();
    let model = graph_compiler.compile_model(&graph, &lowering, &target, false)?;
    println!("[3/3] compiled all kernels");

    // ── Verify compiled DAG ───────────────────────────────────────────────────
    // Every node except the Input placeholder must have a non-empty ptx_path
    // pointing to a file that actually exists on disk.
    let dag = &model.dag;
    assert_eq!(dag.len(), 14, "compiled DAG should have same node count as graph");

    let topo = dag.topological_sort();
    let mut compiled_count = 0;

    for &idx in &topo {
        let node = dag.node(idx);
        let cn = &node.value;

        if cn.ptx_path.is_empty() {
            // Input placeholder — expected to have no kernel.
            println!("  node {idx}: Input (no kernel)");
        } else {
            assert!(
                std::path::Path::new(&cn.ptx_path).exists(),
                "PTX file not found for node {idx}: {}",
                cn.ptx_path,
            );
            println!(
                "  node {idx}: {} → {}",
                cn.entry_point,
                cn.ptx_path,
            );
            compiled_count += 1;
        }
    }

    assert_eq!(compiled_count, 13, "expected 13 compiled kernels (all ops except Input)");

    Ok(())
}
