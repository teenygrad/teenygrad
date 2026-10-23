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

//! Softmax built from primitives -- `x -> max -> sub(x, .) -> exp -> sum ->
//! div(exp, .)` over `[rows, cols]`, both reductions keepdims -- lowered
//! through the real [`TritonLowering`], compiled and run as a real
//! `CudaModel`, and checked against a CPU softmax.
//!
//! Both `sub` and `div` broadcast a `[rows, 1]` operand over the columns, so
//! they lower to the row-broadcast kernels. The flat elementwise kernels would
//! index that operand by the output's flat index and read past its end.

#[test]
#[cfg(all(feature = "cuda", feature = "hardware"))]
fn test_decomposed_softmax_matches_a_cpu_softmax() {
    use dotenv::dotenv;
    use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
    use teeny_core::model::LoweringMode;
    use teeny_cuda::compiler::graph::CudaGraphCompiler;
    use teeny_cuda::compiler::target::Target;
    use teeny_cuda::model::TensorRef;
    use teeny_kernels::graph::TritonLowering;
    use teeny_test::cuda as testing;
    use teeny_test::graph::DecomposedSoftmaxBuilder;
    use teeny_test::teenyc_cache_dir;

    // More columns than one 1024-wide row-broadcast block, and a ragged tail,
    // so both the column tiling and its mask are exercised.
    const ROWS: usize = 6;
    const COLS: usize = 1500;
    const TOL: f32 = 1e-6;

    dotenv().ok();
    let env = testing::setup_cuda_env().expect("cuda env setup should not fail here");
    let target = Target::new(env.capability);

    let softmax = DecomposedSoftmaxBuilder::default()
        .rows(ROWS)
        .cols(COLS)
        .build()
        .expect("every required field is set")
        .graph();

    let lowering = TritonLowering::new();
    let (op_dag, graph_to_dag, lowered_graph) = lowering
        .lower_with_mapping(&softmax.graph, LoweringMode::Inference)
        .expect("lowering should not fail here");
    let compiler = LlvmCompiler::new(
        teeny_compiler::compiler::find_teenyc().expect("teenyc should be found"),
        teenyc_cache_dir(),
    )
    .expect("compiler construction should not fail here");
    let model = CudaGraphCompiler::new(compiler)
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
        .load(&env.device, ROWS)
        .expect("load should not fail here");

    // Rows with different offsets and spreads, so a reduction that mixed rows
    // or skipped the max subtraction would show.
    let x: Vec<f32> = (0..ROWS * COLS)
        .map(|i| {
            let (r, c) = (i / COLS, i % COLS);
            (r as f32 * 3.0 - 7.0) + ((c * 37 + r * 11) % 101) as f32 / (10.0 + r as f32)
        })
        .collect();
    let expected: Vec<f32> = x
        .chunks(COLS)
        .flat_map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f64> = row.iter().map(|&v| ((v - max) as f64).exp()).collect();
            let sum: f64 = exp.iter().sum();
            exp.into_iter().map(move |e| (e / sum) as f32)
        })
        .collect();

    let x_tensor =
        TensorRef::from_host_f32(&x, vec![ROWS, COLS]).expect("tensor upload should not fail");
    let output = loaded
        .forward(&env.device, ROWS, &[x_tensor])
        .expect("forward should not fail here");
    let y = output.to_host_f32().expect("readback should not fail");
    output.free().expect("free should not fail");

    assert_eq!(y.len(), ROWS * COLS);
    for (i, (&got, &want)) in y.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < TOL,
            "softmax mismatch at row {}, col {}: got={got}, expected={want}",
            i / COLS,
            i % COLS
        );
    }
    for (r, row) in y.chunks(COLS).enumerate() {
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row {r} sums to {total}, not 1");
    }
}
