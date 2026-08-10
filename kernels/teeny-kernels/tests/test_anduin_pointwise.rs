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

//! Anduin pointwise fusion: `Input → Relu → Sigmoid` → [`PointwiseFuse`].

use std::rc::Rc;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::{Anduin, GraphOptimizer, PointwiseFuse, TritonLowering};

#[cfg(feature = "cuda")]
use std::mem::size_of;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use insta::assert_debug_snapshot;
#[cfg(feature = "cuda")]
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
#[cfg(feature = "cuda")]
use teeny_core::device::program::Kernel;
#[cfg(feature = "cuda")]
use teeny_core::model::ExecutableOp;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::compile_kernel;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::graph::CudaGraphCompiler;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::target::{Capability, Target};
#[cfg(feature = "cuda")]
use teeny_cuda::device::mem;
#[cfg(feature = "cuda")]
use teeny_cuda::errors::Result;
#[cfg(feature = "cuda")]
use teeny_cuda::model::TensorRef;
#[cfg(feature = "cuda")]
use teeny_cuda::testing;

const N: usize = 64;
const TOL: f32 = 1e-4;

fn shape_1d(n: usize) -> Vec<Option<usize>> {
    vec![None, Some(n)]
}

fn build_relu_sigmoid_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu = input.graph.borrow_mut().add_node(
        Op::Relu,
        vec![input.node_id],
        DtypeRepr::F32,
        shape_1d(N),
    );
    let _ = input.graph.borrow_mut().add_node(
        Op::Sigmoid,
        vec![relu],
        DtypeRepr::F32,
        shape_1d(N),
    );
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_anduin_fuses_relu_sigmoid_into_pointwise_fuse() {
    let graph = build_relu_sigmoid_graph();
    assert_eq!(graph.nodes.len(), 3);

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "Input + PointwiseFuse");
    assert!(matches!(opt.nodes[0].op, Op::Input));
    match &opt.nodes[1].op {
        Op::Custom { data } => {
            let pf = data
                .downcast_ref::<PointwiseFuse>()
                .expect("expected PointwiseFuse");
            assert_eq!(pf.members.len(), 2);
            assert!(matches!(pf.members[0], Op::Relu));
            assert!(matches!(pf.members[1], Op::Sigmoid));
        }
        other => panic!("expected Custom(PointwiseFuse), got {other:?}"),
    }
}

#[cfg(feature = "cuda")]
fn load_fixture(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(feature = "cuda")]
struct ExecKernel<'a>(&'a dyn ExecutableOp);

#[cfg(feature = "cuda")]
impl Kernel for ExecKernel<'_> {
    type Args<'b> = ();

    fn name(&self) -> &str {
        self.0.name()
    }

    fn source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn kernel_source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn entry_point_source(&self) -> &str {
        ""
    }

    fn entry_point_name(&self) -> String {
        self.0.forward_kernel_entry_point().to_string()
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_pointwise_relu_sigmoid_mlir() -> Result<()> {
    dotenv().ok();

    let graph = build_relu_sigmoid_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let (dag, _) = lowering
        .lower_with_mapping(&graph, LoweringMode::Inference)
        .expect("Anduin + PointwiseFuse lowering");

    assert_eq!(dag.len(), 2);
    let exec = dag.node(1).value.as_ref();
    assert!(
        exec.name().starts_with("pointwise_fuse_"),
        "got {}",
        exec.name()
    );

    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&ExecKernel(exec), &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("anduin_pointwise_relu_sigmoid_mlir", mlir.trim());
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_pointwise_relu_sigmoid_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_sigmoid_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);

    let teenyc_path = teeny_compiler::compiler::find_teenyc()?;
    let cache_dir =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
    let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let model = graph_compiler.compile_model(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
    )?;

    assert_eq!(model.dag.len(), 2);
    let loaded = model.load(&env.device, 1)?;

    let x = load_fixture("anduin_pointwise_relu_sigmoid/x.bin");
    let expected = load_fixture("anduin_pointwise_relu_sigmoid/expected_forward.bin");

    let x_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N) }?;
    // Batch dim dynamic in graph shape `[None, N]` → concrete `[1, N]`.
    let x_tensor = TensorRef::new(x_ptr, vec![1, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor])?;
    let mut y_out = vec![0.0f32; N];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    for i in 0..N {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "pointwise relu→sigmoid mismatch at {i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}
