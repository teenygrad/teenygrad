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

//! Anduin pointwise fusion: unary activation chains → [`PointwiseFuse`].

use std::rc::Rc;

use teeny_core::graph::{CustomOp, DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::{Anduin, GraphOptimizer, PointwiseFuse, TritonLowering};

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use insta::assert_debug_snapshot;
#[cfg(feature = "cuda")]
use std::mem::size_of;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use teeny_core::device::program::ArgVisitor;
#[cfg(feature = "cuda")]
use teeny_core::device::program::Kernel;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::compile_cuda_graph;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::compile_kernel;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::target::{Capability, Target};
#[cfg(feature = "cuda")]
use teeny_cuda::device::CudaArgPacker;
#[cfg(feature = "cuda")]
use teeny_cuda::device::mem;
#[cfg(feature = "cuda")]
use teeny_cuda::device::program::ErasedKernel;
#[cfg(feature = "cuda")]
use teeny_cuda::errors::Result;
#[cfg(feature = "cuda")]
use teeny_cuda::model::TensorRef;
#[cfg(feature = "cuda")]
use teeny_cuda::testing;
#[cfg(feature = "cuda")]
use teeny_kernels::testing::load_fixture;

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
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

fn build_relu_sigmoid_tanh_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu = input.graph.borrow_mut().add_node(
        Op::Relu,
        vec![input.node_id],
        DtypeRepr::F32,
        shape_1d(N),
    );
    let sigmoid =
        input
            .graph
            .borrow_mut()
            .add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape_1d(N));
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Tanh, vec![sigmoid], DtypeRepr::F32, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// Two `Neg` nodes chained, with each node's own dtype set independently
/// (rather than both following the input) so mixed/unsupported-dtype
/// scenarios can be built without a real dtype-changing op.
fn build_neg_neg_graph(dtype0: DtypeRepr, dtype1: DtypeRepr) -> Graph {
    let (input, graph_rc) = SymTensor::input(dtype0, shape_1d(N));
    let neg0 = input
        .graph
        .borrow_mut()
        .add_node(Op::Neg, vec![input.node_id], dtype0, shape_1d(N));
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Neg, vec![neg0], dtype1, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

fn build_isnan_isnan_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let isnan0 = input.graph.borrow_mut().add_node(
        Op::IsNaN,
        vec![input.node_id],
        DtypeRepr::Bool,
        shape_1d(N),
    );
    let _ =
        input
            .graph
            .borrow_mut()
            .add_node(Op::IsNaN, vec![isnan0], DtypeRepr::Bool, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

fn assert_no_fusion(graph: &Graph) {
    let opt = Anduin.optimize(graph).unwrap();
    assert_eq!(
        opt.nodes.len(),
        graph.nodes.len(),
        "expected no fusion, got {opt:?}"
    );
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(
                data.downcast_ref::<PointwiseFuse>().is_none(),
                "expected no PointwiseFuse, got {node:?}"
            );
        }
    }
}

#[test]
fn test_anduin_does_not_fuse_mixed_dtype_chain() {
    // Neg(f32) -> Neg(f64): parent/child disagree on dtype.
    let graph = build_neg_neg_graph(DtypeRepr::F32, DtypeRepr::F64);
    assert_eq!(graph.nodes.len(), 3);
    assert_no_fusion(&graph);
}

#[test]
fn test_anduin_does_not_fuse_int_dtype_chain() {
    // Neg(i32) -> Neg(i32): dtype-consistent but not a dtype PointwiseFuse
    // can lower (dtype_name only accepts f32/f64) -- must not fuse, and
    // must not panic (PointwiseFuse::lower is never reached).
    let graph = build_neg_neg_graph(DtypeRepr::I32, DtypeRepr::I32);
    assert_eq!(graph.nodes.len(), 3);
    assert_no_fusion(&graph);
}

#[test]
fn test_anduin_does_not_fuse_isnan_mid_chain() {
    // IsNaN -> IsNaN: the first IsNaN would be an interior member if this
    // fused; bool-producing ops may only ever be a chain's terminal member.
    let graph = build_isnan_isnan_graph();
    assert_eq!(graph.nodes.len(), 3);
    assert_no_fusion(&graph);
}

fn build_abs_neg_sign_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let abs = input.graph.borrow_mut().add_node(
        Op::Abs,
        vec![input.node_id],
        DtypeRepr::F32,
        shape_1d(N),
    );
    let neg = input
        .graph
        .borrow_mut()
        .add_node(Op::Neg, vec![abs], DtypeRepr::F32, shape_1d(N));
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Sign, vec![neg], DtypeRepr::F32, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

fn build_abs_neg_sign_ceil_floor_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let abs = input.graph.borrow_mut().add_node(
        Op::Abs,
        vec![input.node_id],
        DtypeRepr::F32,
        shape_1d(N),
    );
    let neg = input
        .graph
        .borrow_mut()
        .add_node(Op::Neg, vec![abs], DtypeRepr::F32, shape_1d(N));
    let sign = input
        .graph
        .borrow_mut()
        .add_node(Op::Sign, vec![neg], DtypeRepr::F32, shape_1d(N));
    let ceil = input
        .graph
        .borrow_mut()
        .add_node(Op::Ceil, vec![sign], DtypeRepr::F32, shape_1d(N));
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Floor, vec![ceil], DtypeRepr::F32, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `Op::Round` has no lowering yet (`kernels/teeny-kernels/src/graph/mod.rs`,
/// `Op::Round => Err("TODO: Op::Round ...")`), so it can never pass the
/// pointwise-fuse probe -- unrelated to this batch's dtype/bool-terminal
/// guards, just the underlying op not existing yet.
fn build_abs_round_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let abs = input.graph.borrow_mut().add_node(
        Op::Abs,
        vec![input.node_id],
        DtypeRepr::F32,
        shape_1d(N),
    );
    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Round, vec![abs], DtypeRepr::F32, shape_1d(N));
    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_anduin_fuses_abs_neg_sign_into_pointwise_fuse() {
    let graph = build_abs_neg_sign_graph();
    assert_eq!(graph.nodes.len(), 4);

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "Input + PointwiseFuse");
    match &opt.nodes[1].op {
        Op::Custom { data } => {
            let pf = data
                .downcast_ref::<PointwiseFuse>()
                .expect("expected PointwiseFuse");
            assert_eq!(pf.members.len(), 3);
            assert!(matches!(pf.members[0], Op::Abs));
            assert!(matches!(pf.members[1], Op::Neg));
            assert!(matches!(pf.members[2], Op::Sign));
            // None of Abs/Neg/Sign are y-style (dy, y, dx, n) -- this batch
            // is forward-fusion only, not folded into is_y_style_pointwise_bwd.
            assert!(!pf.supports_fused_backward());
        }
        other => panic!("expected Custom(PointwiseFuse), got {other:?}"),
    }
}

#[test]
fn test_anduin_fuses_abs_neg_sign_ceil_floor_into_pointwise_fuse() {
    let graph = build_abs_neg_sign_ceil_floor_graph();
    assert_eq!(graph.nodes.len(), 6);

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "Input + PointwiseFuse");
    match &opt.nodes[1].op {
        Op::Custom { data } => {
            let pf = data
                .downcast_ref::<PointwiseFuse>()
                .expect("expected PointwiseFuse");
            assert_eq!(pf.members.len(), 5);
            assert!(matches!(pf.members[0], Op::Abs));
            assert!(matches!(pf.members[1], Op::Neg));
            assert!(matches!(pf.members[2], Op::Sign));
            assert!(matches!(pf.members[3], Op::Ceil));
            assert!(matches!(pf.members[4], Op::Floor));
            assert!(!pf.supports_fused_backward());
        }
        other => panic!("expected Custom(PointwiseFuse), got {other:?}"),
    }
}

#[test]
fn test_anduin_round_does_not_fuse() {
    let graph = build_abs_round_graph();
    assert_eq!(graph.nodes.len(), 3);
    assert_no_fusion(&graph);
}

fn pointwise_fuse_from(graph: &Graph) -> PointwiseFuse {
    let opt = Anduin.optimize(graph).unwrap();
    match &opt.nodes.last().unwrap().op {
        Op::Custom { data } => data
            .downcast_ref::<PointwiseFuse>()
            .expect("expected PointwiseFuse")
            .clone(),
        other => panic!("expected Custom(PointwiseFuse), got {other:?}"),
    }
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
            assert!(pf.supports_fused_backward());
        }
        other => panic!("expected Custom(PointwiseFuse), got {other:?}"),
    }
}

#[test]
fn test_anduin_fuses_relu_sigmoid_tanh_into_pointwise_fuse() {
    let graph = build_relu_sigmoid_tanh_graph();
    assert_eq!(graph.nodes.len(), 4);

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 2, "Input + PointwiseFuse");
    match &opt.nodes[1].op {
        Op::Custom { data } => {
            let pf = data
                .downcast_ref::<PointwiseFuse>()
                .expect("expected PointwiseFuse");
            assert_eq!(pf.members.len(), 3);
            assert!(matches!(pf.members[0], Op::Relu));
            assert!(matches!(pf.members[1], Op::Sigmoid));
            assert!(matches!(pf.members[2], Op::Tanh));
            assert!(pf.supports_fused_backward());
            #[cfg(feature = "training")]
            {
                let bwd = pf.lower_backward_source();
                assert!(
                    !bwd.is_empty(),
                    "y-style 3-member chain should emit fused backward"
                );
                assert!(bwd.contains("relu_backward"));
                assert!(bwd.contains("sigmoid_backward"));
                assert!(bwd.contains("tanh_backward"));
            }
        }
        other => panic!("expected Custom(PointwiseFuse), got {other:?}"),
    }
}

/// Thin `Kernel` adapter so fused backward source can go through `compile_kernel`.
#[cfg(feature = "cuda")]
struct SourceKernel {
    name: String,
    source: String,
}

#[cfg(feature = "cuda")]
impl Kernel for SourceKernel {
    type Args<'a> = ();

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
        ""
    }
}

// Compiles the fused graph via `compile_cuda_graph(..., debug=true)` so teenyc's
// ttir/ttgpuir/llir/llvmir/ptx pipeline stages are logged to stderr. Run with
// `--nocapture` (and redirect stderr) to capture them, e.g.:
//
//   cargo test -p teeny-kernels --test test_anduin_pointwise --features cuda \
//     test_anduin_pointwise_relu_sigmoid_mlir -- --nocapture 2>pipeline.log
#[test]
#[cfg(feature = "cuda")]
fn test_anduin_pointwise_relu_sigmoid_mlir() -> Result<()> {
    dotenv().ok();

    let graph = build_relu_sigmoid_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let target = Target::new(Capability::Sm89);
    // force=true so teenyc actually runs (a cache hit would emit no pipeline logs).
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        true,
        true,
    )?;

    assert_eq!(model.dag.len(), 2);
    let node = &model.dag.node(1).value;
    assert!(
        node.entry_point.starts_with("pointwise_fuse_"),
        "got {}",
        node.entry_point
    );

    let mlir = std::fs::read_to_string(PathBuf::from(&node.ptx_path).with_extension("mlir"))?;
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
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
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

/// Case-1 rounding/sign batch (teenygrad-1bf.1.7): forward-only -- none of
/// Abs/Neg/Sign are y-style, so there is no fused backward to check here.
#[test]
#[cfg(feature = "cuda")]
fn test_anduin_pointwise_abs_neg_sign_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_abs_neg_sign_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
        false,
    )?;

    assert_eq!(model.dag.len(), 2);
    let loaded = model.load(&env.device, 1)?;

    let x = load_fixture("anduin_pointwise_abs_neg_sign/x.bin");
    let expected = load_fixture("anduin_pointwise_abs_neg_sign/expected_forward.bin");

    let x_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N) }?;
    let x_tensor = TensorRef::new(x_ptr, vec![1, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor])?;
    let mut y_out = vec![0.0f32; N];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    for i in 0..N {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "pointwise abs→neg→sign mismatch at {i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_pointwise_relu_sigmoid_tanh_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_sigmoid_tanh_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
        false,
    )?;

    assert_eq!(model.dag.len(), 2);
    let node = &model.dag.node(1).value;
    assert!(
        node.entry_point.contains("relu")
            && node.entry_point.contains("sigmoid")
            && node.entry_point.contains("tanh"),
        "unexpected fused entry {}",
        node.entry_point
    );

    let loaded = model.load(&env.device, 1)?;
    let x = load_fixture("anduin_pointwise_relu_sigmoid_tanh/x.bin");
    let expected = load_fixture("anduin_pointwise_relu_sigmoid_tanh/expected_forward.bin");

    let x_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N) }?;
    let x_tensor = TensorRef::new(x_ptr, vec![1, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor])?;
    let mut y_out = vec![0.0f32; N];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    for i in 0..N {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "pointwise relu→sigmoid→tanh mismatch at {i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "training"))]
fn test_anduin_pointwise_relu_sigmoid_backward_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_sigmoid_graph();
    let pf = pointwise_fuse_from(&graph);
    let bwd_src = pf.lower_backward_source();
    assert!(!bwd_src.is_empty());

    // Training compile must attach a backward PTX for the fused custom op.
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Training,
        false,
        false,
    )?;
    let node = &model.dag.node(1).value;
    assert!(
        node.backward_ptx_path.is_some(),
        "expected fused backward PTX"
    );
    assert_eq!(
        node.backward_entry_point,
        "pointwise_fuse_relu_sigmoid_backward_entry_point"
    );

    let kernel = SourceKernel {
        name: "pointwise_fuse_relu_sigmoid_backward".into(),
        source: bwd_src,
    };
    let ptx_path = compile_kernel(&kernel, &target, false, false)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<ErasedKernel>(&ptx)?;

    let y = load_fixture("anduin_pointwise_relu_sigmoid/y_backward.bin");
    let dy = load_fixture("anduin_pointwise_relu_sigmoid/dy.bin");
    let scratch0 = load_fixture("anduin_pointwise_relu_sigmoid/scratch0.bin");
    let expected = load_fixture("anduin_pointwise_relu_sigmoid/expected_backward.bin");

    let bytes = N * size_of::<f32>();
    let dy_ptr = mem::alloc(bytes)?;
    let y_ptr = mem::alloc(bytes)?;
    let dx_ptr = mem::alloc(bytes)?;
    let scratch_ptr = mem::alloc(bytes)?;
    unsafe {
        mem::copy_h_to_d(dy_ptr, dy.as_ptr(), N)?;
        mem::copy_h_to_d(y_ptr, y.as_ptr(), N)?;
        mem::copy_h_to_d(scratch_ptr, scratch0.as_ptr(), N)?;
    }

    let cfg = testing::launch_config_from_program(N, &program);
    let mut packer = CudaArgPacker::new();
    packer.visit_ptr(dy_ptr as *mut _);
    packer.visit_ptr(y_ptr as *mut _);
    packer.visit_ptr(dx_ptr as *mut _);
    packer.visit_ptr(scratch_ptr as *mut _);
    packer.visit_i32(N as i32);
    env.device.launch_with_packer(&program, &cfg, &mut packer)?;

    let mut dx_out = vec![0.0f32; N];
    unsafe { mem::copy_d_to_h(dx_out.as_mut_ptr(), dx_ptr, N) }?;
    mem::free(dy_ptr)?;
    mem::free(y_ptr)?;
    mem::free(dx_ptr)?;
    mem::free(scratch_ptr)?;

    for i in 0..N {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "pointwise relu→sigmoid backward mismatch at {i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "cuda", feature = "training"))]
fn test_anduin_pointwise_relu_sigmoid_tanh_backward_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_sigmoid_tanh_graph();
    let pf = pointwise_fuse_from(&graph);
    let bwd_src = pf.lower_backward_source();
    assert!(!bwd_src.is_empty());

    let kernel = SourceKernel {
        name: "pointwise_fuse_relu_sigmoid_tanh_backward".into(),
        source: bwd_src,
    };
    let ptx_path = compile_kernel(&kernel, &target, false, false)?;
    let ptx = std::fs::read(&ptx_path)?;
    let program = testing::load_program_from_ptx::<ErasedKernel>(&ptx)?;

    let y = load_fixture("anduin_pointwise_relu_sigmoid_tanh/y_backward.bin");
    let dy = load_fixture("anduin_pointwise_relu_sigmoid_tanh/dy.bin");
    let scratch0 = load_fixture("anduin_pointwise_relu_sigmoid_tanh/scratch0.bin");
    let scratch1 = load_fixture("anduin_pointwise_relu_sigmoid_tanh/scratch1.bin");
    let expected = load_fixture("anduin_pointwise_relu_sigmoid_tanh/expected_backward.bin");

    let bytes = N * size_of::<f32>();
    let dy_ptr = mem::alloc(bytes)?;
    let y_ptr = mem::alloc(bytes)?;
    let dx_ptr = mem::alloc(bytes)?;
    let s0_ptr = mem::alloc(bytes)?;
    let s1_ptr = mem::alloc(bytes)?;
    unsafe {
        mem::copy_h_to_d(dy_ptr, dy.as_ptr(), N)?;
        mem::copy_h_to_d(y_ptr, y.as_ptr(), N)?;
        mem::copy_h_to_d(s0_ptr, scratch0.as_ptr(), N)?;
        mem::copy_h_to_d(s1_ptr, scratch1.as_ptr(), N)?;
    }

    let cfg = testing::launch_config_from_program(N, &program);
    let mut packer = CudaArgPacker::new();
    packer.visit_ptr(dy_ptr as *mut _);
    packer.visit_ptr(y_ptr as *mut _);
    packer.visit_ptr(dx_ptr as *mut _);
    packer.visit_ptr(s0_ptr as *mut _);
    packer.visit_ptr(s1_ptr as *mut _);
    packer.visit_i32(N as i32);
    env.device.launch_with_packer(&program, &cfg, &mut packer)?;

    let mut dx_out = vec![0.0f32; N];
    unsafe { mem::copy_d_to_h(dx_out.as_mut_ptr(), dx_ptr, N) }?;
    mem::free(dy_ptr)?;
    mem::free(y_ptr)?;
    mem::free(dx_ptr)?;
    mem::free(s0_ptr)?;
    mem::free(s1_ptr)?;

    for i in 0..N {
        assert!(
            (dx_out[i] - expected[i]).abs() < TOL,
            "pointwise relu→sigmoid→tanh backward mismatch at {i}: gpu={} expected={}",
            dx_out[i],
            expected[i]
        );
    }
    Ok(())
}
