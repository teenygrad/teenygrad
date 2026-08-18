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

//! Anduin fan-in fusion: `relu(x) + z` → [`TileFuse`] (teenygrad-3w0.3).
//!
//! This is the fusion shape `PointwiseFuse` cannot reach on its own: `Add`
//! has two distinct graph-node inputs, so it never passes
//! `fuse_pointwise_chain_pass`'s `inputs.len() != 1` guard. `TileFuse` runs
//! one input through a unary chain and combines it with the second,
//! unchained input via a binary tail op — teenygrad-1bf's fusion case 2.

use std::rc::Rc;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_kernels::graph::{Anduin, GraphOptimizer, TileFuse};

#[cfg(feature = "cuda")]
use dotenv::dotenv;
#[cfg(feature = "cuda")]
use std::mem::size_of;
#[cfg(feature = "cuda")]
use teeny_core::model::LoweringMode;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::compile_cuda_graph;
#[cfg(feature = "cuda")]
use teeny_cuda::compiler::target::Target;
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
const TOL: f32 = 1e-4;

fn shape_1d(n: usize) -> Vec<Option<usize>> {
    vec![None, Some(n)]
}

/// `y = relu(x) + z` — x is `Input` node 0, z is `Input` node 2 (added after
/// `relu` so `x`'s subtree keeps the lower indices `fuse_fan_in_pass`/
/// `forward`'s topological walk assign the first caller-provided tensor to).
fn build_relu_add_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_1d(N));
    let z = x
        .graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, shape_1d(N));
    let _ = x
        .graph
        .borrow_mut()
        .add_node(Op::Add, vec![relu, z], DtypeRepr::F32, shape_1d(N));
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = x + z` (no unary chain on either side) — must not fuse into
/// `TileFuse`: `branch` would be empty, which `TileFuse::new` forbids.
fn build_plain_add_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let z = x
        .graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, shape_1d(N));
    let _ = x
        .graph
        .borrow_mut()
        .add_node(Op::Add, vec![x.node_id, z], DtypeRepr::F32, shape_1d(N));
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

/// `y = relu(x) + z`, but `relu(x)` also feeds a second consumer (`sigmoid`)
/// — must not fuse: the branch input isn't single-consumer, so folding it
/// into `TileFuse` would silently drop the other consumer's read.
fn build_relu_add_with_extra_consumer_graph() -> Graph {
    let (x, graph_rc) = SymTensor::input(DtypeRepr::F32, shape_1d(N));
    let relu =
        x.graph
            .borrow_mut()
            .add_node(Op::Relu, vec![x.node_id], DtypeRepr::F32, shape_1d(N));
    let z = x
        .graph
        .borrow_mut()
        .add_node(Op::Input, vec![], DtypeRepr::F32, shape_1d(N));
    let _add = x
        .graph
        .borrow_mut()
        .add_node(Op::Add, vec![relu, z], DtypeRepr::F32, shape_1d(N));
    let _ = x
        .graph
        .borrow_mut()
        .add_node(Op::Sigmoid, vec![relu], DtypeRepr::F32, shape_1d(N));
    drop(x);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_anduin_fuses_relu_add_into_tile_fuse() {
    let graph = build_relu_add_graph();
    assert_eq!(graph.nodes.len(), 4, "x, relu, z, add");

    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(opt.nodes.len(), 3, "x Input, z Input, TileFuse");

    let fused_idx = opt.nodes.len() - 1;
    match &opt.nodes[fused_idx].op {
        Op::Custom { data } => {
            let tf = data.downcast_ref::<TileFuse>().expect("expected TileFuse");
            assert_eq!(tf.branch.len(), 1);
            assert!(matches!(tf.branch[0], Op::Relu));
            assert!(matches!(tf.tail, Op::Add));
        }
        other => panic!("expected Custom(TileFuse), got {other:?}"),
    }
    // Both original inputs must still be reachable as the fused node's inputs
    // (order: branch's root input, then the unchained second input).
    assert_eq!(opt.nodes[fused_idx].inputs.len(), 2);
}

#[test]
fn test_anduin_does_not_fuse_plain_add() {
    let graph = build_plain_add_graph();
    assert_eq!(graph.nodes.len(), 3, "x, z, add");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(
        opt.nodes.len(),
        3,
        "no fusion: no unary branch on either side"
    );
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<TileFuse>().is_none());
        }
    }
}

#[test]
fn test_anduin_does_not_fuse_relu_add_with_extra_consumer() {
    let graph = build_relu_add_with_extra_consumer_graph();
    assert_eq!(graph.nodes.len(), 5, "x, relu, z, add, sigmoid");
    let opt = Anduin.optimize(&graph).unwrap();
    assert_eq!(
        opt.nodes.len(),
        graph.nodes.len(),
        "expected no TileFuse fusion (relu has 2 consumers), got {opt:?}"
    );
    for node in &opt.nodes {
        if let Op::Custom { data } = &node.op {
            assert!(data.downcast_ref::<TileFuse>().is_none());
        }
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_tile_fuse_relu_add_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_relu_add_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        false,
        false,
    )?;

    assert_eq!(model.dag.len(), 3, "x Input, z Input, TileFuse");
    let loaded = model.load(&env.device, 1)?;

    let x = load_fixture("anduin_tile_fuse_relu_add/x.bin");
    let z = load_fixture("anduin_tile_fuse_relu_add/z.bin");
    let expected = load_fixture("anduin_tile_fuse_relu_add/expected_forward.bin");

    let x_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), N) }?;
    let z_ptr = mem::alloc(N * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(z_ptr, z.as_ptr(), N) }?;
    // Batch dim dynamic in graph shape `[None, N]` → concrete `[1, N]`.
    let x_tensor = TensorRef::new(x_ptr, vec![1, N]);
    let z_tensor = TensorRef::new(z_ptr, vec![1, N]);

    let output = loaded.forward(&env.device, 1, &[x_tensor, z_tensor])?;
    let mut y_out = vec![0.0f32; N];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, N) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;
    mem::free(z_ptr)?;

    for i in 0..N {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "mismatch at {i}: got {}, expected {}",
            y_out[i],
            expected[i]
        );
    }
    Ok(())
}
