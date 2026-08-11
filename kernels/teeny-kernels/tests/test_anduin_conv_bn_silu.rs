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

//! Anduin integration: `Input → Conv2d → BatchNorm2d → Silu`.
//!
//! Exercises `TritonLowering::with_optimizer(Anduin)` end-to-end: MLIR snapshots
//! for each lowered kernel, and CUDA numeric correctness vs PyTorch fixtures.
//!
//! Requires the `cuda` feature (GPU + `teenyc`).

use std::mem::size_of;
use std::path::PathBuf;
use std::rc::Rc;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_cuda::compiler::compile_cuda_graph;
use teeny_cuda::compiler::target::{Capability, Target};
use teeny_cuda::device::mem;
use teeny_cuda::errors::Result;
use teeny_cuda::model::TensorRef;
use teeny_cuda::testing;
use teeny_kernels::graph::{Anduin, TritonLowering};
use teeny_kernels::testing::load_fixture;

const NB: usize = 1;
const C_IN: usize = 2;
const C_OUT: usize = 4;
const HH: usize = 6;
const WW: usize = 6;
const KH: usize = 3;
const KW: usize = 3;
const STRIDE: usize = 1;
const PAD: usize = 1;
const OH: usize = (HH + 2 * PAD - KH) / STRIDE + 1; // 6
const OW: usize = (WW + 2 * PAD - KW) / STRIDE + 1; // 6
const EPS: f64 = 1e-5;
const TOL: f32 = 1e-4;

/// Build `Input → Conv2d(no bias) → BatchNorm2d → Silu`.
fn build_conv_bn_silu_graph() -> Graph {
    let (input, graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(NB), Some(C_IN), Some(HH), Some(WW)],
    );
    let out_shape = vec![Some(NB), Some(C_OUT), Some(OH), Some(OW)];

    let conv_id = input.graph.borrow_mut().add_node(
        Op::Conv2d {
            in_channels: C_IN,
            out_channels: C_OUT,
            kernel_h: KH,
            kernel_w: KW,
            stride_h: STRIDE,
            stride_w: STRIDE,
            padding_h: PAD,
            padding_w: PAD,
            groups: 1,
            has_bias: false,
        },
        vec![input.node_id],
        DtypeRepr::F32,
        out_shape.clone(),
    );

    let bn_id = input.graph.borrow_mut().add_node(
        Op::BatchNorm2d {
            num_features: C_OUT,
            eps: EPS,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        },
        vec![conv_id],
        DtypeRepr::F32,
        out_shape.clone(),
    );

    let _ = input
        .graph
        .borrow_mut()
        .add_node(Op::Silu, vec![bn_id], DtypeRepr::F32, out_shape);

    drop(input);
    Rc::try_unwrap(graph_rc).ok().unwrap().into_inner()
}

#[test]
fn test_anduin_conv_bn_silu_mlir() -> Result<()> {
    dotenv().ok();

    let graph = build_conv_bn_silu_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let target = Target::new(Capability::Sm89);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        true,
        false,
    )?;

    // Anduin is currently identity — expect Input + Conv2d + BN + Silu.
    assert_eq!(model.dag.len(), 4);
    assert!(model.dag.node(0).value.ptx_path.is_empty());

    let mut mlir_blobs = Vec::new();
    for i in 1..model.dag.len() {
        let node = &model.dag.node(i).value;
        let name = node.entry_point.trim_end_matches("_entry_point");
        let mlir = std::fs::read_to_string(PathBuf::from(&node.ptx_path).with_extension("mlir"))?;
        mlir_blobs.push(format!("=== {name} ===\n{}", mlir.trim()));
    }

    assert_debug_snapshot!("anduin_conv_bn_silu_mlir", mlir_blobs.join("\n\n"));
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_anduin_conv_bn_silu_matches_pytorch() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    let graph = build_conv_bn_silu_graph();
    let lowering = TritonLowering::new().with_optimizer(Anduin);
    let model = compile_cuda_graph(
        &graph,
        &lowering,
        &target,
        LoweringMode::Inference,
        true,
        true,
    )?;

    assert_eq!(model.dag.len(), 4, "Input + Conv2d + BN + Silu");

    let mut loaded = model.load(&env.device, NB)?;

    let x = load_fixture("anduin_conv_bn_silu/x.bin");
    let w = load_fixture("anduin_conv_bn_silu/w.bin");
    let bn_weight = load_fixture("anduin_conv_bn_silu/bn_weight.bin");
    let bn_bias = load_fixture("anduin_conv_bn_silu/bn_bias.bin");
    let bn_mean = load_fixture("anduin_conv_bn_silu/bn_running_mean.bin");
    let bn_var = load_fixture("anduin_conv_bn_silu/bn_running_var.bin");
    let expected = load_fixture("anduin_conv_bn_silu/expected_forward.bin");

    // DAG: 0=Input, 1=Conv2d(weight), 2=BatchNorm2d(weight,bias,mean,var), 3=Silu
    loaded.load_param_f32(1, 0, &w)?;
    loaded.load_param_f32(2, 0, &bn_weight)?;
    loaded.load_param_f32(2, 1, &bn_bias)?;
    loaded.load_param_f32(2, 2, &bn_mean)?;
    loaded.load_param_f32(2, 3, &bn_var)?;

    let n_in = NB * C_IN * HH * WW;
    let n_out = NB * C_OUT * OH * OW;
    let x_ptr = mem::alloc(n_in * size_of::<f32>())?;
    unsafe { mem::copy_h_to_d(x_ptr, x.as_ptr(), n_in) }?;
    let x_tensor = TensorRef::new(x_ptr, vec![NB, C_IN, HH, WW]);

    let output = loaded.forward(&env.device, NB, &[x_tensor])?;
    let mut y_out = vec![0.0f32; n_out];
    unsafe { mem::copy_d_to_h(y_out.as_mut_ptr(), output.ptr, n_out) }?;
    mem::free(output.ptr)?;
    mem::free(x_ptr)?;

    assert_eq!(y_out.len(), expected.len());
    for i in 0..n_out {
        assert!(
            (y_out[i] - expected[i]).abs() < TOL,
            "anduin conv→bn→silu mismatch at {i}: gpu={} expected={}",
            y_out[i],
            expected[i]
        );
    }

    Ok(())
}
