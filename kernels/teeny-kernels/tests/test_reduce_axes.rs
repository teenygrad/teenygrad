//! Reductions over an axis that is not the innermost one.
//!
//! The reduction kernels walk a flattened `[outer, inner]` view and infer the split
//! from the output shape, so they can only reduce the trailing axes. The lowering puts
//! a transpose in front when the axes are elsewhere; without it a reduce summed
//! adjacent elements instead of strided ones and returned a plausibly-shaped wrong
//! answer, which is what made FoundationStereo's feature normalization produce 1.5e13
//! where nothing can exceed 1.

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

const SHAPE: [usize; 4] = [2, 3, 4, 5];

fn graph_reducing(axis: i64, keepdims: bool) -> Graph {
    let mut g = Graph::new();
    let x = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        SHAPE.iter().map(|d| Some(*d)).collect(),
    );
    let out_shape: Vec<Option<usize>> = {
        let ax = axis as usize;
        if keepdims {
            SHAPE
                .iter()
                .enumerate()
                .map(|(i, d)| Some(if i == ax { 1 } else { *d }))
                .collect()
        } else {
            SHAPE
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, d)| Some(*d))
                .collect()
        }
    };
    g.add_node(
        Op::ReduceSum {
            axes: vec![axis],
            keepdims,
            noop_with_empty_axes: false,
        },
        vec![x],
        DtypeRepr::F32,
        out_shape,
    );
    g
}

/// The innermost axis needs no transpose; anything else gets one.
#[test]
fn test_only_a_non_trailing_axis_gains_a_transpose() {
    for (axis, extra) in [(3i64, 0usize), (2, 1), (1, 1), (0, 1)] {
        let dag = TritonLowering::new()
            .lower(&graph_reducing(axis, true), LoweringMode::Inference)
            .expect("lower");
        // Input, reduce, and the transpose when one is needed.
        assert_eq!(
            dag.len(),
            2 + extra,
            "axis {axis}: expected {} DAG nodes, got {}",
            2 + extra,
            dag.len()
        );
    }
}

#[cfg(feature = "hardware")]
mod execution {
    use super::*;
    use teeny_runtime::Target;
    use teeny_test::model::RuntimeModel;

    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        #[cfg(feature = "cuda")]
        {
            let device = teeny_runtime::open().expect("a CUDA device");
            let target = teeny_runtime::default_target(&device).expect("target");
            (device, target)
        }
        #[cfg(not(feature = "cuda"))]
        {
            use teeny_riscv::compiler::target::Capability;
            (
                teeny_runtime::Device::new(Default::default()),
                Target::new(Capability::GenericRvv1_0),
            )
        }
    }

    /// Values that make a wrong grouping visible: distinct, and not symmetric about any
    /// axis, so summing the wrong 4 of them cannot land on the right answer.
    fn values() -> Vec<f32> {
        let n: usize = SHAPE.iter().product();
        (0..n).map(|i| (i as f32) * 0.125 - 7.0).collect()
    }

    /// The sum over `axis`, computed on the host from the same strides.
    fn reference(x: &[f32], axis: usize) -> Vec<f32> {
        let inner: usize = SHAPE[axis + 1..].iter().product();
        let extent = SHAPE[axis];
        let outer: usize = SHAPE[..axis].iter().product();
        let mut out = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut sum = 0.0f32;
                for k in 0..extent {
                    sum += x[o * extent * inner + k * inner + i];
                }
                out[o * inner + i] = sum;
            }
        }
        out
    }

    #[test]
    fn test_a_reduction_over_every_axis_matches_the_host() {
        let (device, target) = device_and_target();
        let x = values();

        for axis in 0..SHAPE.len() {
            for keepdims in [true, false] {
                let graph = graph_reducing(axis as i64, keepdims);
                let mut model = RuntimeModel::compile(
                    &device,
                    &graph,
                    &TritonLowering::new(),
                    LoweringMode::Inference,
                    &target,
                    1,
                    false,
                )
                .expect("compile");

                let outputs = model.terminal_node_indices_sorted_by_size();
                let got = model.run(&device, &[&x], &outputs).expect("run");
                let want = reference(&x, axis);
                let out = got
                    .iter()
                    .find(|o| o.len() == want.len())
                    .unwrap_or_else(|| panic!("axis {axis}: no output of {} values", want.len()));

                for (i, (g, w)) in out.iter().zip(&want).enumerate() {
                    // The sums are of 2 to 5 terms of order 10, so a float comparison
                    // is tight here.
                    assert!(
                        (g - w).abs() < 1e-3,
                        "axis {axis}, keepdims {keepdims}, element {i}: got {g}, want {w}"
                    );
                }
            }
        }
    }
}
