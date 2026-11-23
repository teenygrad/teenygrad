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

/// A reduction of `input` over `axes` (keepdims), recording the shape that gives.
fn reduce_sum_graph(input: &[Option<usize>], axes: &[i64]) -> Graph {
    let rank = input.len() as i64;
    let output = (0..input.len())
        .map(|a| {
            let reduced = axes.is_empty() || axes.iter().any(|&r| (r + rank) % rank == a as i64);
            if reduced { Some(1) } else { input[a] }
        })
        .collect();
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], DtypeRepr::F32, input.to_vec());
    g.add_node(
        Op::ReduceSum {
            axes: axes.to_vec(),
            keepdims: true,
            noop_with_empty_axes: false,
        },
        vec![x],
        DtypeRepr::F32,
        output,
    );
    g
}

fn lowering_error(g: &Graph) -> Option<String> {
    TritonLowering::new()
        .lower(g, LoweringMode::Inference)
        .err()
        .map(|e| e.to_string())
}

/// The kernels reduce one block of 1024 elements per output and drop the rest, so a
/// longer row is refused rather than truncated.
#[test]
fn a_row_longer_than_the_kernel_block_is_refused() {
    let known = |d: &[usize]| d.iter().map(|&x| Some(x)).collect::<Vec<_>>();
    assert_eq!(
        lowering_error(&reduce_sum_graph(&known(&[2, 1024]), &[-1])),
        None
    );
    for (shape, axes) in [
        (known(&[2, 1025]), vec![-1]),
        // Reducing every axis of an image, which a global average pool exported as
        // ReduceMean does.
        (known(&[1, 3, 224, 224]), vec![]),
        // Two axes that fit alone but not together.
        (known(&[40, 40, 3]), vec![0, 1]),
    ] {
        let err = lowering_error(&reduce_sum_graph(&shape, &axes))
            .unwrap_or_else(|| panic!("{shape:?} over {axes:?} lowered"));
        assert!(err.contains("at most 1024"), "{err}");
    }
}

/// A dynamic axis that stays is fine; a dynamic one that is reduced cannot be checked
/// against the block, so it is refused.
#[test]
fn only_a_reduced_axis_needs_a_known_extent() {
    let g = reduce_sum_graph(&[None, Some(4), Some(10)], &[-1]);
    assert_eq!(lowering_error(&g), None);
    let g = reduce_sum_graph(&[Some(2), Some(4), None], &[-1]);
    let err = lowering_error(&g).expect("an unknown reduced extent");
    assert!(err.contains("unknown extent"), "{err}");
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

    // ── Every reduction against PyTorch (fixtures from reduce_axes/generate.py) ──

    /// As in `generate.py`.
    const AXES: &[&[i64]] = &[&[0], &[1], &[2], &[0, 2], &[1, 3], &[0, 1, 2], &[0, 3]];
    const RTOL: f32 = 1e-4;
    const ATOL: f32 = 1e-5;

    type MakeOp = fn(Vec<i64>, bool) -> Op;

    macro_rules! reduce_op {
        ($v:ident) => {
            |axes, keepdims| Op::$v {
                axes,
                keepdims,
                noop_with_empty_axes: false,
            }
        };
    }

    const OPS: &[(&str, MakeOp)] = &[
        ("reduce_sum", reduce_op!(ReduceSum)),
        ("reduce_mean", reduce_op!(ReduceMean)),
        ("reduce_max", reduce_op!(ReduceMax)),
        ("reduce_min", reduce_op!(ReduceMin)),
        ("reduce_prod", reduce_op!(ReduceProd)),
        ("reduce_l1", reduce_op!(ReduceL1)),
        ("reduce_l2", reduce_op!(ReduceL2)),
        ("reduce_log_sum", reduce_op!(ReduceLogSum)),
        ("reduce_log_sum_exp", reduce_op!(ReduceLogSumExp)),
        ("reduce_sum_square", reduce_op!(ReduceSumSquare)),
    ];

    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= ATOL + RTOL * want.abs()
    }

    /// Compiles `op` over an input of `input` and runs it on `x`.
    fn run_reduce(
        device: &teeny_runtime::Device<'_>,
        target: &Target,
        op: Op,
        input: &[usize],
        axes: &[i64],
        keepdims: bool,
        x: &[f32],
    ) -> Vec<f32> {
        let rank = input.len() as i64;
        let reduced = |a: usize| axes.iter().any(|&r| (r + rank) % rank == a as i64);
        let output: Vec<Option<usize>> = (0..input.len())
            .filter_map(|a| match (reduced(a), keepdims) {
                (true, true) => Some(Some(1)),
                (true, false) => None,
                (false, _) => Some(Some(input[a])),
            })
            .collect();
        let mut g = Graph::new();
        let xi = g.add_node(
            Op::Input,
            vec![],
            DtypeRepr::F32,
            input.iter().map(|d| Some(*d)).collect(),
        );
        g.add_node(op, vec![xi], DtypeRepr::F32, output);
        let mut model = RuntimeModel::compile(
            device,
            &g,
            &TritonLowering::new(),
            LoweringMode::Inference,
            target,
            1,
            false,
        )
        .expect("compile");
        let outputs = model.terminal_node_indices_sorted_by_size();
        model
            .run(device, &[x], &outputs)
            .expect("run")
            .pop()
            .expect("an output")
    }

    /// Every reduction over every axis set, with and without `keepdims`, matches
    /// PyTorch. `keepdims` changes only the recorded shape, so both are checked
    /// against the same values.
    #[test]
    fn every_reduction_over_any_axes_matches_torch() {
        dotenv::dotenv().ok();
        let (device, target) = device_and_target();
        let dir = env!("CARGO_MANIFEST_DIR");
        let x = teeny_test::load_fixture(dir, "reduce_axes/x.bin");
        let mut failures = Vec::new();
        for &(name, make) in OPS {
            let expected =
                teeny_test::load_fixture(dir, &format!("reduce_axes/expected_{name}.bin"));
            let mut offset = 0;
            for &axes in AXES {
                let n_out: usize = (0..SHAPE.len())
                    .filter(|a| !axes.contains(&(*a as i64)))
                    .map(|a| SHAPE[a])
                    .product();
                let want = &expected[offset..offset + n_out];
                offset += n_out;
                for keepdims in [true, false] {
                    let op = make(axes.to_vec(), keepdims);
                    let got = run_reduce(&device, &target, op, &SHAPE, axes, keepdims, &x);
                    if got.len() != n_out {
                        failures.push(format!(
                            "{name} axes {axes:?} keepdims {keepdims}: {} values, want {n_out}",
                            got.len()
                        ));
                    } else if let Some(i) = (0..n_out).find(|&i| !close(got[i], want[i])) {
                        failures.push(format!(
                            "{name} axes {axes:?} keepdims {keepdims}: [{i}] got {}, want {}",
                            got[i], want[i]
                        ));
                    }
                }
            }
            assert_eq!(offset, expected.len(), "{name}: fixture size");
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// teenygrad#151's repro: an L2 norm over axis 2 of `[1, 2, 5, 3, 4]`, the layout
    /// of FoundationStereo's feature normalization.
    #[test]
    fn l2norm5d_matches_torch() {
        dotenv::dotenv().ok();
        let (device, target) = device_and_target();
        let dir = env!("CARGO_MANIFEST_DIR");
        let x = teeny_test::load_fixture(dir, "reduce_axes/x_l2norm5d.bin");
        let want = teeny_test::load_fixture(dir, "reduce_axes/expected_l2norm5d.bin");
        let op = Op::ReduceL2 {
            axes: vec![2],
            keepdims: true,
            noop_with_empty_axes: false,
        };
        let got = run_reduce(&device, &target, op, &[1, 2, 5, 3, 4], &[2], true, &x);
        assert_eq!(got.len(), want.len());
        for i in 0..want.len() {
            assert!(
                close(got[i], want[i]),
                "[{i}] got {}, want {}",
                got[i],
                want[i]
            );
        }
    }
}
