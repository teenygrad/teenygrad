// Elementwise arithmetic over operands of different shapes.
//
// The flat kernels read both operands at the output's own offsets, which is right only
// when all three shapes agree. The export leans on broadcasting heavily — a bias of
// [1024] added to [2, 129, 1024], a scalar divisor, a per-channel scale of
// [1, 112, 1, 1] — and read flat those are read past their ends, which segfaults rather
// than answering wrongly. That is what stopped the first attempt at running the model.
//
//   cargo test -p teeny-kernels --test test_broadcast_binary
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_broadcast_binary

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// The shape ONNX broadcasts `a` and `b` to.
fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    (0..rank)
        .map(|i| {
            let at = |s: &[usize]| {
                let off = rank - s.len();
                if i < off { 1 } else { s[i - off] }
            };
            at(a).max(at(b))
        })
        .collect()
}

fn binary_graph(op: Op, a: &[usize], b: &[usize]) -> Graph {
    let dims = |s: &[usize]| -> Vec<Option<usize>> { s.iter().map(|&d| Some(d)).collect() };
    let mut g = Graph::new();
    let an = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims(a));
    let bn = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims(b));
    g.add_node(op, vec![an, bn], DtypeRepr::F32, dims(&broadcast(a, b)));
    g
}

fn lower(op: Op, a: &[usize], b: &[usize]) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&binary_graph(op, a, b), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn the_shapes_the_export_broadcasts_lower() {
    for (a, b) in [
        // A bias added along the last axis.
        (vec![2usize, 129, 1024], vec![1024usize]),
        // A scalar.
        (vec![2, 129, 4096], vec![]),
        (vec![2, 16, 129, 64], vec![1]),
        // A per-channel scale.
        (vec![2, 112, 6, 8], vec![1, 112, 1, 1]),
    ] {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
            let name = format!("{op:?} {a:?} {b:?}");
            assert_eq!(
                lower(op, &a, &b).unwrap_or_else(|e| panic!("{name}: {e}")),
                3,
                "{name}"
            );
        }
    }
}

/// Equal shapes still take the flat kernels, which are cheaper: no table, no
/// per-lane coordinate arithmetic.
#[test]
fn equal_shapes_do_not_go_through_the_broadcast_kernel() {
    let graph = binary_graph(Op::Add, &[2, 3], &[2, 3]);
    let dag = TritonLowering::new()
        .lower(&graph, LoweringMode::Inference)
        .expect("equal shapes");
    let names: Vec<String> = (0..dag.len())
        .map(|i| dag.node(i).value.as_ref().name().to_string())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("elemwise_add")),
        "expected the flat kernel, got {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains("broadcast_add")),
        "expected no broadcast kernel, got {names:?}"
    );
}

/// Extents that are neither equal nor 1 are not a broadcast, and are refused rather
/// than read at some stride that happens to fit.
#[test]
fn shapes_that_do_not_broadcast_are_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(5)]);
    g.add_node(Op::Add, vec![a, b], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("[2, 5] against [2, 3] should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("do not broadcast"), "{msg}");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware", any(feature = "cuda", feature = "riscv")))]
mod execution {
    use super::*;
    use teeny_runtime::Target;
    use teeny_test::model::RuntimeModel;

    #[cfg(feature = "cuda")]
    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        let env = teeny_test::cuda::setup_cuda_env().expect("cuda env");
        let target = Target::new(env.capability);
        (env.device, target)
    }

    #[cfg(all(feature = "riscv", not(feature = "cuda")))]
    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        use teeny_riscv::compiler::target::Capability;
        (
            teeny_runtime::Device::new(Default::default()),
            Target::new(Capability::SpacemitK3),
        )
    }

    /// The operation on the host, by walking the output's coordinates and reading each
    /// operand at its own, with a length-1 axis read at 0. Written from ONNX's rule
    /// rather than from the kernel's strides.
    fn reference(op: &Op, x: &[f32], a: &[usize], y: &[f32], b: &[usize]) -> Vec<f32> {
        let out = broadcast(a, b);
        let rank = out.len();
        let pad = |s: &[usize]| -> Vec<usize> {
            let mut v = vec![1usize; rank - s.len()];
            v.extend_from_slice(s);
            v
        };
        let (a, b) = (pad(a), pad(b));
        let n: usize = out.iter().product();
        let mut want = vec![0.0f32; n];
        for flat in 0..n {
            let mut rest = flat;
            let mut coord = vec![0usize; rank];
            for ax in (0..rank).rev() {
                coord[ax] = rest % out[ax];
                rest /= out[ax];
            }
            let index = |shape: &[usize]| -> usize {
                let mut i = 0usize;
                for ax in 0..rank {
                    let c = if shape[ax] == 1 { 0 } else { coord[ax] };
                    i = i * shape[ax] + c;
                }
                i
            };
            let (l, r) = (x[index(&a)], y[index(&b)]);
            want[flat] = match op {
                Op::Add => l + r,
                Op::Sub => l - r,
                Op::Mul => l * r,
                _ => l / r,
            };
        }
        want
    }

    fn run(op: Op, a: &[usize], b: &[usize]) {
        let na: usize = a.iter().product::<usize>().max(1);
        let nb: usize = b.iter().product::<usize>().max(1);
        // Distinct, non-round values, and a divisor that never approaches zero.
        let x: Vec<f32> = (0..na).map(|i| i as f32 * 0.25 - 1.5).collect();
        let y: Vec<f32> = (0..nb).map(|i| i as f32 * 0.5 + 2.0).collect();
        let want = reference(&op, &x, a, &y, b);

        let graph = binary_graph(op, a, b);
        let (device, target) = device_and_target();
        let mut model = RuntimeModel::compile(
            &device,
            &graph,
            &TritonLowering::new(),
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
        .expect("compile failed");

        let feeds: Vec<&[f32]> = vec![&x, &y];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("no output of length {}", want.len()));
        for i in 0..want.len() {
            let d = (out[i] - want[i]).abs() / want[i].abs().max(1.0);
            assert!(
                d < 1e-6,
                "{a:?} {b:?}: element {i} is {} not {}",
                out[i],
                want[i]
            );
        }
    }

    /// A bias along the last axis, which is the export's commonest broadcast.
    #[test]
    fn a_bias_along_the_last_axis() {
        run(Op::Add, &[2, 3, 4], &[4]);
        run(Op::Mul, &[2, 3, 4], &[4]);
    }

    /// A scalar on either side, including the subtraction and division where the order
    /// matters.
    #[test]
    fn a_scalar_on_either_side() {
        run(Op::Div, &[2, 3, 4], &[]);
        run(Op::Sub, &[], &[2, 3, 4]);
        run(Op::Sub, &[2, 3, 4], &[1]);
    }

    /// A per-channel scale, where the broadcast axes are in the middle of the shape.
    #[test]
    fn a_per_channel_scale() {
        run(Op::Mul, &[2, 3, 4, 5], &[1, 3, 1, 1]);
        run(Op::Add, &[2, 3, 4, 5], &[1, 3, 1, 1]);
    }

    /// Both operands short on different axes, which only the general strides handle:
    /// neither operand is simply the output's shape.
    #[test]
    fn both_operands_broadcast_at_once() {
        run(Op::Add, &[4, 1, 3], &[1, 5, 3]);
        run(Op::Mul, &[2, 1], &[1, 6]);
    }

    /// More elements than one block of lanes.
    #[test]
    fn an_output_longer_than_a_block() {
        run(Op::Add, &[3, 500], &[500]);
    }
}
