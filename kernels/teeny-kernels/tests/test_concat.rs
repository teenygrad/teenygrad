// Concat along any axis of any rank.
//
// Concatenation is a contiguous copy per leading block, so the kernel is the one
// channel_cat already uses; what is new is working out its arguments for an arbitrary
// axis, and refusing the operand shapes that would copy into the wrong place.
//
//   cargo test -p teeny-kernels --test test_concat
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_concat

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// The output shape of concatenating `shapes` along `axis`.
fn out_shape(shapes: &[Vec<usize>], axis: usize) -> Vec<usize> {
    let mut out = shapes[0].clone();
    out[axis] = shapes.iter().map(|s| s[axis]).sum();
    out
}

fn concat_graph(shapes: &[Vec<usize>], axis: i64) -> Graph {
    let rank = shapes[0].len();
    let norm = if axis < 0 { axis + rank as i64 } else { axis } as usize;
    let mut g = Graph::new();
    let inputs: Vec<usize> = shapes
        .iter()
        .map(|s| {
            g.add_node(
                Op::Input,
                vec![],
                DtypeRepr::F32,
                s.iter().map(|&d| Some(d)).collect(),
            )
        })
        .collect();
    g.add_node(
        Op::Concat { axis },
        inputs,
        DtypeRepr::F32,
        out_shape(shapes, norm).iter().map(|&d| Some(d)).collect(),
    );
    g
}

fn lower(shapes: &[Vec<usize>], axis: i64) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&concat_graph(shapes, axis), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn test_every_axis_and_rank_lowers() {
    for (shapes, axis) in [
        (vec![vec![3usize], vec![2]], 0i64),
        (vec![vec![2, 3], vec![2, 4]], 1),
        (vec![vec![2, 3], vec![5, 3]], 0),
        (vec![vec![2, 3, 4], vec![2, 3, 1], vec![2, 3, 2]], 2),
        (vec![vec![2, 3, 4, 5], vec![2, 7, 4, 5]], 1),
        // A negative axis counts from the end.
        (vec![vec![2, 3], vec![2, 4]], -1),
    ] {
        let n = shapes.len();
        assert_eq!(
            lower(&shapes, axis).unwrap_or_else(|e| panic!("{shapes:?} axis {axis}: {e}")),
            n + 1
        );
    }
}

/// One input is a copy, and the single-launch path has to get the same grid the
/// multi-launch one would.
#[test]
fn test_a_single_operand_lowers() {
    assert_eq!(lower(&[vec![2, 3]], 1).expect("one operand"), 2);
}

#[test]
fn test_an_axis_outside_the_rank_is_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(4)]);
    g.add_node(
        Op::Concat { axis: 2 },
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(2), Some(7)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("axis 2 of a rank-2 output should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("outside a rank-2 output"), "{msg}");
}

/// An operand that differs on an axis other than the concatenated one would be copied
/// into the wrong place, element by element, with no error on the device.
#[test]
fn test_an_operand_that_differs_off_axis_is_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(4), Some(5)]);
    g.add_node(
        Op::Concat { axis: 1 },
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(2), Some(8)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a mismatched off-axis extent should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("on axis 0"), "{msg}");
}

/// Extents that do not add up to the output's leave part of it unwritten, or run past
/// its end.
#[test]
fn test_operands_that_do_not_add_up_are_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(4)]);
    g.add_node(
        Op::Concat { axis: 1 },
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(2), Some(9)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("extents that do not add up should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("add up to 7"), "{msg}");
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

    /// Concatenation on the host, by walking the output's coordinates.
    ///
    /// Independent of the kernel's block arithmetic: it finds which operand owns each
    /// output position and reads that operand by its own coordinates, so an offset
    /// mistake shows up as a mismatch rather than being reproduced on both sides.
    fn reference(inputs: &[Vec<f32>], shapes: &[Vec<usize>], axis: usize) -> Vec<f32> {
        let out = out_shape(shapes, axis);
        let n: usize = out.iter().product();
        let mut want = vec![0.0f32; n];
        for flat in 0..n {
            // Unflatten the output position.
            let mut rest = flat;
            let mut coords = vec![0usize; out.len()];
            for d in (0..out.len()).rev() {
                coords[d] = rest % out[d];
                rest /= out[d];
            }
            // Find the operand that owns this coordinate along the axis.
            let mut along = coords[axis];
            let mut which = 0usize;
            while along >= shapes[which][axis] {
                along -= shapes[which][axis];
                which += 1;
            }
            let mut src = 0usize;
            for d in 0..out.len() {
                let c = if d == axis { along } else { coords[d] };
                src = src * shapes[which][d] + c;
            }
            want[flat] = inputs[which][src];
        }
        want
    }

    fn run(shapes: &[Vec<usize>], axis: i64) {
        let rank = shapes[0].len();
        let norm = if axis < 0 { axis + rank as i64 } else { axis } as usize;

        // Distinct values per operand and per element, so a block copied from the
        // wrong operand or at the wrong offset is visible.
        let inputs: Vec<Vec<f32>> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n: usize = s.iter().product();
                (0..n).map(|e| (i * 1000 + e) as f32 * 0.25 - 1.5).collect()
            })
            .collect();
        let want = reference(&inputs, shapes, norm);

        let graph = concat_graph(shapes, axis);
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

        let feeds: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("{shapes:?}: no output of length {}", want.len()));
        for i in 0..want.len() {
            assert_eq!(
                out[i], want[i],
                "{shapes:?} axis {axis}: element {i} is {} not {}",
                out[i], want[i]
            );
        }
    }

    #[test]
    fn test_concatenating_vectors() {
        run(&[vec![3], vec![2], vec![4]], 0);
    }

    /// The innermost axis: every operand's block is one contiguous run per row, and
    /// the runs interleave in the output.
    #[test]
    fn test_concatenating_the_innermost_axis() {
        run(&[vec![3, 2], vec![3, 5]], 1);
    }

    /// The outermost axis: each operand is one unbroken run of the output, which is
    /// the case a wrong `inner` would still get right, so it is checked alongside the
    /// others rather than on its own.
    #[test]
    fn test_concatenating_the_outermost_axis() {
        run(&[vec![2, 4], vec![3, 4]], 0);
    }

    /// A middle axis of a rank-4 tensor, which is where both `outer` and `inner` are
    /// greater than one and an arithmetic mistake has nowhere to hide.
    #[test]
    fn test_concatenating_a_middle_axis() {
        run(&[vec![2, 3, 2, 3], vec![2, 1, 2, 3], vec![2, 4, 2, 3]], 1);
    }

    /// More elements than one block of lanes, so the last tile of a chunk is partly
    /// masked off.
    #[test]
    fn test_a_chunk_longer_than_a_block() {
        run(&[vec![2, 300], vec![2, 101]], 1);
    }
}
