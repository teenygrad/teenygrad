// Einsum: a two-operand contraction over one summed axis.
//
// Values are checked against a host reference that walks the equation's labels
// directly, rather than against the plan's strides, so the plan is validated too and
// not merely replayed. A batched matrix multiply is one instance of the same kernel,
// so it is checked here alongside the export's own equation.
//
//   cargo test -p teeny-kernels --test test_einsum
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_einsum

use teeny_core::graph::{DtypeRepr, Graph, Op, einsum_plan};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// Einsum on the host, by iterating the labels the equation names.
///
/// Deliberately independent of `einsum_plan`: it builds a coordinate per label and
/// indexes each operand by looking its labels up, so a wrong stride in the plan shows
/// up as a mismatch rather than being reproduced on both sides.
#[cfg_attr(not(feature = "hardware"), allow(dead_code))]
fn reference(
    equation: &str,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    let compact: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    let (inputs, out) = compact.split_once("->").expect("explicit output");
    let (al, bl) = inputs.split_once(',').expect("two operands");
    let a_labels: Vec<char> = al.chars().collect();
    let b_labels: Vec<char> = bl.chars().collect();
    let out_labels: Vec<char> = out.chars().collect();

    let extent = |label: char| -> usize {
        if let Some(i) = a_labels.iter().position(|&l| l == label) {
            a_shape[i]
        } else {
            b_shape[b_labels
                .iter()
                .position(|&l| l == label)
                .expect("known label")]
        }
    };

    // Everything summed over: the labels in the operands but not the output.
    let mut sum_labels: Vec<char> = Vec::new();
    for &l in a_labels.iter().chain(b_labels.iter()) {
        if !out_labels.contains(&l) && !sum_labels.contains(&l) {
            sum_labels.push(l);
        }
    }

    let out_shape: Vec<usize> = out_labels.iter().map(|&l| extent(l)).collect();
    let sum_shape: Vec<usize> = sum_labels.iter().map(|&l| extent(l)).collect();
    let n_out: usize = out_shape.iter().product();
    let n_sum: usize = sum_shape.iter().product::<usize>().max(1);

    let index = |labels: &[char], shape: &[usize], coord: &dyn Fn(char) -> usize| -> usize {
        let mut idx = 0usize;
        for (d, &l) in labels.iter().enumerate() {
            idx = idx * shape[d] + coord(l);
        }
        idx
    };

    let unflatten = |mut flat: usize, shape: &[usize]| -> Vec<usize> {
        let mut c = vec![0usize; shape.len()];
        for d in (0..shape.len()).rev() {
            c[d] = flat % shape[d];
            flat /= shape[d];
        }
        c
    };

    let mut out_vals = vec![0.0f32; n_out];
    for (o, slot) in out_vals.iter_mut().enumerate() {
        let oc = unflatten(o, &out_shape);
        let mut acc = 0.0f32;
        for s in 0..n_sum {
            let sc = unflatten(s, &sum_shape);
            let coord = |l: char| -> usize {
                if let Some(i) = out_labels.iter().position(|&x| x == l) {
                    oc[i]
                } else {
                    sc[sum_labels
                        .iter()
                        .position(|&x| x == l)
                        .expect("summed label")]
                }
            };
            acc += a[index(&a_labels, a_shape, &coord)] * b[index(&b_labels, b_shape, &coord)];
        }
        *slot = acc;
    }
    (out_vals, out_shape)
}

fn einsum_graph(equation: &str, a_shape: &[usize], b_shape: &[usize]) -> (Graph, Vec<usize>) {
    let plan = einsum_plan(equation, a_shape, b_shape).expect("plannable equation");
    let mut g = Graph::new();
    let a = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        a_shape.iter().map(|&d| Some(d)).collect(),
    );
    let b = g.add_node(
        Op::Input,
        vec![],
        DtypeRepr::F32,
        b_shape.iter().map(|&d| Some(d)).collect(),
    );
    g.add_node(
        Op::Einsum {
            equation: equation.to_string(),
        },
        vec![a, b],
        DtypeRepr::F32,
        plan.out_shape.iter().map(|&d| Some(d)).collect(),
    );
    (g, plan.out_shape)
}

fn lower(equation: &str, a_shape: &[usize], b_shape: &[usize]) -> anyhow::Result<usize> {
    let (graph, _) = einsum_graph(equation, a_shape, b_shape);
    TritonLowering::new()
        .lower(&graph, LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn a_contraction_lowers() {
    assert_eq!(lower("ij,jk->ik", &[2, 3], &[3, 4]).expect("matmul"), 3);
    assert_eq!(
        lower("bij,bjk->bik", &[2, 3, 4], &[2, 4, 5]).expect("batched matmul"),
        3
    );
    assert_eq!(
        lower("aijk,aijh->ajkh", &[1, 4, 3, 5], &[1, 4, 3, 2]).expect("the export's form"),
        3
    );
}

/// An equation the plan cannot express fails at lowering with the reason, rather than
/// as an unsupported op.
#[test]
fn an_unplannable_equation_says_why() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(3), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(3), Some(4)]);
    g.add_node(
        Op::Einsum {
            // A diagonal, which is not a strided read.
            equation: "ii,ij->ij".to_string(),
        },
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(3), Some(4)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a diagonal should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("takes a diagonal"), "{msg}");
}

/// If the equation and the recorded shape disagree, the output buffer and the
/// contraction describe different tensors, so lowering refuses.
#[test]
fn a_shape_that_disagrees_with_the_equation_is_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(3), Some(4)]);
    g.add_node(
        Op::Einsum {
            equation: "ij,jk->ik".to_string(),
        },
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(2), Some(5)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a disagreeing shape should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("but the node's shape is"), "{msg}");
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

    /// Values that make a wrong index or a dropped term visible: not symmetric, not
    /// round, and distinct across the whole tensor.
    fn vals(n: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.125) - offset).collect()
    }

    fn run(equation: &str, a_shape: &[usize], b_shape: &[usize]) {
        let a = vals(a_shape.iter().product(), 1.5);
        let b = vals(b_shape.iter().product(), 0.75);
        let (want, out_shape) = reference(equation, &a, a_shape, &b, b_shape);

        let (graph, plan_shape) = einsum_graph(equation, a_shape, b_shape);
        assert_eq!(
            plan_shape, out_shape,
            "{equation}: the plan and the reference disagree on the output shape"
        );

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

        let feeds: Vec<&[f32]> = vec![&a, &b];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == want.len())
            .unwrap_or_else(|| panic!("{equation}: no output of length {}", want.len()));

        let mut worst = 0.0f32;
        let mut at = 0usize;
        for i in 0..want.len() {
            // The contraction sums K products, so compare against the magnitude.
            let d = (out[i] - want[i]).abs() / want[i].abs().max(1.0);
            if d > worst {
                worst = d;
                at = i;
            }
        }
        if worst > 1e-5 {
            println!("  got[{at}]  = {}", out[at]);
            println!("  want[{at}] = {}", want[at]);
        }
        assert!(
            worst <= 1e-5,
            "{equation} over {a_shape:?} and {b_shape:?}: worst relative error {worst:e} at {at}"
        );
        println!("{equation} {a_shape:?} x {b_shape:?} -> {out_shape:?}, rel err {worst:e}");
    }

    /// The export's own equation, at a size that keeps the reference cheap.
    #[test]
    fn the_foundationstereo_equation_matches_the_reference() {
        run("aijk,aijh->ajkh", &[1, 4, 3, 5], &[1, 4, 3, 2]);
        run("aijk,aijh->ajkh", &[2, 3, 2, 4], &[2, 3, 2, 3]);
    }

    /// A matrix multiply is the same kernel, which is the claim the module docs make.
    #[test]
    fn matmul_is_one_instance_of_the_same_kernel() {
        run("ij,jk->ik", &[2, 3], &[3, 4]);
        run("ij,jk->ik", &[5, 7], &[7, 3]);
        // Contracting the other axis of the left operand.
        run("ji,jk->ik", &[3, 2], &[3, 4]);
    }

    #[test]
    fn a_batched_matmul_is_too() {
        run("bij,bjk->bik", &[2, 3, 4], &[2, 4, 5]);
        // Rank 4, the shape attention takes.
        run("bhij,bhjk->bhik", &[2, 2, 3, 4], &[2, 2, 4, 3]);
    }

    /// The output may name its labels in any order, and a transposed output is the
    /// case a stride mistake would most easily survive.
    #[test]
    fn a_transposed_output_is_indexed_correctly() {
        run("ij,jk->ki", &[2, 3], &[3, 4]);
        run("bij,bjk->bki", &[2, 3, 4], &[2, 4, 5]);
    }

    /// A contraction of length 1 still multiplies, and one of length 1 on a unit batch
    /// exercises the extent clamping in the index peel.
    #[test]
    fn degenerate_extents_still_work() {
        run("ij,jk->ik", &[1, 1], &[1, 1]);
        run("bij,bjk->bik", &[1, 2, 1], &[1, 1, 3]);
    }
}
