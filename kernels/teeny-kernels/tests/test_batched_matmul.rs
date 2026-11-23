// MatMul with batch axes.
//
// An export's matrix multiplies are rarely rank 2: a weight multiply carries the
// batch (and often the spatial axes) in front of its rows, and attention carries a
// batch and a head on both operands. The two cases reach the same GEMM differently,
// and both are checked here against a host reference that indexes the operands by
// their own coordinates rather than through the plan.
//
//   cargo test -p teeny-kernels --test test_batched_matmul
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_batched_matmul

use teeny_core::graph::{DtypeRepr, Graph, Op, matmul_plan};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

/// A batched matrix product on the host, indexed from the shapes alone.
///
/// Deliberately independent of `matmul_plan`: the batch coordinate is unflattened
/// against each operand's own prefix, with a length-1 axis read at 0, so a wrong row
/// stride in the plan shows up as a mismatch rather than being reproduced on both
/// sides.
#[cfg_attr(not(feature = "hardware"), allow(dead_code))]
fn reference(a: &[f32], a_shape: &[usize], b: &[f32], b_shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let (m, k) = (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]);
    let n = b_shape[b_shape.len() - 1];
    let a_prefix = &a_shape[..a_shape.len() - 2];
    let b_prefix = &b_shape[..b_shape.len() - 2];

    let rank = a_prefix.len().max(b_prefix.len());
    let batch_shape: Vec<usize> = (0..rank)
        .map(|i| {
            let at = |p: &[usize]| {
                let off = rank - p.len();
                if i < off { 1 } else { p[i - off] }
            };
            at(a_prefix).max(at(b_prefix))
        })
        .collect();
    let batch: usize = batch_shape.iter().product();

    // The flat offset of batch `bi`'s matrix in an operand whose prefix is `prefix`,
    // in matrices: a length-1 axis is shared, so its coordinate is read as 0.
    let offset = |bi: usize, prefix: &[usize]| -> usize {
        let off = rank - prefix.len();
        let mut rest = bi;
        let mut coords = vec![0usize; rank];
        for d in (0..rank).rev() {
            coords[d] = rest % batch_shape[d];
            rest /= batch_shape[d];
        }
        let mut flat = 0usize;
        for (d, &extent) in prefix.iter().enumerate() {
            let c = if extent == 1 { 0 } else { coords[off + d] };
            flat = flat * extent + c;
        }
        flat
    };

    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a0 = offset(bi, a_prefix) * m * k;
        let b0 = offset(bi, b_prefix) * k * n;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += a[a0 + mi * k + ki] * b[b0 + ki * n + ni];
                }
                out[bi * m * n + mi * n + ni] = acc;
            }
        }
    }

    let mut out_shape = batch_shape;
    out_shape.push(m);
    out_shape.push(n);
    (out, out_shape)
}

/// The output shape ONNX gives a `MatMul`: the broadcast batch, then `m` and `n`.
fn out_shape(a_shape: &[usize], b_shape: &[usize]) -> Vec<usize> {
    let a_prefix = &a_shape[..a_shape.len() - 2];
    let b_prefix = &b_shape[..b_shape.len() - 2];
    let rank = a_prefix.len().max(b_prefix.len());
    let mut shape: Vec<usize> = (0..rank)
        .map(|i| {
            let at = |p: &[usize]| {
                let off = rank - p.len();
                if i < off { 1 } else { p[i - off] }
            };
            at(a_prefix).max(at(b_prefix))
        })
        .collect();
    shape.push(a_shape[a_shape.len() - 2]);
    shape.push(b_shape[b_shape.len() - 1]);
    shape
}

fn matmul_graph(a_shape: &[usize], b_shape: &[usize]) -> Graph {
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
        Op::MatMul,
        vec![a, b],
        DtypeRepr::F32,
        out_shape(a_shape, b_shape)
            .iter()
            .map(|&d| Some(d))
            .collect(),
    );
    g
}

fn lower(a_shape: &[usize], b_shape: &[usize]) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&matmul_graph(a_shape, b_shape), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn test_the_export_s_shapes_lower() {
    // A weight multiply, batch and spatial axes in front of the rows.
    assert_eq!(lower(&[2, 1921, 1024], &[1024, 1024]).expect("weight"), 3);
    assert_eq!(lower(&[1, 120, 160, 128], &[128, 512]).expect("spatial"), 3);
    // Attention, batched on both sides.
    assert_eq!(
        lower(&[2, 16, 1921, 64], &[2, 16, 64, 1921]).expect("attention"),
        3
    );
}

/// The whole point of the plan: the leading axes of a batched product are not rows of
/// one matrix, and reading `m` off the front of the shape would make them so.
#[test]
fn test_the_rows_come_from_the_plan_and_not_from_the_shape_s_front() {
    let plan = matmul_plan(&[2, 1921, 1024], &[1024, 1024]).expect("plan");
    assert_ne!(plan.m, 2, "2 is the batch, not the row count");
    assert_eq!(plan.m, 2 * 1921);

    let plan = matmul_plan(&[2, 16, 1921, 64], &[2, 16, 64, 1921]).expect("plan");
    assert_eq!(plan.m, 1921);
    assert_eq!(plan.batch, 32);
}

#[test]
fn test_an_unplannable_product_says_why() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(3), Some(4)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(5), Some(2)]);
    g.add_node(
        Op::MatMul,
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(3), Some(2)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("mismatched extents should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("contracts over 4"), "{msg}");
}

/// An operand with no usable shape cannot be planned, and guessing would put every
/// element in the wrong place.
#[test]
fn test_an_unshaped_operand_is_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(4), Some(2)]);
    g.add_node(
        Op::MatMul,
        vec![a, b],
        DtypeRepr::F32,
        vec![Some(3), Some(2)],
    );
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("an unshaped operand should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("left operand"), "{msg}");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// These ran only on RISC-V until now. They were ignored on CUDA against
/// teenygrad#133, integer division by a scalar kernel argument; that was the wrong
/// cause. #133 is fixed as of teenyc `18d0e1214`, and every other test ignored for
/// it now passes on CUDA, while all nine here still fail:
///
/// ```text
/// [8, 8]       @ [8, 5]       relative error 3.78
/// [8, 1]       @ [1, 8]                      0.66
/// [64, 6]      @ [6, 64]                     1.19
/// [1, 7, 8]    @ [4, 8, 9]                 172.56
/// [4, 7, 8]    @ [4, 8, 9]                   6.03
/// [3, 8, 8]    @ [8, 5]                      3.78
/// [3, 8, 129]  @ [129, 8]                    0.0077
/// ```
///
/// The unbatched `[8, 8] @ [8, 5]` is wrong too, so this is not about batching, and it
/// is not ours either: teenygrad#143. Upstream's own unmodified `MatmulForward`, run
/// directly in a clean worktree, is wrong on CUDA whenever `k` or `n` is not a multiple
/// of 4, exact whenever it is, and exact for every case on RISC-V. Those are the
/// innermost extents of the three tensor descriptors, so a `k` or `n` that is not a
/// multiple of 4 f32 gives a row stride that is not a multiple of 16 bytes and
/// `PaddingOption::Zero` stops holding. At `BLOCK_N` 4 the same shape faults with
/// `CUDA error: 716 (misaligned address)` instead of answering wrongly.
///
/// Every case here fits that: `n` of 5 and 9, and `k` of 1, 6 and 129. The
/// `[3, 8, 129]` case at 0.0077 is just TF32 against a 1e-3 tolerance and would pass a
/// looser one.
///
/// So these stay ignored on CUDA until #143 is fixed, not because of anything in
/// `matmul_forward`'s batching or K tail, both of which are exact on RISC-V.
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

    fn run(a_shape: &[usize], b_shape: &[usize]) {
        let a = vals(a_shape.iter().product(), 1.5);
        let b = vals(b_shape.iter().product(), 0.75);
        let (want, want_shape) = reference(&a, a_shape, &b, b_shape);
        assert_eq!(
            want_shape,
            out_shape(a_shape, b_shape),
            "{a_shape:?} @ {b_shape:?}: the reference and the graph disagree on the shape"
        );

        let graph = matmul_graph(a_shape, b_shape);
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
            .unwrap_or_else(|| panic!("no output of length {}", want.len()));

        let mut worst = 0.0f32;
        let mut at = 0usize;
        for i in 0..want.len() {
            // The product sums K terms, so compare against the magnitude.
            let d = (out[i] - want[i]).abs() / want[i].abs().max(1.0);
            if d > worst {
                worst = d;
                at = i;
            }
        }
        if worst > 1e-3 {
            println!("  got[{at}]  = {}", out[at]);
            println!("  want[{at}] = {}", want[at]);
            panic!("{a_shape:?} @ {b_shape:?}: worst relative error {worst} at {at}");
        }
    }

    /// `m` and `n` are deliberately not multiples of a tile size, so the single tile
    /// is partly out of bounds in both and the descriptors' padding is exercised.
    /// `k` is a multiple of `BLOCK_K`, which `a_partial_k_tile_is_dropped` below says
    /// why.
    macro_rules! device_test {
        ($name:ident, $a:expr, $b:expr, $doc:expr) => {
            #[doc = $doc]
            #[test]
            #[cfg_attr(
                feature = "cuda",
                ignore = "CUDA matmul is wrong for ragged k or n, teenygrad#143"
            )]
            fn $name() {
                run(&$a, &$b);
            }
        };
    }

    device_test!(
        an_unbatched_product_still_works,
        [8, 8],
        [8, 5],
        "The 2-D case the plan folds everything else into."
    );
    device_test!(
        a_shared_weight_multiplies_every_batch,
        [3, 8, 8],
        [8, 5],
        "A weight multiply: the batch is folded into the rows, so this is one GEMM."
    );
    device_test!(
        a_shared_weight_with_short_rows,
        [7, 3, 8],
        [8, 5],
        "Rows that do not fill a tile, folded across seven batches: 21 rows in all."
    );
    device_test!(
        both_operands_batched,
        [4, 7, 8],
        [4, 8, 9],
        "The attention form. Seven rows per batch against a 64-row tile, so every \
         batch's tile straddles where the next batch's rows begin and the \
         descriptors' per-batch bound is what stops it."
    );
    device_test!(
        two_batch_axes_on_both_operands,
        [2, 3, 7, 8],
        [2, 3, 8, 9],
        "Two batch axes, flattened into one batch of six."
    );
    device_test!(
        a_shared_left_operand,
        [1, 7, 8],
        [4, 8, 9],
        "A left operand shared across a batched right one: its row stride is 0, so \
         every batch reads the same matrix."
    );

    /// Every `k`, not just the multiples of a tile width.
    ///
    /// The kernel accumulates whole K tiles and then adds one more read back from
    /// `K - BLOCK_K`, masked to the lanes the loop has not already summed, because a
    /// tile hanging off the end of the contracted axis reads as all zeros on the
    /// RISC-V backend rather than as its in-bounds part. Before that, a `k` of 12
    /// with `BLOCK_K` of 8 summed only the first eight terms and a `k` of 4 summed
    /// none and returned zero: silently short, not obviously broken.
    ///
    /// The widths here cover a tail of every size against `BLOCK_K` of 1, 2, 4 and
    /// 8, including `k` equal to `BLOCK_K`, where the extra tile is nothing but
    /// overlap and its mask has to come out zero everywhere.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA matmul is wrong for ragged k or n, teenygrad#143"
    )]
    fn every_contracted_extent_is_summed_whole() {
        for k in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20, 28, 33] {
            println!("k = {k}");
            run(&[8, k], &[k, 8]);
        }
    }

    /// On CUDA a contraction whose length is not a multiple of four loses its last
    /// one to three terms.
    ///
    /// A tensor descriptor there loads the innermost axis in 16-byte units, so a
    /// partial tile is rounded down to a whole number of them and a tail of one, two
    /// or three f32 lanes falls off; the lanes beyond a short contraction are not
    /// zero either, which is why `k = 6` comes back as noise rather than as a short
    /// sum. Reading the tail tile back from inside the operands, which is what the
    /// RISC-V backend does, is no answer there: the offset it needs is not 16-byte
    /// aligned, and an unaligned descriptor offset is an illegal instruction on CUDA.
    ///
    /// Unchanged by the fix for the RISC-V side, and verified unchanged: these same
    /// widths were equally wrong before it. The tail wants reading with masked
    /// pointer loads rather than a descriptor to be right on both, and nothing in
    /// this workspace does a 2-D masked load yet.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA is wrong for ragged k or n, teenygrad#143"]
    fn test_a_short_k_tail_on_cuda() {
        run(&[64, 6], &[6, 64]);
        run(&[64, 257], &[257, 64]);
    }

    /// The K tail against the wider tiles, and under a batch.
    ///
    /// `BLOCK_K` is 16 from k = 32 and 32 from k = 128, so these leave tails of 12,
    /// 4 and 1 lane in tiles the first sweep never reaches. The batched pair also
    /// checks that the tail tile is read per batch and not once for the buffer.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA matmul is wrong for ragged k or n, teenygrad#143"
    )]
    fn a_k_tail_in_the_wider_tiles() {
        run(&[8, 44], &[44, 8]);
        run(&[8, 132], &[132, 8]);
        run(&[3, 8, 129], &[129, 8]);
        run(&[3, 7, 44], &[3, 44, 9]);
    }
}
