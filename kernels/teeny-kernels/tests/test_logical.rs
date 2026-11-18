// And, Or and Xor on bool tensors.
//
// Two paths have to agree: the constant folder evaluates these when both operands are
// known, which is how the export's own six `And` nodes disappear, and the kernels run
// the rest. Both read a non-zero byte as true, and both write 1 or 0, so a tensor that
// has been through either stays normalised.
//
//   cargo test -p teeny-kernels --test test_logical
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_logical

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn graph_of(op: Op, dtype: DtypeRepr, n: usize) -> Graph {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], dtype, vec![Some(n)]);
    let b = g.add_node(Op::Input, vec![], dtype, vec![Some(n)]);
    g.add_node(op, vec![a, b], dtype, vec![Some(n)]);
    g
}

fn lower(op: Op, dtype: DtypeRepr, n: usize) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&graph_of(op, dtype, n), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn all_three_lower_over_bools() {
    for op in [Op::And, Op::Or, Op::Xor] {
        let name = format!("{op:?}");
        assert_eq!(
            lower(op, DtypeRepr::Bool, 8).unwrap_or_else(|e| panic!("{name}: {e}")),
            3
        );
    }
}

/// The dtype sets how wide the buffers are and how the kernel reads them. An export
/// without value_info leaves a logical node carrying its first operand's dtype, so a
/// non-bool one is a mistake that actually happens, and reading a wider tensor a byte
/// at a time would be silent.
#[test]
fn a_non_bool_operand_is_refused() {
    let msg = match lower(Op::And, DtypeRepr::I64, 8) {
        Ok(_) => panic!("an I64 And should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("rather than Bool"), "{msg}");
}

/// Mixed dtypes: the output says Bool but an operand does not.
#[test]
fn a_bool_output_over_non_bool_operands_is_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::Bool, vec![Some(4)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::I32, vec![Some(4)]);
    g.add_node(Op::Or, vec![a, b], DtypeRepr::Bool, vec![Some(4)]);
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a non-bool operand should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("operand 1 is I32"), "{msg}");
}

/// ONNX broadcasts these; this kernel reads both operands at the output's offsets, so
/// a mismatch is refused rather than read past the shorter operand's end.
#[test]
fn operands_that_would_broadcast_are_refused() {
    let mut g = Graph::new();
    let a = g.add_node(Op::Input, vec![], DtypeRepr::Bool, vec![Some(4), Some(3)]);
    let b = g.add_node(Op::Input, vec![], DtypeRepr::Bool, vec![Some(1), Some(3)]);
    g.add_node(Op::And, vec![a, b], DtypeRepr::Bool, vec![Some(4), Some(3)]);
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("a broadcast should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("does not broadcast"), "{msg}");
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

    /// A bool tensor's bytes as the `f32` words the model's buffers are made of.
    ///
    /// Buffers are allocated in words whatever the dtype, and a bool is one byte per
    /// element, so four elements travel in one word. Feeding them as one f32 each
    /// would ask the buffer to hold four times what it has.
    fn as_words(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                f32::from_ne_bytes(w)
            })
            .collect()
    }

    /// Runs `op` over two bool tensors given as bytes, and returns the output bytes.
    ///
    /// The inputs include a true byte of 2 as well as 1, which is what tells a kernel
    /// that tests `!= 0` apart from one that only looks at the low bit.
    fn run(op: Op, a: &[u8], b: &[u8]) -> Vec<u8> {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let graph = graph_of(op, DtypeRepr::Bool, n);
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

        let af = as_words(a);
        let bf = as_words(b);
        let feeds: Vec<&[f32]> = vec![&af, &bf];
        // Nothing is read as f32: the output is bytes, so `run` is asked for no
        // outputs and the buffer is read at its own width afterwards.
        let terminal = model.terminal_node_indices_sorted_by_size();
        model.run(&device, &feeds, &[]).expect("run failed");
        let out = terminal
            .iter()
            .map(|&i| model.read_output_bytes(i).expect("read failed"))
            .find(|o| o.len() == n)
            .unwrap_or_else(|| panic!("no output of length {n}"));
        out
    }

    const A: [u8; 6] = [0, 0, 1, 1, 2, 0];
    const B: [u8; 6] = [0, 1, 0, 1, 1, 2];

    #[test]
    fn and_is_true_only_when_both_are() {
        assert_eq!(run(Op::And, &A, &B), vec![0, 0, 0, 1, 1, 0]);
    }

    #[test]
    fn or_is_true_when_either_is() {
        assert_eq!(run(Op::Or, &A, &B), vec![0, 1, 1, 1, 1, 1]);
    }

    /// A true byte of 2 against a true byte of 1 is false: both are normalised before
    /// they are compared, so `Xor` is a logical operator and not a bitwise one.
    #[test]
    fn xor_is_true_when_exactly_one_is() {
        assert_eq!(run(Op::Xor, &A, &B), vec![0, 1, 1, 0, 0, 1]);
    }

    /// More elements than one block of lanes, so the last block is partly masked off.
    #[test]
    fn a_longer_tensor_is_handled_whole() {
        let a: Vec<u8> = (0..3000).map(|i| u8::from(i % 3 == 0)).collect();
        let b: Vec<u8> = (0..3000).map(|i| u8::from(i % 5 == 0)).collect();
        let want: Vec<u8> = (0..3000)
            .map(|i| u8::from(i % 3 == 0 && i % 5 == 0))
            .collect();
        assert_eq!(run(Op::And, &a, &b), want);
    }
}
