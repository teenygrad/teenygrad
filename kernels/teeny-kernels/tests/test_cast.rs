// Cast: the only one that lowers is the one that changes nothing.
//
// Exporters emit casts in quantity between a dtype and itself. All 300 in the
// FoundationStereo export are F32 to F32, so refusing them for want of a conversion
// kernel held up the whole graph for conversions none of them asks for.
//
//   cargo test -p teeny-kernels --test test_cast

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn cast_graph(from: DtypeRepr, to: DtypeRepr, node_dtype: DtypeRepr) -> Graph {
    let mut g = Graph::new();
    let x = g.add_node(Op::Input, vec![], from, vec![Some(6)]);
    g.add_node(Op::Cast { to }, vec![x], node_dtype, vec![Some(6)]);
    g
}

fn lower(from: DtypeRepr, to: DtypeRepr, node_dtype: DtypeRepr) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&cast_graph(from, to, node_dtype), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn a_cast_to_the_same_dtype_is_a_copy() {
    assert_eq!(
        lower(DtypeRepr::F32, DtypeRepr::F32, DtypeRepr::F32).expect("f32 to f32"),
        2
    );
    assert_eq!(
        lower(DtypeRepr::I64, DtypeRepr::I64, DtypeRepr::I64).expect("i64 to i64"),
        2
    );
}

/// A converting cast between F32, I32, I64 and Bool lowers to a conversion kernel.
#[test]
fn a_converting_cast_lowers_to_a_kernel() {
    for (from, to) in [
        (DtypeRepr::I64, DtypeRepr::F32),
        (DtypeRepr::I64, DtypeRepr::Bool),
        (DtypeRepr::Bool, DtypeRepr::F32),
        (DtypeRepr::F32, DtypeRepr::I64),
        (DtypeRepr::Bool, DtypeRepr::I64),
        (DtypeRepr::I32, DtypeRepr::I64),
    ] {
        assert_eq!(
            lower(from, to, to).unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e}")),
            2
        );
    }
}

/// A conversion with no kernel is refused with both dtypes named, rather than copying
/// the bytes across and calling them the new dtype.
#[test]
fn an_unsupported_conversion_says_what_it_would_have_to_convert() {
    let msg = match lower(DtypeRepr::F64, DtypeRepr::I32, DtypeRepr::I32) {
        Ok(_) => panic!("an F64 to I32 cast has no kernel"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("F64 to I32"), "{msg}");
}

/// A node whose own dtype disagrees with what it claims to cast to is refused too: the
/// output buffer's width comes from the node, so a copy would write the wrong amount.
#[test]
fn a_node_dtype_that_disagrees_with_the_target_is_refused() {
    let msg = match lower(DtypeRepr::F32, DtypeRepr::F32, DtypeRepr::I64) {
        Ok(_) => panic!("a disagreeing node dtype should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("the node is I64"), "{msg}");
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

    /// Bytes as the f32 words the executor feeds and returns.
    fn words(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                f32::from_le_bytes(w)
            })
            .collect()
    }

    fn bytes(words: &[f32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Casts `input` (raw bytes of `n` elements of `from`) to `to`, returning the
    /// output's raw bytes.
    fn run(from: DtypeRepr, to: DtypeRepr, input: &[u8], n: usize) -> Vec<u8> {
        let mut g = Graph::new();
        let x = g.add_node(Op::Input, vec![], from, vec![Some(n)]);
        g.add_node(Op::Cast { to }, vec![x], to, vec![Some(n)]);
        let (device, target) = device_and_target();
        let mut model = RuntimeModel::compile(
            &device,
            &g,
            &TritonLowering::new(),
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
        .expect("compile failed");
        let feed = words(input);
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &[&feed], &outputs).expect("run failed");
        bytes(&got[0])
    }

    fn i64s(v: &[i64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    const IDS: [i64; 6] = [0, 1, -3, 7, 0, 30521];

    #[test]
    fn i64_to_f32_on_device() {
        let n = IDS.len();
        let f = run(DtypeRepr::I64, DtypeRepr::F32, &i64s(&IDS), n);
        let f: Vec<f32> = f
            .chunks(4)
            .take(n)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(f, vec![0.0, 1.0, -3.0, 7.0, 0.0, 30521.0], "I64 to F32");
    }

    #[test]
    fn i64_to_bool_on_device() {
        let n = IDS.len();
        let b = run(DtypeRepr::I64, DtypeRepr::Bool, &i64s(&IDS), n);
        assert_eq!(&b[..n], &[0, 1, 1, 1, 0, 1], "I64 to Bool");
    }

    #[test]
    fn bool_to_f32_on_device() {
        let n = 6;
        let bf = run(DtypeRepr::Bool, DtypeRepr::F32, &[1, 0, 1, 1, 0, 0], n);
        let bf: Vec<f32> = bf
            .chunks(4)
            .take(n)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(bf, vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0], "Bool to F32");
    }

    #[test]
    fn f32_to_i64_on_device() {
        let n = 6;
        let src: Vec<u8> = [2.9f32, -2.9, 0.0, 41.0, -1.0, 7.5]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let fi = run(DtypeRepr::F32, DtypeRepr::I64, &src, n);
        let fi: Vec<i64> = fi
            .chunks(8)
            .take(n)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(fi, vec![2, -2, 0, 41, -1, 7], "F32 to I64 truncates");
    }

    #[test]
    fn bool_to_i64_on_device() {
        let n = 6;
        let bi = run(DtypeRepr::Bool, DtypeRepr::I64, &[1, 0, 1, 1, 0, 0], n);
        let bi: Vec<i64> = bi
            .chunks(8)
            .take(n)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(bi, vec![1, 0, 1, 1, 0, 0], "Bool to I64");
    }

    #[test]
    fn i32_to_i64_on_device() {
        let n = 6;
        let i32s: Vec<u8> = [5i32, -6, 0, 1, -1, 9]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let ii = run(DtypeRepr::I32, DtypeRepr::I64, &i32s, n);
        let ii: Vec<i64> = ii
            .chunks(8)
            .take(n)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(ii, vec![5, -6, 0, 1, -1, 9], "I32 to I64 sign-extends");
    }
}
