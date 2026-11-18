// The executor's buffers, and whether they are reused.
//
// A node's output buffer is only needed until the last node that reads it has run.
// Holding them all instead is what made a forward pass of the FoundationStereo export
// at 480x640 cost 273.60 GiB of buffers where its live set never exceeds 42.42 GiB.
//
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_buffer_reuse

#![cfg(all(feature = "hardware", any(feature = "cuda", feature = "riscv")))]

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
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

/// A chain of `n` unary ops over one input, each reading only the one before it.
fn chain(n: usize, elements: usize) -> Graph {
    let mut g = Graph::new();
    let mut last = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(elements)]);
    for _ in 0..n {
        last = g.add_node(Op::Relu, vec![last], DtypeRepr::F32, vec![Some(elements)]);
    }
    g
}

/// In a chain, only one intermediate is ever live, so every buffer after the first two
/// comes back from the pool rather than from the device.
#[test]
fn a_chain_reuses_one_buffer() {
    let n = 8usize;
    let elements = 64usize;
    let graph = chain(n, elements);
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

    let x: Vec<f32> = (0..elements).map(|i| i as f32 - 20.0).collect();
    let feeds: Vec<&[f32]> = vec![&x];
    let outputs = model.terminal_node_indices_sorted_by_size();
    let got = model.run(&device, &feeds, &outputs).expect("run failed");

    // The answer still has to be right: a relu chain is one relu.
    let out = got
        .iter()
        .find(|o| o.len() == elements)
        .expect("an output of the chain's width");
    for i in 0..elements {
        assert_eq!(out[i], x[i].max(0.0), "element {i}");
    }

    let stats = model.buffer_stats();
    assert_eq!(stats.acquired, n, "one buffer per op");
    // The last node is pinned as an output, so it is never released; everything before
    // it is a single buffer handed back and forth.
    assert!(
        stats.reused >= n - 2,
        "a chain should reuse almost every buffer: {stats:?}"
    );
    assert!(
        stats.allocated_words <= 2 * elements,
        "a chain should allocate about two buffers, not {} words",
        stats.allocated_words
    );
}

/// What the caller asks for is kept, whatever else happens to it: an intermediate that
/// is also an output stays readable after the run.
#[test]
fn a_pinned_output_is_not_recycled() {
    let elements = 32usize;
    let mut graph = Graph::new();
    let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(elements)]);
    let middle = graph.add_node(Op::Relu, vec![x], DtypeRepr::F32, vec![Some(elements)]);
    let end = graph.add_node(Op::Relu, vec![middle], DtypeRepr::F32, vec![Some(elements)]);

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

    let v: Vec<f32> = (0..elements).map(|i| i as f32 - 10.0).collect();
    let feeds: Vec<&[f32]> = vec![&v];
    // Both the middle and the end, in DAG order: the middle would otherwise be
    // released the moment the end has run.
    let outputs = vec![middle, end];
    let got = model.run(&device, &feeds, &outputs).expect("run failed");
    assert_eq!(got.len(), 2);
    for (k, out) in got.iter().enumerate() {
        for i in 0..elements {
            assert_eq!(out[i], v[i].max(0.0), "output {k} element {i}");
        }
    }
    // And it is still readable afterwards, which is what pinning means.
    assert!(model.read_output_f32(middle).is_ok());
}

/// An intermediate nothing asked for is gone once nothing reads it, and says so rather
/// than handing back whatever tensor the buffer now holds.
#[test]
fn a_released_buffer_says_it_was_released() {
    let elements = 16usize;
    let mut graph = Graph::new();
    let x = graph.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(elements)]);
    let middle = graph.add_node(Op::Relu, vec![x], DtypeRepr::F32, vec![Some(elements)]);
    let end = graph.add_node(Op::Relu, vec![middle], DtypeRepr::F32, vec![Some(elements)]);

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

    let v: Vec<f32> = (0..elements).map(|i| i as f32).collect();
    let feeds: Vec<&[f32]> = vec![&v];
    model.run(&device, &feeds, &[end]).expect("run failed");

    let err = model
        .read_output_f32(middle)
        .expect_err("the middle buffer should be gone");
    assert!(err.to_string().contains("holds no buffer"), "{err}");
}

/// A narrowing chain: each step is smaller than the last, so every request after the
/// first can be served by the buffer the previous step gave back even though none of
/// them is the same size.
///
/// With exact matching this allocated one buffer per distinct size and held them all,
/// which is what put the FoundationStereo export at 480x640 out of memory on a 16 GiB
/// card: 16.60 GiB ever allocated against a 3.62 GiB live peak. The values matter as
/// much as the counts, since a node served by a larger buffer must still read back
/// exactly its own elements.
#[test]
fn a_narrowing_chain_reuses_the_largest_buffer() {
    let (device, target) = device_and_target();

    // Slice each step down by one element: 64, 63, 62, ...
    let widths = [64usize, 63, 62, 61, 60, 59];
    let mut g = Graph::new();
    let input = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(widths[0])]);
    let mut last = input;
    for w in widths.iter().skip(1) {
        let relu = g.add_node(Op::Relu, vec![last], DtypeRepr::F32, vec![Some(*w)]);
        last = relu;
    }

    let mut model = RuntimeModel::compile(
        &device,
        &g,
        &TritonLowering::new(),
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("compile");

    let x: Vec<f32> = (0..widths[0]).map(|i| i as f32 - 32.0).collect();
    let got = model.run(&device, &[&x], &[last]).expect("run");
    let out = &got[0];

    // Relu of a relu is a relu, and each step keeps the leading elements.
    assert_eq!(out.len(), *widths.last().unwrap());
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, x[i].max(0.0), "element {i}");
    }

    let stats = model.buffer_stats();
    let planned = model.planned_buffer_stats(&[last]);
    assert_eq!(
        stats, planned,
        "the plan should predict the run: {stats:?} against {planned:?}"
    );
    // Five ops: the first two need buffers of their own, since the input keeps its own
    // and the first op's output is still live when the second runs. Every later step
    // fits in one that came back.
    assert_eq!(stats.acquired, widths.len() - 1);
    assert_eq!(
        stats.reused,
        widths.len() - 3,
        "a smaller step should take the larger free buffer: {stats:?}"
    );
}
