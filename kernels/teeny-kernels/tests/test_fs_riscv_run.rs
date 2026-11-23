// Running FoundationStereo on riscv64.
//
// The export at 480x640 compiles but cannot run here: the executor allocates one buffer
// per node and holds them all, which is 273 GiB for that graph. This runs the same
// model exported at 96x128, where the buffers come to 3.6 GiB, against the disparity
// PyTorch produces for the same pair of real images.
//
// The weights come from the file's initializers, bound in the order the executor
// consumes its feeds, and the two images are the ones the export was traced on,
// preprocessed by the same resize the export script uses and dumped as f32.
//
//   FS96_ONNX=/path/to/fs_rank4_fp32_static_96x128.onnx \
//   FS96_DATA=/path/with/left.f32,right.f32,disp_ref.f32 \
//   TEENYC_QEMU_RISCV64=/usr/bin/qemu-riscv64-static \
//     cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//     --test test_fs_riscv_run -- --ignored --nocapture

#![cfg(all(feature = "hardware", feature = "riscv", not(feature = "cuda")))]

use std::collections::BTreeMap;
use std::time::Instant;

use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};
use teeny_riscv::compiler::target::Capability;
use teeny_runtime::Target;
use teeny_test::model::RuntimeModel;

/// A file of little-endian f32s.
fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Raw bytes as the f32 words a buffer is made of: a weight of any dtype travels this
/// way, since the executor's buffers are words and its feeds are f32 slices.
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

#[test]
#[ignore = "needs FS96_ONNX, FS96_DATA, and a long time under qemu"]
fn test_the_export_runs_on_riscv_and_matches_pytorch() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let data = std::env::var("FS96_DATA").expect("set FS96_DATA");

    let started = Instant::now();
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!("imported and folded in {:.1?}", started.elapsed());

    let lowering = TritonLowering::new().with_constants(folded);
    let (dag, graph_to_dag) = lowering
        .lower_with_mapping(&import.graph, LoweringMode::Inference)
        .expect("the whole graph should lower");
    println!("lowered {} nodes", dag.len());

    // The executor hands its feeds to the input nodes in the DAG's topological order,
    // so the feeds have to be built in that order too. Walking it here is what lets a
    // weight be matched to its own node rather than to whichever came next.
    let mut dag_to_graph = vec![usize::MAX; dag.len()];
    for (graph_idx, &dag_idx) in graph_to_dag.iter().enumerate() {
        if dag_idx < dag.len() {
            dag_to_graph[dag_idx] = graph_idx;
        }
    }

    let left = read_f32(&format!("{data}/left.f32"));
    let right = read_f32(&format!("{data}/right.f32"));
    let want = read_f32(&format!("{data}/disp_ref.f32"));
    println!(
        "inputs: left {} right {} reference {} values",
        left.len(),
        right.len(),
        want.len()
    );

    let mut feeds: Vec<Vec<f32>> = Vec::new();
    let mut fed_weights = 0usize;
    for dag_idx in dag.topological_sort() {
        if !dag.node(dag_idx).value.as_ref().is_input() {
            continue;
        }
        let graph_idx = dag_to_graph[dag_idx];
        let name = import
            .graph
            .names
            .get(&graph_idx)
            .cloned()
            .unwrap_or_default();
        match name.as_str() {
            "left" => feeds.push(left.clone()),
            "right" => feeds.push(right.clone()),
            _ => {
                let tensor = import.tensors.get(&graph_idx).unwrap_or_else(|| {
                    panic!("input '{name}' (graph node {graph_idx}) has no tensor data")
                });
                fed_weights += tensor.bytes.len();
                feeds.push(as_words(&tensor.bytes));
            }
        }
    }
    println!(
        "{} feeds, {:.2} GiB of weights",
        feeds.len(),
        fed_weights as f64 / (1 << 30) as f64
    );

    let device = teeny_runtime::Device::new(Default::default());
    let target = Target::new(Capability::SpacemitK3);
    let started = Instant::now();
    let mut model = RuntimeModel::compile(
        &device,
        &import.graph,
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("compiling the model for riscv64");
    println!("compiled and allocated in {:.1?}", started.elapsed());

    // Conv2d and friends keep their weights in parameter slots, not as graph inputs,
    // and a slot nobody fills is zero: every one of the graph's 382 Conv2d nodes
    // returned exactly zero before this, weight and bias slot both being empty
    // (teenygrad#131's hidden param slots). The data is in the file; what is missing is
    // the binding. A node's slots take its own constant operands in order, so each
    // slot gets the next operand after the first that carries tensor data.
    let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
    let mut by_node: BTreeMap<usize, Vec<(usize, String)>> = BTreeMap::new();
    for (name, node_idx, param_idx) in slots {
        by_node.entry(node_idx).or_default().push((param_idx, name));
    }

    let mut bound = 0usize;
    let mut bound_values = 0usize;
    let mut unbound: Vec<String> = Vec::new();
    for (node_idx, mut node_slots) in by_node {
        node_slots.sort_by_key(|(pi, _)| *pi);
        let g = dag_to_graph[node_idx];
        if g == usize::MAX {
            unbound.push(format!("node {node_idx}: no graph node"));
            continue;
        }
        // The operands that carry tensor data, in the order the op takes them. A
        // `Constant` has no operands and is its own value, keyed by its own index.
        let operands: Vec<usize> = import.graph.nodes[g]
            .inputs
            .iter()
            .copied()
            .filter(|op_idx| import.tensors.contains_key(op_idx))
            .collect();
        let sources: Vec<usize> = if operands.is_empty() {
            import
                .tensors
                .contains_key(&g)
                .then_some(g)
                .into_iter()
                .collect()
        } else {
            operands
        };
        let mut constants = sources
            .into_iter()
            .map(|op_idx| (op_idx, &import.tensors[&op_idx]));
        for (param_idx, name) in node_slots {
            let Some((op_idx, tensor)) = constants.next() else {
                unbound.push(format!("{name}: node {node_idx} has no operand left"));
                continue;
            };
            let data = as_words(&tensor.bytes);
            match model.load_param_f32(node_idx, param_idx, &data) {
                Ok(()) => {
                    bound += 1;
                    bound_values += data.len();
                }
                Err(e) => unbound.push(format!("{name}: operand {op_idx}: {e}")),
            }
        }
    }
    println!(
        "bound {bound} parameter slots, {:.2} GiB; {} not bound",
        (bound_values * 4) as f64 / (1 << 30) as f64,
        unbound.len()
    );
    for u in unbound.iter().take(10) {
        println!("  unbound: {u}");
    }

    model.set_trace(std::env::var("FS96_TRACE").is_ok());
    let refs: Vec<&[f32]> = feeds.iter().map(|v| v.as_slice()).collect();
    let outputs = model.terminal_node_indices_sorted_by_size();
    let started = Instant::now();
    let got = model
        .run(&device, &refs, &outputs)
        .expect("running the model");
    println!("ran in {:.1?}", started.elapsed());

    let out = got
        .iter()
        .find(|o| o.len() == want.len())
        .unwrap_or_else(|| {
            panic!(
                "no output of length {}; got {:?}",
                want.len(),
                got.iter().map(|o| o.len()).collect::<Vec<_>>()
            )
        });

    let mut worst = 0.0f32;
    let n = want.len() as f64;
    let mut at = 0usize;
    let mut sum = 0.0f64;
    for i in 0..want.len() {
        let d = (out[i] - want[i]).abs();
        sum += d as f64;
        if d > worst {
            worst = d;
            at = i;
        }
    }
    let mean = sum / n;
    println!(
        "disparity: mean |riscv - pytorch| = {:.4}, worst {:.4} at {at} (got {:.3}, \
         want {:.3})",
        mean, worst, out[at], want[at]
    );
    // A disparity in pixels: a mean error below a tenth of a pixel is the same answer.
    assert!(
        mean < 0.1,
        "the disparity does not match PyTorch's: mean {mean}"
    );
}
