// Proves the ONNX -> import -> lower -> compile -> run path end to end.
//
// A two-op model (Conv + Relu) imported with teeny-onnx, lowered by TritonLowering,
// compiled by RuntimeModel and executed, with the result compared against ONNX
// Runtime's output for the same input.
//
// This path has never been exercised: RuntimeModel has only ever run graphs built by
// hand in Rust, never one imported from ONNX. Everything else planned for running
// FoundationStereo on RISC-V sits on top of it.
//
//   TINY_DIR=/tmp/tiny cargo test -p teeny-kernels --test test_onnx_execution_probe \
//       -- --nocapture --ignored

use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_runtime::Target;
use teeny_test::model::RuntimeModel;

fn floats(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("missing {path}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
#[ignore = "needs TINY_DIR with tiny_conv_relu.onnx, x.bin and y.bin"]
fn probe_onnx_execution() {
    dotenv::dotenv().ok();
    let dir = std::env::var("TINY_DIR").unwrap_or_else(|_| "/tmp/tiny".to_string());
    let model_path = format!("{dir}/tiny_conv_relu.onnx");

    let (graph, tensors) =
        teeny_onnx::Onnx::from_path_with_tensors(&model_path).expect("import failed");
    println!(
        "imported {} nodes, {} tensors",
        graph.nodes.len(),
        tensors.len()
    );
    for (idx, t) in &tensors {
        println!(
            "  tensor node {idx} ({}) shape={:?} elems={}",
            graph.names.get(idx).map(String::as_str).unwrap_or("?"),
            t.shape,
            t.bytes.len() / 4
        );
    }

    #[cfg(feature = "cuda")]
    let (device, target) = {
        let env = teeny_test::cuda::setup_cuda_env().expect("cuda env");
        let target = Target::new(env.capability);
        (env.device, target)
    };
    #[cfg(all(feature = "riscv", not(feature = "cuda")))]
    let (device, target) = {
        use teeny_riscv::compiler::target::Capability;
        (
            teeny_runtime::Device::new(Default::default()),
            Target::new(Capability::SpacemitK3),
        )
    };
    let lowering = TritonLowering::new();

    let mut model = RuntimeModel::compile(
        &device,
        &graph,
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("compile failed");
    println!("compiled: {} dag nodes", model.len());

    println!("-- param slots the compiled model expects --");
    for (name, node_idx, param_idx) in model.param_info_named() {
        println!("   node {node_idx} param {param_idx}: {name}");
    }
    println!("-- dag node names --");
    for i in 0..model.len() {
        if let Some(n) = model.node_name(i) {
            println!("   dag {i}: {n}  out_shape={:?}", model.output_shape(i));
        }
    }

    // Weights are NOT fed as inputs. The lowered Conv exposes param slots
    // ("conv1.weight", "conv1.bias") that must be bound with load_param_f32; the `w`
    // and `b` Op::Input nodes the importer creates for the same initializers are left
    // dangling, and still consume a feed each because `run` supplies one per op-less
    // node. So: bind the real values as params, and feed the dangling inputs zeros.
    let tensor_by_name: std::collections::HashMap<String, Vec<f32>> = tensors
        .iter()
        .filter_map(|(idx, t)| {
            let name = graph.names.get(idx)?.clone();
            let vals: Vec<f32> = t
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            Some((name, vals))
        })
        .collect();

    let mut bound = 0usize;
    for (slot, node_idx, param_idx) in model.param_info_named().collect::<Vec<_>>() {
        // "conv1.weight" -> the initializer feeding conv1's weight input.
        let onnx_name = match slot.rsplit_once('.') {
            Some((_, "weight")) => "w",
            Some((_, "bias")) => "b",
            _ => continue,
        };
        let data = tensor_by_name
            .get(onnx_name)
            .unwrap_or_else(|| panic!("no tensor for {slot}"));
        model
            .load_param_f32(node_idx, param_idx, data)
            .unwrap_or_else(|e| panic!("binding {slot}: {e}"));
        println!("bound {slot} <- '{onnx_name}' ({} values)", data.len());
        bound += 1;
    }
    assert!(bound > 0, "no params bound");

    let x = floats(&format!("{dir}/x.bin"));

    // `run` consumes one feed per op-less node in the DAG's *topological* order, which
    // is neither the graph's node order nor the DAG's index order, and is not exposed.
    // Empirically it follows the graph's node order here (initializers, then model
    // inputs), so feeds are assembled that way and the sizes are asserted to match the
    // compiled buffers, which is what catches a wrong order.
    let mut feeds: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, teeny_core::graph::Op::Input) {
            continue;
        }
        let name = graph.names.get(&idx).cloned().unwrap_or_default();
        if let Some(t) = tensor_by_name.get(&name) {
            labels.push(format!("{name}(dangling)"));
            feeds.push(vec![0.0f32; t.len()]);
        } else {
            labels.push(name);
            feeds.push(x.clone());
        }
    }
    println!(
        "feed order: {labels:?} sizes={:?}",
        feeds.iter().map(|f| f.len()).collect::<Vec<_>>()
    );

    let refs: Vec<&[f32]> = feeds.iter().map(|v| v.as_slice()).collect();
    let outputs = model.terminal_node_indices_sorted_by_size();
    let got = model.run(&device, &refs, &outputs).expect("run failed");

    let want = floats(&format!("{dir}/y.bin"));
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

    let mut max = 0.0f32;
    for i in 0..want.len() {
        let d = (out[i] - want[i]).abs();
        if d > max {
            max = d;
        }
    }
    println!("EXECUTION OK — max |teenygrad - onnxruntime| = {max:e}");
    assert!(max < 1e-4, "mismatch vs onnxruntime: {max:e}");
}
