// Where does FoundationStereo's output stop being written?
//
// The 96x128 pass on riscv64 returns a disparity that is exactly zero, and the single
// terminal of the right length is the export's own `disp` (see test_fs_terminals), so
// the zeros come from inside the network. This is the same model and the same feeds on
// CUDA, which takes minutes instead of hours, with a ladder of intermediate nodes
// pinned so the first all-zero one shows up. CUDA has its own wrong answers for ragged
// matmuls (teenygrad#143), so what this can tell apart is zero from non-zero, not right
// from wrong.
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

#![cfg(all(feature = "hardware", feature = "cuda"))]

use std::collections::BTreeMap;
use std::time::Instant;

use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};
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
#[ignore = "needs FS96_ONNX and FS96_DATA"]
fn where_the_output_goes_to_zero() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let data = std::env::var("FS96_DATA").expect("set FS96_DATA");

    let started = Instant::now();
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!("imported and folded in {:.1?}", started.elapsed());

    // Are the normalization Constants that come out zero present as values at all?
    for (g, name) in import.graph.names.iter() {
        if name.ends_with("deconv32_16/conv1/IN/Constant_1")
            || name.ends_with("deconv32_16/conv2/bn1/Constant_1")
        {
            let in_tensors = import.tensors.get(g).map(|t| t.bytes.len());
            let in_folded = folded.get(g).map(|c| (c.bytes.len(), c.dtype));
            println!(
                "{name}: graph {g} shape {:?} tensors {:?} folded {:?}",
                import.graph.nodes[*g].shape, in_tensors, in_folded
            );
        }
    }

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

    let device = teeny_runtime::open().expect("a CUDA device");
    let target = teeny_runtime::default_target(&device).expect("target");
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
    .expect("compiling the model");
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

    // Pin every node, so each one's output can be read afterwards, and walk the DAG's
    // own topological order to find the earliest node that is all zero while every one
    // of its inputs is not. That node is where the zeros start.
    // Pinning every node keeps every buffer, which is 16.6 GiB at 480x640. The
    // diagnostic needs that; a plain end-to-end run only needs the output, so the pool
    // can reuse the rest.
    let disp_dag = graph_to_dag[*import.value_to_node.get("disp").expect("disp")];
    let pin_all = std::env::var("FS_PIN_ALL").is_ok();
    let outputs: Vec<usize> = if pin_all {
        (0..model.len()).collect()
    } else {
        vec![disp_dag]
    };
    println!(
        "pinning {} node(s){}",
        outputs.len(),
        if pin_all { " (FS_PIN_ALL)" } else { "" }
    );

    let started = Instant::now();
    let got = model
        .run(&device, &refs, &outputs)
        .expect("running the model");
    println!("ran in {:.1?}", started.elapsed());

    if !pin_all {
        let out = &got[0];
        let max = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "disp (dag {disp_dag}): {} values, max |cuda| = {max:.4}",
            out.len()
        );
        if out.len() == want.len() {
            let sum: f64 = out
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs() as f64)
                .sum();
            println!("mean |cuda - pytorch| = {:.4}", sum / want.len() as f64);
        } else {
            println!(
                "no reference of this length ({} values), reporting the output only",
                want.len()
            );
        }
        return;
    }

    let all_zero: Vec<bool> = got
        .iter()
        .map(|v| !v.is_empty() && v.iter().all(|x| *x == 0.0))
        .collect();
    let empty: Vec<bool> = got.iter().map(|v| v.is_empty()).collect();

    // Only what the output actually depends on: a zero in a dead branch says nothing.
    let disp_dag_for_walk = graph_to_dag[*import.value_to_node.get("disp").expect("disp")];
    let mut needed = vec![false; dag.len()];
    let mut stack = vec![disp_dag_for_walk];
    while let Some(d) = stack.pop() {
        if needed[d] {
            continue;
        }
        needed[d] = true;
        for &pp in &dag.node(d).parents {
            stack.push(pp);
        }
    }
    println!(
        "{} of {} all-zero nodes are on disp's path",
        (0..dag.len()).filter(|&d| all_zero[d] && needed[d]).count(),
        all_zero.iter().filter(|x| **x).count()
    );

    let mut found = 0usize;
    for d in dag.topological_sort() {
        if !all_zero[d] || !needed[d] {
            continue;
        }
        // A constant or a fed weight can be legitimately zero: a normalization bias
        // usually is. Only a computed node that turns non-zero inputs into zeros is
        // evidence of a fault.
        let gk = dag_to_graph[d];
        if gk != usize::MAX
            && matches!(
                import.graph.nodes[gk].op,
                teeny_core::graph::Op::Constant { .. }
                    | teeny_core::graph::Op::Input
                    | teeny_core::graph::Op::ConstantOfShape { .. }
            )
        {
            continue;
        }
        let parents = &dag.node(d).parents;
        if parents.iter().any(|&p| all_zero[p] || empty[p]) {
            continue; // inherited zeros, not the origin
        }
        let g = dag_to_graph[d];
        println!(
            "\nFIRST all-zero node with non-zero inputs: dag {d} {} shape {:?}",
            model.node_name(d).unwrap_or("unnamed"),
            model.output_shape(d)
        );
        if g != usize::MAX {
            println!("  graph node {g}, op {:?}", import.graph.nodes[g].op);
        }
        for &pp in parents {
            let pg = dag_to_graph[pp];
            let maxv = got[pp].iter().fold(0.0f32, |m, x| m.max(x.abs()));
            println!(
                "  input dag {pp:>6} len {:>8} max |v| {maxv:>12.6} {:?} {}",
                got[pp].len(),
                pg.checked_sub(0).and_then(|g| if g == usize::MAX {
                    None
                } else {
                    Some(&import.graph.nodes[g].op)
                }),
                model.node_name(pp).unwrap_or("unnamed"),
            );
        }
        found += 1;
        if found == 5 {
            break;
        }
    }
    if found == 0 {
        println!("\nno node is all zero with non-zero inputs");
    }
    // Which kinds of node are zero, counted over the whole graph.
    {
        use std::collections::BTreeMap;
        let mut tally: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for d in 0..dag.len() {
            let g = dag_to_graph[d];
            if g == usize::MAX {
                continue;
            }
            let kind = format!("{:?}", import.graph.nodes[g].op);
            let kind = kind
                .split_whitespace()
                .next()
                .unwrap_or("?")
                .trim_end_matches('{')
                .to_string();
            let e = tally.entry(kind).or_insert((0, 0));
            e.1 += 1;
            if all_zero[d] {
                e.0 += 1;
            }
        }
        let mut rows: Vec<_> = tally.into_iter().filter(|(_, (z, _))| *z > 0).collect();
        rows.sort_by_key(|(_, (z, _))| std::cmp::Reverse(*z));
        println!("\nall-zero nodes by op kind (zero / total):");
        for (k, (z, t)) in rows.iter().take(25) {
            println!("  {k:<28} {z:>5} / {t:<5}");
        }
    }

    println!(
        "\n{} of {} nodes are all zero, {} are empty",
        all_zero.iter().filter(|x| **x).count(),
        got.len(),
        empty.iter().filter(|x| **x).count()
    );

    // The disparity itself. With every node pinned, several outputs have the
    // reference's length, so take the export's declared output by name rather than the
    // first of that length.
    let out = &got[outputs
        .iter()
        .position(|&i| i == disp_dag)
        .expect("disp was pinned")];
    assert_eq!(out.len(), want.len(), "disp has the wrong length");

    let max = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let mut worst = 0.0f32;
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
    let mean = sum / want.len() as f64;
    println!(
        "\ndisp (dag {disp_dag}): max |cuda| = {max:.4}, mean |cuda - pytorch| = {mean:.4}, \
         worst {worst:.4} at {at} (got {:.3}, want {:.3})",
        out[at], want[at]
    );
}
