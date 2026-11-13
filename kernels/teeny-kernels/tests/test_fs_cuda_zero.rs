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

    // What to keep. Pinning every node keeps every buffer, which is 273.60 GiB at
    // 480x640, so the diagnostic walks a ladder instead: the nodes the output depends
    // on, in the DAG's own topological order, every FS_PIN_STRIDE-th one. The earliest
    // rung that is all zero brackets where the zeros start, and FS_PIN_RANGE=a..b then
    // pins every node between two rungs to narrow it. Without either, only the output
    // is kept and the pool reuses the rest.
    let disp_dag = graph_to_dag[*import.value_to_node.get("disp").expect("disp")];

    let mut needed = vec![false; dag.len()];
    let mut stack = vec![disp_dag];
    while let Some(d) = stack.pop() {
        if needed[d] {
            continue;
        }
        needed[d] = true;
        for &pp in &dag.node(d).parents {
            stack.push(pp);
        }
    }
    // The output's ancestors, in the order they run.
    let path: Vec<usize> = dag
        .topological_sort()
        .into_iter()
        .filter(|&d| needed[d])
        .collect();
    println!("disp depends on {} of {} nodes", path.len(), dag.len());

    let stride: Option<usize> = std::env::var("FS_PIN_STRIDE")
        .ok()
        .map(|v| v.parse().unwrap());
    let range: Option<(usize, usize)> = std::env::var("FS_PIN_RANGE").ok().map(|v| {
        let (a, b) = v.split_once("..").expect("FS_PIN_RANGE=a..b");
        (a.parse().unwrap(), b.parse().unwrap())
    });

    let mut rungs: Vec<usize> = match (stride, range) {
        (n, Some((a, b))) => path[a.min(path.len())..b.min(path.len())]
            .iter()
            .copied()
            .step_by(n.unwrap_or(1).max(1))
            .collect(),
        (Some(n), None) => path.iter().copied().step_by(n.max(1)).collect(),
        (None, None) => Vec::new(),
    };
    if !rungs.contains(&disp_dag) {
        rungs.push(disp_dag);
    }
    let outputs = rungs.clone();
    println!("pinning {} node(s)", outputs.len());

    let started = Instant::now();
    let got = model
        .run(&device, &refs, &outputs)
        .expect("running the model");
    println!("ran in {:.1?}", started.elapsed());

    // Every pinned node in run order, with where the zeros begin.
    let pos: BTreeMap<usize, usize> = path
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();
    let mut rows: Vec<(usize, usize, usize)> = outputs
        .iter()
        .enumerate()
        .map(|(slot, &d)| (*pos.get(&d).unwrap_or(&usize::MAX), d, slot))
        .collect();
    rows.sort_unstable();

    println!(
        "\n{:>8}  {:>7}  {:>9}  {:>12}  {:>5}  {}",
        "step", "node", "len", "max |v|", "zero", "name"
    );
    let mut last_nonzero: Option<(usize, usize)> = None;
    let mut first_zero: Option<(usize, usize)> = None;
    let mut first_huge: Option<(usize, usize)> = None;
    for (step, d, slot) in &rows {
        let v = &got[*slot];
        let max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let gz = dag_to_graph[*d];
        let is_constant = gz != usize::MAX
            && matches!(
                import.graph.nodes[gz].op,
                teeny_core::graph::Op::Constant { .. }
                    | teeny_core::graph::Op::Input
                    | teeny_core::graph::Op::ConstantOfShape { .. }
            );
        let zero = !v.is_empty() && max == 0.0 && !is_constant;
        if zero && first_zero.is_none() {
            first_zero = Some((*step, *d));
        }
        if !zero {
            if first_zero.is_none() {
                last_nonzero = Some((*step, *d));
            }
        }
        // FoundationStereo's activations are of order 100; anything far past that is
        // garbage rather than a large answer.
        let huge = max > 1e6;
        if huge && first_huge.is_none() {
            first_huge = Some((*step, *d));
        }
        let flag = if zero {
            "ZERO"
        } else if huge {
            "HUGE"
        } else if max.is_nan() {
            "NAN"
        } else {
            ""
        };
        println!(
            "{step:>8}  {d:>7}  {:>9}  {max:>12.3e}  {flag:>5}  {}",
            v.len(),
            model.node_name(*d).unwrap_or("unnamed")
        );
    }
    match (last_nonzero, first_zero) {
        (Some((a, da)), Some((b, db))) => println!(
            "\nzeros start between step {a} (node {da}, {}) and step {b} (node {db}, {}): \
             rerun with FS_PIN_RANGE={a}..{}",
            model.node_name(da).unwrap_or("unnamed"),
            model.node_name(db).unwrap_or("unnamed"),
            b + 1
        ),
        (None, Some((b, db))) => println!(
            "\nthe first rung is already zero: step {b}, node {db} ({})",
            model.node_name(db).unwrap_or("unnamed")
        ),
        _ => println!("\nno pinned node is all zero"),
    }
    // Where a rung first blows up, with its inputs, so the op that did it is named
    // rather than guessed. An input that was not pinned cannot be reported.
    let maxes: BTreeMap<usize, f32> = outputs
        .iter()
        .enumerate()
        .map(|(slot, &d)| (d, got[slot].iter().fold(0.0f32, |m, x| m.max(x.abs()))))
        .collect();
    match first_huge {
        Some((step, d)) => {
            let g = dag_to_graph[d];
            println!(
                "\nfirst rung past 1e6: step {step}, node {d} ({}), max {:.3e}",
                model.node_name(d).unwrap_or("unnamed"),
                maxes[&d]
            );
            if g != usize::MAX {
                println!("  op {:?}", import.graph.nodes[g].op);
            }
            let slot_of = |n: usize| outputs.iter().position(|&o| o == n);
            for &pp in &dag.node(d).parents {
                let pg = dag_to_graph[pp];
                let max = maxes
                    .get(&pp)
                    .map(|m| format!("{m:.3e}"))
                    .unwrap_or_else(|| "not pinned".to_string());
                println!(
                    "  input dag {pp:>6} max {max:>12} {:?} {}",
                    if pg == usize::MAX {
                        None
                    } else {
                        Some(&import.graph.nodes[pg].op)
                    },
                    model.node_name(pp).unwrap_or("unnamed")
                );
            }
            // For a two-operand elementwise node, the element that blew up and what it
            // was computed from. `x / norm` cannot exceed one, since a norm is at least
            // the magnitude of any element it covers, so an operand pair that produces
            // 1e13 says which side is wrong.
            let parents = dag.node(d).parents.clone();
            if parents.len() == 2
                && let (Some(sa), Some(sb)) = (slot_of(parents[0]), slot_of(parents[1]))
            {
                let (a, b) = (&got[sa], &got[sb]);
                let out = &got[slot_of(d).expect("pinned")];
                if a.len() == out.len() && b.len() == out.len() {
                    let mut at = 0usize;
                    let mut worst = 0.0f32;
                    for (i, v) in out.iter().enumerate() {
                        if v.abs() > worst {
                            worst = v.abs();
                            at = i;
                        }
                    }
                    println!(
                        "  worst element {at}: {:?} = {:.6e}, {:?} = {:.6e}, out = {:.6e}",
                        model.node_name(parents[0]).unwrap_or("a"),
                        a[at],
                        model.node_name(parents[1]).unwrap_or("b"),
                        b[at],
                        out[at]
                    );
                    // How many divisors are far below the numerator they divide.
                    let bad = a
                        .iter()
                        .zip(b.iter())
                        .filter(|(x, n)| n.abs() < x.abs() * 0.999)
                        .count();
                    println!(
                        "  {bad} of {} elements have |numerator| > |divisor|, which a norm \
                         cannot allow",
                        a.len()
                    );
                }
            }
        }
        None => println!("no pinned node exceeds 1e6"),
    }

    // A norm is at least the magnitude of any element it covers, so when x / norm
    // exceeds one, either the reduction computed the wrong value or the broadcast
    // paired elements with the wrong norm. Which one is answered by recomputing the
    // norm from x under each candidate layout and seeing which matches what the
    // reduction produced: contiguous groups (the reduced axis innermost) or strided
    // ones (the reduced axis outermost).
    if let Some((_, huge)) = first_huge
        && let Some(x_node) = dag.node(huge).parents.first().copied()
        && let Some(reduce_node) = outputs
            .iter()
            .copied()
            .find(|&d| model.node_name(d).is_some_and(|n| n.contains("ReduceL2")))
        && let (Some(xs), Some(rs)) = (
            outputs.iter().position(|&o| o == x_node),
            outputs.iter().position(|&o| o == reduce_node),
        )
    {
        let x = &got[xs];
        let norm = &got[rs];
        if !norm.is_empty() && x.len() % norm.len() == 0 {
            let m = norm.len();
            let g = x.len() / m;
            let mut err_contiguous = 0.0f32;
            let mut err_strided = 0.0f32;
            for j in 0..m {
                let a: f32 = (0..g)
                    .map(|k| x[j * g + k] * x[j * g + k])
                    .sum::<f32>()
                    .sqrt();
                let b: f32 = (0..g)
                    .map(|k| x[j + k * m] * x[j + k * m])
                    .sum::<f32>()
                    .sqrt();
                err_contiguous = err_contiguous.max((a - norm[j]).abs());
                err_strided = err_strided.max((b - norm[j]).abs());
            }
            let zeros = norm.iter().filter(|v| **v == 0.0).count();
            println!(
                "\n{} over {} groups of {g}: {zeros} of {m} norms are exactly zero",
                model.node_name(reduce_node).unwrap_or("reduce"),
                m
            );
            println!(
                "  recomputed from {}: worst error {:.4e} with contiguous groups, \
                 {:.4e} with strided groups",
                model.node_name(x_node).unwrap_or("x"),
                err_contiguous,
                err_strided
            );

            // Which norm each element was actually given. With the reduced axis
            // innermost, element i belongs to group i / g; pairing it with i % m
            // instead is the same mistake read the other way round.
            if let Some(div_node) = dag.node(huge).parents.get(1).copied()
                && let Some(es) = outputs.iter().position(|&o| o == div_node)
            {
                let expanded = &got[es];
                if expanded.len() == x.len() {
                    let mut err_by_group = 0.0f32;
                    let mut err_by_modulo = 0.0f32;
                    for i in 0..expanded.len() {
                        err_by_group = err_by_group.max((expanded[i] - norm[i / g]).abs());
                        err_by_modulo = err_by_modulo.max((expanded[i] - norm[i % m]).abs());
                    }
                    println!(
                        "  {} holds: worst error {:.4e} against norm[i / {g}] (correct), \
                         {:.4e} against norm[i % {m}]",
                        model.node_name(div_node).unwrap_or("expand"),
                        err_by_group,
                        err_by_modulo
                    );
                }
            }
        }
    }

    // The disparity itself. With every node pinned, several outputs have the
    // reference's length, so take the export's declared output by name rather than the
    // first of that length.
    let out = &got[outputs
        .iter()
        .position(|&i| i == disp_dag)
        .expect("disp was pinned")];
    if out.len() != want.len() {
        let max = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "\ndisp (dag {disp_dag}): {} values, max |cuda| = {max:.4}; the reference is \
             {} values, so no comparison here",
            out.len(),
            want.len()
        );
        return;
    }

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
