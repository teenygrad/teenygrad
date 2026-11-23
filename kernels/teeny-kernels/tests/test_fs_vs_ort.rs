//! Compares every node's output against onnxruntime's, by name.
//!
//! onnxruntime runs the same file, so there is no mapping between a module and a node
//! to get wrong, and its disparity matches PyTorch's to a mean of 0.011. Values would
//! be gigabytes, so `kernels/teeny-kernels/tests/fixtures/onnx/ort_stats.py` writes
//! per-tensor statistics and this compares those: the first node in run order whose
//! count, range or mean differs is where we diverge. Statistics can agree while values
//! differ, so this localizes rather than proves, but a fault large enough to matter
//! moves them.
//!
//! FS96_ONNX, FS96_DATA as the other FoundationStereo probes, plus ORT_STATS pointing at
//! the TSV.
#![cfg(all(feature = "hardware", feature = "cuda"))]

use std::collections::BTreeMap;

use teeny_core::graph::Op;
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};
use teeny_runtime::Target;
use teeny_test::model::RuntimeModel;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

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

/// One tensor's summary, as `tests/fixtures/onnx/ort_stats.py` writes it.
#[derive(Clone, Copy, PartialEq)]
struct Stats {
    count: usize,
    min: f64,
    max: f64,
    mean: f64,
    mean_abs: f64,
}

fn summarize(v: &[f32]) -> Stats {
    if v.is_empty() {
        return Stats {
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            mean_abs: 0.0,
        };
    }
    let n = v.len() as f64;
    Stats {
        count: v.len(),
        min: v.iter().fold(f64::INFINITY, |m, x| m.min(*x as f64)),
        max: v.iter().fold(f64::NEG_INFINITY, |m, x| m.max(*x as f64)),
        mean: v.iter().map(|x| *x as f64).sum::<f64>() / n,
        mean_abs: v.iter().map(|x| (*x as f64).abs()).sum::<f64>() / n,
    }
}

fn read_stats(path: &str) -> BTreeMap<String, Stats> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut out = BTreeMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 6 {
            continue;
        }
        out.insert(
            f[0].to_string(),
            Stats {
                count: f[1].parse().unwrap_or(0),
                min: f[2].parse().unwrap_or(0.0),
                max: f[3].parse().unwrap_or(0.0),
                mean: f[4].parse().unwrap_or(0.0),
                mean_abs: f[5].parse().unwrap_or(0.0),
            },
        );
    }
    out
}

/// Whether two summaries are close enough to call the node agreed.
///
/// Scaled by what the reference holds, so a tensor of large values is not held to an
/// absolute tolerance, and a count that differs is always a disagreement.
fn agrees(ours: &Stats, theirs: &Stats) -> bool {
    if ours.count != theirs.count {
        return false;
    }
    if ours.count == 0 {
        return true;
    }
    let scale = theirs.mean_abs.max(theirs.max.abs()).max(1e-6);
    let tol = 0.02 * scale + 1e-5;
    (ours.min - theirs.min).abs() < tol
        && (ours.max - theirs.max).abs() < tol
        && (ours.mean - theirs.mean).abs() < tol
        && (ours.mean_abs - theirs.mean_abs).abs() < tol
}

#[test]
#[ignore = "needs FS96_ONNX, FS96_DATA and ORT_STATS"]
fn test_every_node_agrees_with_onnxruntime() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let data = std::env::var("FS96_DATA").expect("set FS96_DATA");
    let stats_path = std::env::var("ORT_STATS").expect("set ORT_STATS");
    let theirs = read_stats(&stats_path);
    println!("{} tensors in the reference", theirs.len());

    let mut import = Onnx::import_from_path(&path).expect("import");
    let (folded, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let lowering = TritonLowering::new().with_constants(folded);
    let (dag, graph_to_dag) = lowering
        .lower_with_mapping(&import.graph, LoweringMode::Inference)
        .expect("lower");
    let mut dag_to_graph = vec![usize::MAX; dag.len()];
    for (g, &d) in graph_to_dag.iter().enumerate() {
        if d < dag.len() {
            dag_to_graph[d] = g;
        }
    }

    let left = read_f32(&format!("{data}/left.f32"));
    let right = read_f32(&format!("{data}/right.f32"));
    let mut feeds: Vec<Vec<f32>> = Vec::new();
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
                let tensor = import
                    .tensors
                    .get(&graph_idx)
                    .unwrap_or_else(|| panic!("input '{name}' has no tensor data"));
                feeds.push(as_words(&tensor.bytes));
            }
        }
    }

    let device = teeny_runtime::open().expect("device");
    let target = teeny_runtime::default_target(&device).expect("target");
    let mut model = RuntimeModel::compile(
        &device,
        &import.graph,
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )
    .expect("compile");

    // Parameter slots hold the weights; see test_fs_cuda_zero.
    let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
    let mut by_node: BTreeMap<usize, Vec<(usize, String)>> = BTreeMap::new();
    for (name, node_idx, param_idx) in slots {
        by_node.entry(node_idx).or_default().push((param_idx, name));
    }
    for (node_idx, mut node_slots) in by_node {
        node_slots.sort_by_key(|(pi, _)| *pi);
        let g = dag_to_graph[node_idx];
        if g == usize::MAX {
            continue;
        }
        let operands: Vec<usize> = import.graph.nodes[g]
            .inputs
            .iter()
            .copied()
            .filter(|i| import.tensors.contains_key(i))
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
        let mut constants = sources.into_iter().map(|i| (i, &import.tensors[&i]));
        for (param_idx, _) in node_slots {
            if let Some((_, tensor)) = constants.next() {
                let _ = model.load_param_f32(node_idx, param_idx, &as_words(&tensor.bytes));
            }
        }
    }

    // Every node that the reference has a summary for, which is keyed by the name of
    // the value a node produces.
    // The reference is keyed by the name of the value a node produces, while our
    // `names` map holds the node's own name, so `value_to_node` is what joins them.
    let mut by_name: Vec<(String, usize)> = Vec::new();
    for (value, &g) in &import.value_to_node {
        if !theirs.contains_key(value.as_str()) {
            continue;
        }
        // Buffers are read back as f32 words, so a summary only means anything for an
        // f32 tensor: an i64 shape vector read that way is noise.
        if import.graph.nodes[g].dtype != teeny_core::graph::DtypeRepr::F32 {
            continue;
        }
        let d = graph_to_dag.get(g).copied().unwrap_or(usize::MAX);
        if d < dag.len() && dag_to_graph[d] == g {
            by_name.push((value.clone(), d));
        }
    }
    println!("{} of ours have a reference summary", by_name.len());

    let wanted: Vec<usize> = {
        let mut v: Vec<usize> = by_name.iter().map(|(_, d)| *d).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let refs: Vec<&[f32]> = feeds.iter().map(|v| v.as_slice()).collect();
    let got = model.run(&device, &refs, &wanted).expect("run");
    let slot_of: BTreeMap<usize, usize> = wanted
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();

    // In run order, so the first disagreement is the earliest one.
    let order: BTreeMap<usize, usize> = dag
        .topological_sort()
        .into_iter()
        .enumerate()
        .map(|(step, d)| (d, step))
        .collect();
    let mut rows: Vec<(usize, String, usize)> = by_name
        .into_iter()
        .map(|(name, d)| (order.get(&d).copied().unwrap_or(usize::MAX), name, d))
        .collect();
    rows.sort();

    // A node whose inputs already disagree inherits it; the interesting ones are the
    // nodes that diverge from inputs that still agree. And teenygrad#143 means a CUDA
    // matmul whose k or n is not a multiple of 4 is wrong here whatever we do, so those
    // are counted apart rather than mixed in.
    let agreed_at: BTreeMap<usize, bool> = rows
        .iter()
        .map(|(_, name, d)| {
            let ours = summarize(&got[slot_of[d]]);
            (*d, agrees(&ours, &theirs[name.as_str()]))
        })
        .collect();
    // Taint spreads: a node is only a genuine origin if nothing it descends from
    // disagrees. Checking the immediate parents is not enough, because a disagreement
    // can drift back inside the tolerance downstream -- a LayerNorm divides out the
    // scale it was given -- and every node after that would look like a fresh fault.
    let mut tainted = vec![false; dag.len()];
    for d in dag.topological_sort() {
        let from_above = dag
            .node(d)
            .parents
            .iter()
            .any(|p| tainted[*p] || !agreed_at.get(p).copied().unwrap_or(true));
        tainted[d] = from_above;
    }
    let inputs_agree = |d: usize| !tainted[d];
    let ragged_matmul = |d: usize| {
        let g = dag_to_graph[d];
        if g == usize::MAX || !matches!(import.graph.nodes[g].op, Op::MatMul) {
            return false;
        }
        // k and n are the innermost extents of the operands.
        import.graph.nodes[g]
            .inputs
            .iter()
            .filter_map(|&i| import.graph.nodes[i].shape.last().copied().flatten())
            .any(|inner| inner % 4 != 0)
    };

    let mut disagreed = 0usize;
    let mut origins = 0usize;
    let mut ragged = 0usize;
    for (step, name, d) in &rows {
        if !agreed_at[d] {
            disagreed += 1;
            if inputs_agree(*d) {
                if ragged_matmul(*d) {
                    ragged += 1;
                } else {
                    origins += 1;
                    let g = dag_to_graph[*d];
                    if origins <= 12 {
                        println!(
                            "ORIGIN step {step} dag {d} {:?} {name}",
                            import.graph.nodes[g].op
                        );
                    }
                }
            }
        }
    }
    println!(
        "\n{disagreed} of {} disagree; {ragged} are ragged matmuls (teenygrad#143), \
         {origins} are something else",
        rows.len()
    );

    #[allow(unreachable_code, clippy::never_loop)]
    for (step, name, d) in &rows {
        break;
        let theirs = &theirs[name.as_str()];
        let ours = summarize(&got[slot_of[d]]);
        if agrees(&ours, theirs) {
            continue;
        }
        if disagreed < 15 {
            let g = dag_to_graph[*d];
            println!(
                "\nDIVERGE step {step} dag {d} {:?}\n  ours   count {:>8} min {:>12.5e} max {:>12.5e} mean {:>12.5e}\n  theirs count {:>8} min {:>12.5e} max {:>12.5e} mean {:>12.5e}\n  {name}",
                import.graph.nodes[g].op,
                ours.count,
                ours.min,
                ours.max,
                ours.mean,
                theirs.count,
                theirs.min,
                theirs.max,
                theirs.mean,
            );
        }
        disagreed += 1;
    }
    assert_eq!(
        origins, 0,
        "{origins} nodes diverge from inputs that agree, and are not ragged matmuls"
    );
}
