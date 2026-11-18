//! Recomputes every reduction and softmax in an export from its own inputs.
//!
//! Three faults in this class have cost a day between them: the axes dropped at
//! lowering, the axes dropped at import when ONNX moved them to an operand, and the
//! softmax grid that wrote one row. Each was found by hand, from a symptom a long way
//! downstream. This checks the whole class at once: every node of these kinds is pinned,
//! its inputs are pinned, and its output is compared against the same operation done on
//! the host from those inputs. A kernel that reduces the wrong axis cannot agree with
//! it.
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

/// How a reduction combines its elements.
#[derive(Clone, Copy, PartialEq)]
enum Reduce {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    L1,
    L2,
    SumSquare,
}

fn reduce_kind(op: &Op) -> Option<(Reduce, &[i64], bool)> {
    match op {
        Op::ReduceSum { axes, keepdims, .. } => Some((Reduce::Sum, axes, *keepdims)),
        Op::ReduceMean { axes, keepdims, .. } => Some((Reduce::Mean, axes, *keepdims)),
        Op::ReduceMax { axes, keepdims, .. } => Some((Reduce::Max, axes, *keepdims)),
        Op::ReduceMin { axes, keepdims, .. } => Some((Reduce::Min, axes, *keepdims)),
        Op::ReduceProd { axes, keepdims, .. } => Some((Reduce::Prod, axes, *keepdims)),
        Op::ReduceL1 { axes, keepdims, .. } => Some((Reduce::L1, axes, *keepdims)),
        Op::ReduceL2 { axes, keepdims, .. } => Some((Reduce::L2, axes, *keepdims)),
        Op::ReduceSumSquare { axes, keepdims, .. } => Some((Reduce::SumSquare, axes, *keepdims)),
        _ => None,
    }
}

/// The reduction, done on the host over `in_shape`'s `axes`.
fn reduce_on_host(x: &[f32], in_shape: &[usize], axes: &[usize], kind: Reduce) -> Vec<f32> {
    let out_len: usize = in_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, d)| *d)
        .product::<usize>()
        .max(1);
    let init = match kind {
        Reduce::Max => f32::NEG_INFINITY,
        Reduce::Min => f32::INFINITY,
        Reduce::Prod => 1.0,
        _ => 0.0,
    };
    let mut out = vec![init; out_len];
    let mut count = vec![0usize; out_len];

    // Strides of the output, over the axes that survive.
    let kept: Vec<usize> = (0..in_shape.len()).filter(|i| !axes.contains(i)).collect();
    let mut out_strides = vec![1usize; kept.len()];
    for i in (0..kept.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * in_shape[kept[i + 1]];
    }

    let mut coord = vec![0usize; in_shape.len()];
    for v in x.iter().take(in_shape.iter().product::<usize>()) {
        let o: usize = kept
            .iter()
            .enumerate()
            .map(|(k, &axis)| coord[axis] * out_strides[k])
            .sum();
        let slot = &mut out[o];
        match kind {
            Reduce::Sum | Reduce::Mean => *slot += *v,
            Reduce::Max => *slot = slot.max(*v),
            Reduce::Min => *slot = slot.min(*v),
            Reduce::Prod => *slot *= *v,
            Reduce::L1 => *slot += v.abs(),
            Reduce::L2 | Reduce::SumSquare => *slot += *v * *v,
        }
        count[o] += 1;
        // Odometer over the input's coordinates.
        for axis in (0..coord.len()).rev() {
            coord[axis] += 1;
            if coord[axis] < in_shape[axis] {
                break;
            }
            coord[axis] = 0;
        }
    }
    for (o, slot) in out.iter_mut().enumerate() {
        match kind {
            Reduce::Mean => *slot /= count[o].max(1) as f32,
            Reduce::L2 => *slot = slot.sqrt(),
            _ => {}
        }
    }
    out
}

/// Softmax over `axis` of `in_shape`, done on the host.
fn softmax_on_host(x: &[f32], in_shape: &[usize], axis: usize, log: bool) -> Vec<f32> {
    let n: usize = in_shape.iter().product();
    let inner: usize = in_shape[axis + 1..].iter().product();
    let extent = in_shape[axis];
    let outer: usize = in_shape[..axis].iter().product();
    let mut out = vec![0.0f32; n];
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| o * extent * inner + k * inner + i;
            let max = (0..extent).fold(f32::NEG_INFINITY, |m, k| m.max(x[at(k)]));
            let sum: f32 = (0..extent).map(|k| (x[at(k)] - max).exp()).sum();
            for k in 0..extent {
                out[at(k)] = if log {
                    (x[at(k)] - max) - sum.ln()
                } else {
                    (x[at(k)] - max).exp() / sum
                };
            }
        }
    }
    out
}

#[test]
#[ignore = "needs FS96_ONNX and FS96_DATA"]
fn every_reduction_and_softmax_agrees_with_the_host() {
    let path = std::env::var("FS96_ONNX").expect("set FS96_ONNX");
    let data = std::env::var("FS96_DATA").expect("set FS96_DATA");

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

    // Every node of interest, and every input they have.
    let mut wanted: Vec<usize> = Vec::new();
    for d in 0..dag.len() {
        let g = dag_to_graph[d];
        if g == usize::MAX {
            continue;
        }
        let op = &import.graph.nodes[g].op;
        let interesting =
            reduce_kind(op).is_some() || matches!(op, Op::Softmax { .. } | Op::LogSoftmax { .. });
        if interesting {
            wanted.push(d);
            // The graph's own data input, not the DAG parent: where the lowering had to
            // transpose the axis innermost first, the DAG parent is that transpose and
            // its shape is permuted, while the op's axes count in the original order.
            if let Some(&data) = import.graph.nodes[g].inputs.first() {
                let dag_data = graph_to_dag[data];
                if dag_data < dag.len() {
                    wanted.push(dag_data);
                }
            }
        }
    }
    wanted.sort_unstable();
    wanted.dedup();
    println!("auditing {} nodes, pinning {}", wanted.len(), wanted.len());

    let refs: Vec<&[f32]> = feeds.iter().map(|v| v.as_slice()).collect();
    let got = model.run(&device, &refs, &wanted).expect("run");
    let slot_of: BTreeMap<usize, usize> = wanted
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();

    let mut checked = 0usize;
    let mut wrong = 0usize;
    for &d in &wanted {
        let g = dag_to_graph[d];
        // A pinned input may be a DAG node the lowering added, with no graph node of
        // its own; only the ops themselves are being audited.
        if g == usize::MAX {
            continue;
        }
        let op = &import.graph.nodes[g].op;
        let Some(&data_graph) = import.graph.nodes[g].inputs.first() else {
            continue;
        };
        let parent = graph_to_dag[data_graph];
        if parent >= dag.len() {
            continue;
        }
        let (Some(&os), Some(&ps)) = (slot_of.get(&d), slot_of.get(&parent)) else {
            continue;
        };
        let out = &got[os];
        let x = &got[ps];
        let in_shape: Vec<usize> = model.output_shape(parent).to_vec();
        if x.len() != in_shape.iter().product::<usize>() {
            continue;
        }

        let want = if let Some((kind, axes, _)) = reduce_kind(op) {
            let rank = in_shape.len();
            let resolved: Vec<usize> = if axes.is_empty() {
                (0..rank).collect()
            } else {
                axes.iter()
                    .map(|a| if *a < 0 { a + rank as i64 } else { *a } as usize)
                    .collect()
            };
            reduce_on_host(x, &in_shape, &resolved, kind)
        } else {
            let (dim, log) = match op {
                Op::Softmax { dim, .. } => (*dim, false),
                Op::LogSoftmax { axis, .. } => (*axis, true),
                _ => continue,
            };
            let rank = in_shape.len();
            let axis = if dim < 0 { dim + rank as i64 } else { dim } as usize;
            if axis >= rank {
                continue;
            }
            softmax_on_host(x, &in_shape, axis, log)
        };

        if want.len() != out.len() {
            println!(
                "SHAPE  dag {d:>6} {:?} host gives {} values, kernel gives {} -- {}",
                op,
                want.len(),
                out.len(),
                model.node_name(d).unwrap_or("unnamed")
            );
            wrong += 1;
            continue;
        }
        checked += 1;
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let worst = want
            .iter()
            .zip(out)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        if worst > 1e-3 * scale {
            if wrong < 12 {
                println!(
                    "WRONG  dag {d:>6} {:?} shape {:?} worst {:.4e} (scale {:.3e}) -- {}",
                    op,
                    in_shape,
                    worst,
                    scale,
                    model.node_name(d).unwrap_or("unnamed")
                );
            }
            wrong += 1;
        }
    }
    println!("\n{checked} nodes compared, {wrong} disagree with the host");
    assert_eq!(wrong, 0, "{wrong} reductions or softmaxes are wrong");
}
