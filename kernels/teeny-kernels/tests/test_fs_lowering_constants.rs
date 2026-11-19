//! How far materialised constants get the FoundationStereo graph through lowering.
//!
//! Ignored by default: needs the export, which is ~1.75 GB and lives outside the
//! repo. Point `FS_ONNX` at an `.onnx` file and run with `--ignored`.

use teeny_core::graph::Op;
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::errors::Error;
use teeny_kernels::graph::{TritonLowering, elidable_constants};
use teeny_onnx::Onnx;
use teeny_onnx::branches::materialize_and_resolve;
use teeny_onnx::materialize::MaterializeConfig;

#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn materialised_constants_clear_the_shape_operands() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    println!("unresolved If nodes at import: {}", import.branches.len());
    // Before materialisation, every shape operand is a live op, so nothing is
    // elidable and lowering has to produce a buffer for each.
    let before = count_constants(&import.graph, &elidable_constants(&import.graph));
    println!("before materialisation: {before:?}");

    // One call does the folding and the branch resolution, keeping the constant map
    // seeded correctly across rounds. Materialising separately here as well would
    // mis-seed this call and silently lose the first round's folds.
    let (all, report, branch_report) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!("folded {} nodes (first round)", report.folded);
    println!("branches: {branch_report:?}");

    let graph = &import.graph;
    let elide = elidable_constants(graph);
    let after = count_constants(graph, &elide);
    println!("after materialisation:  {after:?}");

    assert!(
        after.elidable > 0,
        "materialisation should make shape operands elidable"
    );

    // What would still need a device buffer, by dtype: these are the data constants,
    // the remaining work for lowering.
    let mut by_dtype: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Constant { .. }) && !elide[i] {
            *by_dtype.entry(format!("{:?}", node.dtype)).or_insert(0) += 1;
        }
    }
    println!("data constants still needing a buffer, by dtype: {by_dtype:?}");

    println!("constant values available to lowering: {}", all.len());

    // Gather's indices are a tensor operand, not shape metadata. Whether they are
    // constant decides what a Gather kernel has to support, so count both.
    {
        let mut const_idx = 0usize;
        let mut dynamic_idx = 0usize;
        let mut idx_lens: std::collections::BTreeMap<String, usize> = Default::default();
        for node in &graph.nodes {
            if !matches!(node.op, Op::Gather { .. }) {
                continue;
            }
            match node.inputs.get(1) {
                Some(&i) if all.contains_key(&i) => {
                    const_idx += 1;
                    let n = all[&i].len().unwrap_or(0);
                    let bucket = match n {
                        0 => "0",
                        1 => "1 (scalar)",
                        2..=16 => "2-16",
                        _ => "17+",
                    };
                    *idx_lens.entry(bucket.to_string()).or_insert(0) += 1;
                }
                _ => dynamic_idx += 1,
            }
        }
        println!("Gather nodes: {const_idx} with materialised indices, {dynamic_idx} dynamic");
        println!("  constant index counts: {idx_lens:?}");
    }

    // What ops lowering actually has to handle, most common first. The lowering error
    // names an op family rather than a node, so without this the next blocker is a
    // guess: an op's *presence* in the error is not evidence of how much of the model
    // depends on it.
    {
        let mut by_op: std::collections::BTreeMap<String, usize> = Default::default();
        for (i, node) in graph.nodes.iter().enumerate() {
            if elide[i] {
                continue;
            }
            let dbg = format!("{:?}", node.op);
            let variant = dbg
                .split_once([' ', '(', '{'])
                .map_or(dbg.as_str(), |(head, _)| head)
                .to_string();
            *by_op.entry(variant).or_insert(0) += 1;
        }
        let mut ranked: Vec<_> = by_op.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("ops lowering must handle ({} kinds):", ranked.len());
        for (op, n) in &ranked {
            println!("  {n:>6}  {op}");
        }
    }

    // Attempt lowering and report where it stops, so Op::Constant's absence from the
    // message is visible rather than assumed.
    let lowering = TritonLowering::new().with_constants(all);
    match lowering.lower(graph, LoweringMode::Inference) {
        Ok(dag) => println!("lowered the whole graph: {} DAG nodes", dag.len()),
        Err(e) => println!("lowering stops at: {e}"),
    }
}

#[derive(Debug)]
struct Counts {
    constants: usize,
    elidable: usize,
}

fn count_constants(graph: &teeny_core::graph::Graph, elide: &[bool]) -> Counts {
    let mut c = Counts {
        constants: 0,
        elidable: 0,
    };
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Constant { .. }) {
            c.constants += 1;
            if elide[i] {
                c.elidable += 1;
            }
        }
    }
    c
}

/// Whether the ops that read their parameters from operands actually lower.
///
/// The whole-graph attempt stops at the first op with no kernel, which hides what the
/// ops that *do* have one would have done. So each node is rebuilt as a one-node graph
/// with its own shapes and operand values and lowered on its own. The failures are
/// grouped by reason, since a count alone would not say whether the gap is a rank
/// limit, a dynamic extent or a missing operand.
#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn the_operand_driven_ops_lower() -> Result<(), Error> {
    use teeny_core::graph::ConstantMap;

    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut totals: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();

    for (i, node) in graph.nodes.iter().enumerate() {
        let op = match &node.op {
            Op::Transpose { .. } => "Transpose",
            Op::Slice => "Slice",
            Op::Expand => "Expand",
            Op::Clip { .. } => "Clip",
            Op::MatMul | Op::Gemm { .. } => "MatMul",
            _ => continue,
        };

        // Rebuild this one node: the positions it reads as shape metadata become
        // Constants carrying their values, and everything else becomes an Op::Input of
        // the same dtype and shape. Rebuilding an activation as a valueless Constant
        // would make the op report a missing value, which is a fault in this probe
        // rather than in the op, so the split has to follow the op's own reading.
        let mut g = teeny_core::graph::Graph::new();
        let mut constants = ConstantMap::new();
        let mut inputs = Vec::new();
        let operand_positions = node.op.shape_operand_positions();
        for (pos, &src) in node.inputs.iter().enumerate() {
            let s = &graph.nodes[src];
            if !operand_positions.contains(&pos) {
                inputs.push(g.add_node(Op::Input, vec![], s.dtype, s.shape.clone()));
                continue;
            }
            // An omitted operand has to stay omitted: rebuilding it as a Constant
            // with no value would make the consuming op report a missing value, which
            // is a fault in this probe rather than in the op.
            if matches!(s.op, Op::Omitted) {
                inputs.push(g.add_node(Op::Omitted, vec![], s.dtype, Vec::new()));
                continue;
            }
            let id = g.add_node(
                Op::Constant {
                    dtype: s.dtype,
                    shape: s.shape.clone(),
                },
                vec![],
                s.dtype,
                s.shape.clone(),
            );
            if let Some(v) = all.get(&src) {
                constants.insert(id, v.clone());
            }
            inputs.push(id);
        }
        let out = g.add_node(node.op.clone(), inputs, node.dtype, node.shape.clone());
        g.names
            .insert(out, graph.names.get(&i).cloned().unwrap_or_default());

        let entry = totals.entry(op).or_insert((0, 0));
        match TritonLowering::new()
            .with_constants(constants)
            .lower(&g, LoweringMode::Inference)
        {
            Ok(_) => entry.0 += 1,
            Err(e) => {
                entry.1 += 1;
                // Keep the shape of the reason, not the specific numbers in it.
                let msg = e.to_string();
                let key = [
                    "exceeds the",
                    "dynamic or unknown",
                    "has no integer value",
                    "is not a permutation",
                    "shape is empty",
                    "describe different",
                    "do not describe a slice",
                    "nor is 1",
                    "not supported",
                    "has no value",
                    "is above max",
                    "outside f32 range",
                    "contracts over",
                    "do not broadcast",
                    "cannot address it",
                    "is not supported",
                ]
                .iter()
                .find(|k| msg.contains(**k))
                .map_or(msg.clone(), |k| (*k).to_string());
                let seen = reasons.entry(key).or_insert(0);
                *seen += 1;
                if *seen <= 3 {
                    let operands =
                        node.inputs
                            .iter()
                            .skip(1)
                            .map(|&src| {
                                all.get(&src).map(|t| t.to_i64_vec()).transpose().map_err(
                                    |reason| Error::UnreadableConstant {
                                        context: format!("{op} node {i}: operand {src}"),
                                        reason,
                                    },
                                )
                            })
                            .collect::<Result<Vec<Option<Vec<i64>>>, Error>>()?;
                    println!(
                        "  e.g. {op} {:?} dtype {:?} in {:?} -> out {:?} operands {:?}",
                        graph.names.get(&i),
                        node.dtype,
                        node.inputs.first().map(|&s| graph.nodes[s].shape.clone()),
                        node.shape,
                        operands
                    );
                }
            }
        }
    }

    for (op, (ok, failed)) in &totals {
        println!("{op}: {ok} lower, {failed} do not");
    }
    if !reasons.is_empty() {
        println!("failure reasons:");
        for (r, n) in &reasons {
            println!("  {n:>5}  {r}");
        }
    }

    let dtypes: std::collections::BTreeMap<String, usize> = graph
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.op,
                Op::Transpose { .. } | Op::Slice | Op::Expand | Op::Clip { .. }
            )
        })
        .fold(Default::default(), |mut m, n| {
            *m.entry(format!("{:?}", n.dtype)).or_insert(0) += 1;
            m
        });
    println!("by dtype: {dtypes:?}");

    assert!(
        totals.values().map(|(ok, _)| ok).sum::<usize>() > 0,
        "at least some strided copies should lower"
    );
    Ok(())
}

/// Walks back from the nodes whose shapes are missing, to find where the rank is lost.
#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn where_the_shapes_are_lost() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (_, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let label = |i: usize| -> String {
        graph
            .names
            .get(&i)
            .cloned()
            .unwrap_or_else(|| format!("node {i}"))
    };
    let variant = |i: usize| -> String {
        let d = format!("{:?}", graph.nodes[i].op);
        d.split_once([' ', '(', '{'])
            .map_or(d.clone(), |(h, _)| h.to_string())
    };

    // The frontier: nodes with no shape whose inputs all have one. That is where
    // inference stopped, as opposed to everything downstream of it.
    let has_shape = |i: usize| {
        !graph.nodes[i].shape.is_empty() && graph.nodes[i].shape.iter().all(Option::is_some)
    };
    let mut frontier: std::collections::BTreeMap<String, usize> = Default::default();
    let mut examples: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        if has_shape(i) || node.inputs.is_empty() {
            continue;
        }
        if node.inputs.iter().all(|&s| has_shape(s)) {
            *frontier.entry(variant(i)).or_insert(0) += 1;
            examples.entry(variant(i)).or_insert(i);
        }
    }
    let mut ranked: Vec<_> = frontier.into_iter().collect();
    ranked.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    println!("inference frontier: ops with no shape whose inputs all have one");
    for (op, n) in &ranked {
        let i = examples[op];
        let ins: Vec<String> = graph.nodes[i]
            .inputs
            .iter()
            .map(|&s| format!("{}{:?}", variant(s), graph.nodes[s].shape))
            .collect();
        println!("  {n:>5}  {op}");
        println!("         e.g. {} <- {}", label(i), ins.join(", "));
    }

    let total_missing = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|&(i, n)| !has_shape(i) && !n.inputs.is_empty())
        .count();
    println!(
        "nodes with no usable shape: {total_missing} of {}",
        graph.nodes.len()
    );
}

/// The ranks `MatMul` is asked to handle.
///
/// Its lowering reads `m` from input 0's first dimension and `k` from its last, which
/// is the 2-D reading. A rank-3-or-higher operand is a batch of matrices, and that
/// reading takes the batch extent for the row count, so it is worth knowing how many
/// of a real model's matmuls are batched before trusting any of them.
#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn matmul_ranks() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (_, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut by_rank: std::collections::BTreeMap<String, usize> = Default::default();
    for node in &graph.nodes {
        if !matches!(node.op, Op::MatMul | Op::Gemm { .. }) {
            continue;
        }
        let lhs = node
            .inputs
            .first()
            .map(|&i| graph.nodes[i].shape.len())
            .unwrap_or(0);
        let key = if graph.nodes[node.inputs[0]].shape.is_empty() {
            "unknown".to_string()
        } else {
            format!("rank {lhs}")
        };
        *by_rank.entry(key).or_insert(0) += 1;
    }
    println!("MatMul/Gemm by left operand rank: {by_rank:?}");

    // What the old 2-D reading made of each node: m from the output's first extent,
    // k from the left operand's last, n from the output's last. Counting where that
    // disagrees with the plan says how many products it was placing wrongly.
    let mut wrong = 0usize;
    let mut elements_missed = 0usize;
    for node in &graph.nodes {
        if !matches!(node.op, Op::MatMul | Op::Gemm { .. }) {
            continue;
        }
        let shape =
            |i: usize| -> Option<Vec<usize>> { graph.nodes[i].shape.iter().copied().collect() };
        let (Some(a), Some(b)) = (shape(node.inputs[0]), shape(node.inputs[1])) else {
            continue;
        };
        let Some(out): Option<Vec<usize>> = node.shape.iter().copied().collect() else {
            continue;
        };
        let plan = teeny_core::graph::matmul_plan(&a, &b).expect("plannable");
        let old = (
            out.first().copied().unwrap_or(1),
            a.last().copied().unwrap_or(1),
            out.last().copied().unwrap_or(1),
        );
        if old != (plan.m, plan.k, plan.n) || plan.batch > 1 {
            wrong += 1;
            // Rows the old reading would have computed, against the rows that exist.
            let covered = old.0 * old.2;
            let want: usize = out.iter().product();
            elements_missed += want.saturating_sub(covered);
        }
    }
    println!(
        "MatMul/Gemm the 2-D reading placed wrongly: {wrong}, output elements it \
         would never have written: {elements_missed}"
    );

    let mut shapes: std::collections::BTreeMap<String, usize> = Default::default();
    for node in &graph.nodes {
        if !matches!(node.op, Op::MatMul | Op::Gemm { .. }) {
            continue;
        }
        let show = |i: usize| -> String {
            let s = &graph.nodes[i].shape;
            if s.is_empty() {
                "?".to_string()
            } else {
                format!(
                    "{:?}",
                    s.iter()
                        .map(|d| d.map_or(-1i64, |v| v as i64))
                        .collect::<Vec<_>>()
                )
            }
        };
        let key = format!(
            "{} x {} -> {:?} {:?}",
            show(node.inputs[0]),
            show(node.inputs[1]),
            node.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            node.op,
        );
        *shapes.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &shapes {
        println!("{n:5}  {k}");
    }
}

/// What the export's remaining `ConstantOfShape` nodes look like after folding.
#[test]
#[ignore = "needs FS_ONNX"]
fn constant_of_shape_nodes() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut by_kind: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        let Op::ConstantOfShape { dtype, fill } = &node.op else {
            continue;
        };
        let folded = all.contains_key(&i);
        let shape: Option<Vec<usize>> = node.shape.iter().copied().collect();
        let elems = shape.as_ref().map(|s| s.iter().product::<usize>());
        let fill_bytes = fill.len();
        let operand = node
            .inputs
            .first()
            .map(|&s| format!("{:?}", graph.nodes[s].op))
            .unwrap_or_else(|| "none".into());
        let operand = operand
            .split_once([' ', '{'])
            .map_or(operand.clone(), |(h, _)| h.to_string());
        let key = format!(
            "folded={folded} dtype={dtype:?} shape={:?} elems={elems:?} fill={fill:?} ({fill_bytes}B) from={operand}",
            shape
        );
        *by_kind.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &by_kind {
        println!("{n:5}  {k}");
    }
}

/// The export's boolean logical nodes, with the dtypes they and their operands carry.
#[test]
#[ignore = "needs FS_ONNX"]
fn logical_nodes() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        let op = match &node.op {
            Op::And => "And",
            Op::Or => "Or",
            Op::Xor => "Xor",
            Op::Not => "Not",
            Op::Equal => "Equal",
            Op::Greater => "Greater",
            Op::Less => "Less",
            Op::Where => "Where",
            _ => continue,
        };
        let operands: Vec<String> = node
            .inputs
            .iter()
            .map(|&s| {
                format!(
                    "{:?}{}",
                    graph.nodes[s].dtype,
                    if all.contains_key(&s) { "(const)" } else { "" }
                )
            })
            .collect();
        let key = format!(
            "{op} out={:?} shape={:?} folded={} operands={operands:?}",
            node.dtype,
            node.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            all.contains_key(&i)
        );
        *kinds.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &kinds {
        println!("{n:5}  {k}");
    }
}

/// What feeds the export's `NonZero` nodes, and what consumes them.
#[test]
#[ignore = "needs FS_ONNX"]
fn nonzero_nodes() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    // Who reads each node.
    let mut readers: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        for &src in &node.inputs {
            readers.entry(src).or_default().push(i);
        }
    }

    let brief = |i: usize| -> String {
        let n = &graph.nodes[i];
        let op = format!("{:?}", n.op);
        let op = op
            .split_once([' ', '{'])
            .map_or(op.clone(), |(h, _)| h.to_string());
        format!(
            "{op}[{:?}]{}",
            n.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            if all.contains_key(&i) { "(const)" } else { "" }
        )
    };

    for (i, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, Op::NonZero) {
            continue;
        }
        let operand = node
            .inputs
            .first()
            .copied()
            .expect("NonZero has an operand");
        println!(
            "NonZero {} dtype={:?}  <- {}  -> {:?}",
            graph.names.get(&i).cloned().unwrap_or_default(),
            node.dtype,
            brief(operand),
            readers
                .get(&i)
                .map(|r| r.iter().map(|&c| brief(c)).collect::<Vec<_>>())
                .unwrap_or_default(),
        );
        // Walk back a couple of steps to see where the operand comes from.
        let mut frontier = vec![operand];
        for depth in 0..3 {
            let mut next = Vec::new();
            for &f in &frontier {
                for &src in &graph.nodes[f].inputs {
                    next.push(src);
                }
            }
            if next.is_empty() {
                break;
            }
            println!(
                "    -{}: {:?}",
                depth + 1,
                next.iter().map(|&s| brief(s)).collect::<Vec<_>>()
            );
            frontier = next;
        }
    }
}

/// The export's `GatherND` nodes: their operands, batch_dims and consumers.
#[test]
#[ignore = "needs FS_ONNX"]
fn gathernd_nodes() -> Result<(), Error> {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut readers: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        for &src in &node.inputs {
            readers.entry(src).or_default().push(i);
        }
    }
    let brief = |i: usize| -> String {
        let n = &graph.nodes[i];
        let op = format!("{:?}", n.op);
        let op = op
            .split_once([' ', '{'])
            .map_or(op.clone(), |(h, _)| h.to_string());
        format!(
            "{op}{:?}{}",
            n.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            if all.contains_key(&i) { "(const)" } else { "" }
        )
    };

    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        let Op::GatherND { batch_dims } = &node.op else {
            continue;
        };
        let values = node
            .inputs
            .iter()
            .map(|&s| -> Result<String, Error> {
                let Some(v) = all.get(&s) else {
                    return Ok("-".into());
                };
                let ints = v.to_i64_vec().map_err(|reason| Error::UnreadableConstant {
                    context: format!("GatherND node {i}: operand {s}"),
                    reason,
                })?;
                Ok(format!("{:?}", &ints[..ints.len().min(14)]))
            })
            .collect::<Result<Vec<String>, Error>>()?;
        let key = format!(
            "batch_dims={batch_dims} dtype={:?} operands={:?} values={values:?} -> {:?}",
            node.dtype,
            node.inputs.iter().map(|&s| brief(s)).collect::<Vec<_>>(),
            readers
                .get(&i)
                .map(|r| r.iter().map(|&c| brief(c)).collect::<Vec<_>>()),
        );
        *kinds.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &kinds {
        println!("{n:3}  {k}");
    }
    Ok(())
}

/// The export's Pad, Resize, GridSample, ConvTranspose and ScatterElements nodes.
#[test]
#[ignore = "needs FS_ONNX"]
fn remaining_nodes() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;
    let brief = |i: usize| -> String {
        let n = &graph.nodes[i];
        format!(
            "{:?}{:?}{}",
            n.dtype,
            n.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            if all.contains_key(&i) { "(c)" } else { "" }
        )
    };
    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for node in &graph.nodes {
        let tag = match &node.op {
            Op::GridSample { .. } => {
                let d = format!("{:?}", node.op);
                d.chars().take(150).collect::<String>()
            }
            _ => continue,
        };
        let key = format!(
            "{tag}\n        operands={:?} out={:?}",
            node.inputs.iter().map(|&s| brief(s)).collect::<Vec<_>>(),
            node.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
        );
        *kinds.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &kinds {
        println!("{n:4}  {k}");
    }
}

/// The export's `Split` nodes: axis, how many outputs, and who reads them.
#[test]
#[ignore = "needs FS_ONNX"]
fn split_nodes() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;
    let mut readers: std::collections::BTreeMap<usize, usize> = Default::default();
    for node in &graph.nodes {
        for &src in &node.inputs {
            *readers.entry(src).or_insert(0) += 1;
        }
    }
    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        let Op::Split { axis, num_outputs } = &node.op else {
            continue;
        };
        let key = format!(
            "axis={axis} num_outputs={num_outputs} operands={:?} out={:?} readers={}",
            node.inputs
                .iter()
                .map(|&s| format!(
                    "{:?}{:?}{}",
                    graph.nodes[s].dtype,
                    graph.nodes[s]
                        .shape
                        .iter()
                        .map(|d| d.map_or(-1i64, |v| v as i64))
                        .collect::<Vec<_>>(),
                    if all.contains_key(&s) { "(const)" } else { "" }
                ))
                .collect::<Vec<_>>(),
            node.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            readers.get(&i).copied().unwrap_or(0),
        );
        *kinds.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &kinds {
        println!("{n:5}  {k}");
    }
}

/// What the export's scatter nodes look like: shapes, dtypes, constant or not.
#[test]
#[ignore = "needs FS_ONNX"]
fn scatter_nodes() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;
    let brief = |i: usize| -> String {
        let n = &graph.nodes[i];
        format!(
            "{:?}{:?}{}",
            n.dtype,
            n.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            if all.contains_key(&i) { "(const)" } else { "" }
        )
    };
    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for node in &graph.nodes {
        let tag = match &node.op {
            Op::ScatterND => "ScatterND".to_string(),
            Op::ScatterElements { axis } => format!("ScatterElements axis={axis}"),
            _ => continue,
        };
        let key = format!(
            "{tag} out={:?}{:?} operands={:?}",
            node.dtype,
            node.shape
                .iter()
                .map(|d| d.map_or(-1i64, |v| v as i64))
                .collect::<Vec<_>>(),
            node.inputs.iter().map(|&s| brief(s)).collect::<Vec<_>>(),
        );
        *kinds.entry(key).or_insert(0) += 1;
    }
    for (k, n) in &kinds {
        println!("{n:5}  {k}");
    }
}

/// Which dtype pairs the export's `Cast` nodes ask for.
#[test]
#[ignore = "needs FS_ONNX"]
fn cast_pairs() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (_, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;
    let mut pairs: std::collections::BTreeMap<String, usize> = Default::default();
    for node in &graph.nodes {
        let Op::Cast { to } = &node.op else { continue };
        let from = node
            .inputs
            .first()
            .map(|&i| format!("{:?}", graph.nodes[i].dtype))
            .unwrap_or_else(|| "?".into());
        *pairs.entry(format!("{from} -> {to:?}")).or_insert(0) += 1;
    }
    for (k, n) in &pairs {
        println!("{n:5}  {k}");
    }
}

/// Every op in the export that still cannot be lowered, with a count and one reason.
///
/// The whole-graph lowering reports only the first stop, so clearing one blocker tells
/// you nothing about what is behind it. This rebuilds each node on its own, the way
/// `the_operand_driven_ops_lower` does, and tallies what fails.
#[test]
#[ignore = "needs FS_ONNX"]
fn what_still_does_not_lower() {
    use teeny_core::graph::ConstantMap;

    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut ok: std::collections::BTreeMap<String, usize> = Default::default();
    let mut bad: std::collections::BTreeMap<String, (usize, String)> = Default::default();

    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Input | Op::Constant { .. } | Op::Omitted) {
            continue;
        }
        let kind = format!("{:?}", node.op);
        let kind = kind
            .split_once([' ', '{', '('])
            .map_or(kind.clone(), |(h, _)| h.to_string());

        let mut g = teeny_core::graph::Graph::new();
        let mut constants = ConstantMap::new();
        let mut inputs = Vec::new();
        let operand_positions = node.op.shape_operand_positions();
        for (pos, &src) in node.inputs.iter().enumerate() {
            let s = &graph.nodes[src];
            if matches!(s.op, Op::Omitted) {
                inputs.push(g.add_node(Op::Omitted, vec![], s.dtype, Vec::new()));
                continue;
            }
            if !operand_positions.contains(&pos) {
                inputs.push(g.add_node(Op::Input, vec![], s.dtype, s.shape.clone()));
                continue;
            }
            let id = g.add_node(
                Op::Constant {
                    dtype: s.dtype,
                    shape: s.shape.clone(),
                },
                vec![],
                s.dtype,
                s.shape.clone(),
            );
            if let Some(v) = all.get(&src) {
                constants.insert(id, v.clone());
            }
            inputs.push(id);
        }
        let out = g.add_node(node.op.clone(), inputs, node.dtype, node.shape.clone());
        g.names
            .insert(out, graph.names.get(&i).cloned().unwrap_or_default());

        match TritonLowering::new()
            .with_constants(constants)
            .lower(&g, LoweringMode::Inference)
        {
            Ok(_) => *ok.entry(kind).or_insert(0) += 1,
            Err(e) => {
                let entry = bad.entry(kind).or_insert((0, String::new()));
                entry.0 += 1;
                if entry.1.is_empty() {
                    entry.1 = e.to_string().lines().take(2).collect::<Vec<_>>().join(" ");
                }
            }
        }
    }

    let total_bad: usize = bad.values().map(|(n, _)| n).sum();
    let total_ok: usize = ok.values().sum();
    println!("lowers: {total_ok} nodes / {} kinds", ok.len());
    println!("does not: {total_bad} nodes / {} kinds", bad.len());
    let mut by_count: Vec<_> = bad.iter().collect();
    by_count.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (kind, (n, why)) in by_count {
        println!("{n:6}  {kind:22}  {why}");
    }
}

/// Which elementwise binary nodes have operands that would need broadcasting.
///
/// The elementwise kernels are flat: they read both operands at the output's own
/// offsets. An operand of a different shape is then read past its end, which is a
/// segfault rather than a wrong answer, so this counts them by op and by shape.
#[test]
#[ignore = "needs FS_ONNX"]
fn which_elementwise_nodes_broadcast() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    let mut total = 0usize;
    for (i, node) in graph.nodes.iter().enumerate() {
        let name = match &node.op {
            Op::Add => "Add",
            Op::Sub => "Sub",
            Op::Mul => "Mul",
            Op::Div => "Div",
            Op::Pow => "Pow",
            Op::ElemMin => "Min",
            Op::ElemMax => "Max",
            Op::Equal => "Equal",
            Op::Greater => "Greater",
            Op::Less => "Less",
            Op::And => "And",
            Op::Or => "Or",
            Op::Xor => "Xor",
            Op::Where => "Where",
            _ => continue,
        };
        if all.contains_key(&i) {
            continue;
        }
        let out: Option<Vec<usize>> = node.shape.iter().copied().collect();
        let Some(out) = out else { continue };
        let operands: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .map(|&s| {
                graph.nodes[s]
                    .shape
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default()
            })
            .collect();
        if operands.iter().all(|s| *s == out) {
            continue;
        }
        total += 1;
        *kinds.entry(name.to_string()).or_insert(0) += 1;
    }
    println!("{total} elementwise nodes would need broadcasting");
    let mut by_count: Vec<_> = kinds.iter().collect();
    by_count.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (what, n) in by_count.iter().take(20) {
        println!("{n:6}  {what}");
    }
}

/// What running the whole export would cost in buffers, before any reuse.
///
/// The executor allocates one buffer per DAG node and holds them all for the whole
/// run, so this is the floor on memory for a single forward pass as things stand.
#[test]
#[ignore = "needs FS_ONNX"]
fn what_a_run_would_cost() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    let (all, _, _) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    let graph = &import.graph;

    let width = |d: teeny_core::graph::DtypeRepr| -> usize {
        use teeny_core::graph::DtypeRepr as T;
        match d {
            T::Bool | T::I8 | T::U8 => 1,
            T::I16 | T::U16 | T::F16 | T::BF16 => 2,
            T::I32 | T::U32 | T::F32 => 4,
            T::I64 | T::U64 | T::F64 => 8,
        }
    };

    let mut total = 0usize;
    let mut weights = 0usize;
    let mut biggest: Vec<(usize, String)> = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        // A constant the lowering reads is elided rather than allocated.
        if all.contains_key(&i) && !matches!(node.op, Op::Input) {
            continue;
        }
        let Some(shape): Option<Vec<usize>> = node.shape.iter().copied().collect() else {
            continue;
        };
        let bytes = shape.iter().product::<usize>() * width(node.dtype);
        total += bytes;
        if matches!(node.op, Op::Input) {
            weights += bytes;
        }
        biggest.push((bytes, format!("{:?} {shape:?}", node.dtype)));
    }
    biggest.sort_by_key(|(b, _)| std::cmp::Reverse(*b));

    println!(
        "buffers for one forward pass: {:.2} GiB over {} nodes",
        total as f64 / (1 << 30) as f64,
        graph.nodes.len()
    );
    println!(
        "of which inputs and weights: {:.2} GiB",
        weights as f64 / (1 << 30) as f64
    );
    for (bytes, what) in biggest.iter().take(6) {
        println!("  {:>8.1} MiB  {what}", *bytes as f64 / (1 << 20) as f64);
    }

    // What it would cost with the buffers reused: a node's output can go back in the
    // pool once every node that reads it has run. Walking the topological order and
    // tracking what is still live gives the peak, which is the number that decides
    // whether a forward pass fits in memory at all.
    let order = graph.topological_sort();
    let mut last_use: std::collections::BTreeMap<usize, usize> = Default::default();
    for (step, &id) in order.iter().enumerate() {
        for &src in &graph.nodes[id].inputs {
            last_use.insert(src, step);
        }
    }
    let bytes_of = |i: usize| -> usize {
        let Some(shape): Option<Vec<usize>> = graph.nodes[i].shape.iter().copied().collect() else {
            return 0;
        };
        shape.iter().product::<usize>() * width(graph.nodes[i].dtype)
    };

    let mut live = 0usize;
    let mut peak = 0usize;
    let mut held = 0usize;
    for (step, &id) in order.iter().enumerate() {
        let node = &graph.nodes[id];
        let elided = all.contains_key(&id) && !matches!(node.op, Op::Input);
        if !elided {
            live += bytes_of(id);
            // An input or a weight is held for the whole pass.
            if matches!(node.op, Op::Input) {
                held += bytes_of(id);
            }
        }
        peak = peak.max(live);
        // Release what nothing reads again.
        for (&buffer, &last) in last_use.iter() {
            if last == step && !matches!(graph.nodes[buffer].op, Op::Input) {
                live = live.saturating_sub(bytes_of(buffer));
            }
        }
    }
    println!(
        "peak live with reuse: {:.2} GiB (of which {:.2} GiB held for the pass)",
        peak as f64 / (1 << 30) as f64,
        held as f64 / (1 << 30) as f64
    );
}
