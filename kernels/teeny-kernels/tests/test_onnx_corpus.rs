//! Imports every model of the ONNX corpus, runs it, and compares its outputs against
//! onnxruntime's.
//!
//! The corpus is built by `scripts/onnx_corpus/export.py`: per model, a `model.onnx`, its
//! inputs and onnxruntime's outputs as raw tensors, and a `manifest.json` naming them.
//! Each model goes through the same path as the FoundationStereo probes (import,
//! materialise constant subgraphs, lower, compile, bind weights, run), and is caught on
//! its own: a model that fails to import, lower, compile or run is reported with the
//! stage and the error, and the next model still runs. The test fails at the end if any
//! model did not match.
//!
//! ONNX_CORPUS points at the corpus directory; ONNX_CORPUS_ONLY optionally restricts it
//! to a comma-separated list of model names.
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Onnx, branches::materialize_and_resolve};
use teeny_test::model::RuntimeModel;

/// Raw bytes as the f32 words a buffer is made of: a tensor of any dtype travels this
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

#[derive(Debug)]
struct TensorSpec {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    file: PathBuf,
}

fn specs(dir: &Path, list: &serde_json::Value) -> Vec<TensorSpec> {
    list.as_array()
        .expect("manifest list")
        .iter()
        .map(|t| TensorSpec {
            name: t["name"].as_str().unwrap().to_string(),
            dtype: t["dtype"].as_str().unwrap().to_string(),
            shape: t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect(),
            file: dir.join(t["file"].as_str().unwrap()),
        })
        .collect()
}

/// How one model fared.
enum Outcome {
    /// Every output within tolerance; the worst relative difference.
    Pass(f32),
    /// Ran, but an output differs; a description.
    Mismatch(String),
    /// Failed at a stage, with the error.
    Failed(&'static str, String),
}

/// Worst `|got - want| / max|want|` over one output.
fn rel_diff(got: &[f32], want: &[f32]) -> Result<f32, String> {
    if got.len() != want.len() {
        return Err(format!("{} values, want {}", got.len(), want.len()));
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() {
            return Err(format!("non-finite output at {i}"));
        }
        worst = worst.max((g - w).abs() / scale);
    }
    Ok(worst)
}

/// Runs one stage, turning an error or a panic into a failure at `stage`.
fn stage<T>(name: &'static str, f: impl FnOnce() -> anyhow::Result<T>) -> Result<T, Outcome> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(Outcome::Failed(name, format!("{e:#}"))),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            Err(Outcome::Failed(name, format!("panicked: {msg}")))
        }
    }
}

const TOLERANCE: f32 = 1e-3;

fn run_model(dir: &Path) -> Outcome {
    match try_model(dir) {
        Ok(o) | Err(o) => o,
    }
}

fn try_model(dir: &Path) -> Result<Outcome, Outcome> {
    let manifest: serde_json::Value = stage("manifest", || {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            dir.join("manifest.json"),
        )?)?)
    })?;
    let inputs = specs(dir, &manifest["inputs"]);
    let outputs = specs(dir, &manifest["outputs"]);

    let mut import = stage("import", || {
        Ok(Onnx::import_from_path(dir.join("model.onnx"))?)
    })?;
    let folded = stage("materialize", || {
        let (folded, _, _) = materialize_and_resolve(&mut import, &MaterializeConfig::default())?;
        Ok(folded)
    })?;
    // The lowering reads shape operands (a Slice's bounds, a Reshape's target) from
    // these. Materialising returns only the values it computed, so the file's own
    // initializers join them: a bound that is an initializer is as constant as one a
    // Constant node folded to.
    let mut constants = import.tensors.clone();
    constants.extend(folded);
    let lowering = TritonLowering::new().with_constants(constants);
    let (dag, graph_to_dag) = stage("lower", || {
        Ok(lowering.lower_with_mapping(&import.graph, LoweringMode::Inference)?)
    })?;
    let mut dag_to_graph = vec![usize::MAX; dag.len()];
    for (g, &d) in graph_to_dag.iter().enumerate() {
        if d < dag.len() {
            dag_to_graph[d] = g;
        }
    }

    // Feeds, in the executor's order (Input nodes in topological order): the model's
    // own inputs by name, everything else from the file's tensor values.
    let by_name: BTreeMap<&str, &TensorSpec> =
        inputs.iter().map(|s| (s.name.as_str(), s)).collect();
    let mut feeds: Vec<Vec<f32>> = Vec::new();
    for dag_idx in dag.topological_sort() {
        if !dag.node(dag_idx).value.as_ref().is_input() {
            continue;
        }
        let g = dag_to_graph[dag_idx];
        let name = import.graph.names.get(&g).cloned().unwrap_or_default();
        if let Some(spec) = by_name.get(name.as_str()) {
            let bytes = std::fs::read(&spec.file)
                .map_err(|e| Outcome::Failed("inputs", format!("{}: {e}", spec.file.display())))?;
            feeds.push(as_words(&bytes));
        } else if let Some(t) = import.tensors.get(&g) {
            feeds.push(as_words(&t.bytes));
        } else {
            return Err(Outcome::Failed(
                "inputs",
                format!("Input node {g} ('{name}') is neither a model input nor a constant"),
            ));
        }
    }

    let device = stage("device", || teeny_runtime::open())?;
    let mut model = stage("compile", || {
        let target = teeny_runtime::default_target(&device)?;
        RuntimeModel::compile(
            &device,
            &import.graph,
            &lowering,
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
    })?;

    // Parameter slots hold weights folded into ops: bind each from the constants its
    // graph node reads, in operand order (as the FoundationStereo probes do).
    let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
    if std::env::var("ONNX_CORPUS_SLOTS").is_ok() {
        for (name, node, idx) in &slots {
            println!("slot {name} node {node} param {idx}");
        }
    }
    // Only weight and bias slots hold the file's constants. Other named slots are an
    // op's own metadata (a broadcast's `axes`), which its runtime initialises itself.
    let mut by_node: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (name, node_idx, param_idx) in slots {
        if name.ends_with(".weight") || name.ends_with(".bias") {
            by_node.entry(node_idx).or_default().push(param_idx);
        }
    }
    stage("weights", || {
        for (node_idx, mut params) in by_node {
            params.sort_unstable();
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
            let sources = if operands.is_empty() {
                import
                    .tensors
                    .contains_key(&g)
                    .then_some(g)
                    .into_iter()
                    .collect()
            } else {
                operands
            };
            for (param_idx, src) in params.into_iter().zip(sources) {
                model.load_param_f32(
                    node_idx,
                    param_idx,
                    &as_words(&import.tensors[&src].bytes),
                )?;
            }
        }
        Ok(())
    })?;

    let wanted: Vec<usize> = stage("outputs", || {
        outputs
            .iter()
            .map(|o| {
                let g = *import
                    .value_to_node
                    .get(&o.name)
                    .ok_or_else(|| anyhow::anyhow!("no node produces output '{}'", o.name))?;
                let d = graph_to_dag[g];
                anyhow::ensure!(d < dag.len(), "output '{}' was lowered away", o.name);
                Ok(d)
            })
            .collect()
    })?;
    // With ONNX_CORPUS_TRACE and a trace/ from scripts/onnx_corpus/trace.py, every
    // traced intermediate is read back too, to find where a mismatch starts.
    let traced = if std::env::var("ONNX_CORPUS_TRACE").is_ok() {
        trace_targets(dir, &import, &graph_to_dag, &dag_to_graph, dag.len())
    } else {
        Vec::new()
    };
    let mut request = wanted.clone();
    request.extend(traced.iter().map(|t| t.dag));
    let refs: Vec<&[f32]> = feeds.iter().map(Vec::as_slice).collect();
    let mut got = stage("run", || model.run(&device, &refs, &request))?;
    let trace_vals = got.split_off(wanted.len());
    if !traced.is_empty() {
        let order: BTreeMap<usize, usize> = dag
            .topological_sort()
            .into_iter()
            .enumerate()
            .map(|(step, d)| (d, step))
            .collect();
        report_first_divergence(&import, &dag_to_graph, &order, &traced, &trace_vals);
    }

    let mut worst = 0.0f32;
    for (o, g) in outputs.iter().zip(&got) {
        if o.dtype != "f32" {
            continue;
        }
        let want = as_words(
            &std::fs::read(&o.file)
                .map_err(|e| Outcome::Failed("outputs", format!("{}: {e}", o.file.display())))?,
        );
        match rel_diff(g, &want) {
            Ok(d) if d <= TOLERANCE => worst = worst.max(d),
            Ok(d) => {
                return Ok(Outcome::Mismatch(format!(
                    "output '{}' {:?}: relative difference {d:e}",
                    o.name, o.shape
                )));
            }
            Err(e) => return Ok(Outcome::Mismatch(format!("output '{}': {e}", o.name))),
        }
    }
    Ok(Outcome::Pass(worst))
}

/// One traced tensor: its name, the DAG node producing it, and onnxruntime's value.
struct Traced {
    name: String,
    dag: usize,
    file: PathBuf,
    /// Element count and bytes per element, for comparing a non-f32 tensor's bytes.
    count: usize,
    width: usize,
}

fn trace_targets(
    dir: &Path,
    import: &teeny_onnx::Import,
    graph_to_dag: &[usize],
    dag_to_graph: &[usize],
    dag_len: usize,
) -> Vec<Traced> {
    let Ok(index) = std::fs::read_to_string(dir.join("trace/index.tsv")) else {
        return Vec::new();
    };
    index
        .lines()
        .filter_map(|line| {
            let mut f = line.split('\t');
            let (name, file) = (f.next()?, f.next()?);
            let count: usize = f.next()?.parse().ok()?;
            let width = match f.next().unwrap_or("f32") {
                "f32" | "i32" => 4,
                "i64" => 8,
                "bool" => 1,
                _ => return None,
            };
            let g = *import.value_to_node.get(name)?;
            let d = *graph_to_dag.get(g)?;
            (d < dag_len && dag_to_graph[d] == g).then(|| Traced {
                name: name.to_string(),
                dag: d,
                file: dir.join("trace").join(file),
                count,
                width,
            })
        })
        .collect()
}

/// Prints the first traced tensor, in run order, that differs from onnxruntime's, with
/// its op and its operands: the node a mismatch starts at, when its inputs still agree.
fn report_first_divergence(
    import: &teeny_onnx::Import,
    dag_to_graph: &[usize],
    order: &BTreeMap<usize, usize>,
    traced: &[Traced],
    vals: &[Vec<f32>],
) {
    let mut rows: Vec<(usize, usize)> = (0..traced.len())
        .map(|i| (order.get(&traced[i].dag).copied().unwrap_or(usize::MAX), i))
        .collect();
    rows.sort();
    let mut agreed = 0;
    for (_, i) in rows {
        let t = &traced[i];
        let raw = std::fs::read(&t.file).unwrap_or_default();
        let want = as_words(&raw);
        let g = dag_to_graph[t.dag];
        let verdict = if import.graph.nodes[g].dtype == teeny_core::graph::DtypeRepr::F32 {
            rel_diff(&vals[i], &want)
        } else {
            // Integers and bools must match byte for byte.
            let ours: Vec<u8> = vals[i].iter().flat_map(|w| w.to_ne_bytes()).collect();
            let n = t.count * t.width;
            match ours.get(..n) {
                Some(b) if b == &raw[..n.min(raw.len())] => Ok(0.0),
                Some(b) => Err(format!(
                    "bytes differ: ours {:?}.., ort {:?}..",
                    &b[..b.len().min(16)],
                    &raw[..raw.len().min(16)]
                )),
                None => Err(format!("{} bytes read back, {n} expected", ours.len())),
            }
        };
        let bad = match &verdict {
            Ok(d) => *d > TOLERANCE,
            Err(_) => true,
        };
        if !bad {
            agreed += 1;
            continue;
        }
        let node = &import.graph.nodes[g];
        let name_of = |n: usize| {
            import
                .graph
                .names
                .get(&n)
                .cloned()
                .unwrap_or_else(|| format!("#{n}"))
        };
        println!(
            "  first divergence after {agreed} agreeing tensors: '{}' = {:?} {:?}: {}",
            t.name,
            node.op,
            node.shape,
            match verdict {
                Ok(d) => format!("relative difference {d:e}"),
                Err(e) => e,
            }
        );
        for &inp in &node.inputs {
            let n = &import.graph.nodes[inp];
            println!("    input {} = {:?} {:?}", name_of(inp), n.op, n.shape);
        }
        println!("    ours (first 8): {:?}", &vals[i][..vals[i].len().min(8)]);
        println!("    ort  (first 8): {:?}", &want[..want.len().min(8)]);
        return;
    }
    println!("  every one of {agreed} traced tensors agrees");
}

#[test]
#[ignore = "needs ONNX_CORPUS (built by scripts/onnx_corpus/export.py)"]
fn onnx_corpus_matches_onnxruntime() {
    let root = PathBuf::from(std::env::var("ONNX_CORPUS").expect("set ONNX_CORPUS"));
    let only: Vec<String> = std::env::var("ONNX_CORPUS_ONLY")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .expect("corpus directory")
        .filter_map(|e| {
            let e = e.ok()?;
            e.path()
                .join("manifest.json")
                .exists()
                .then(|| e.file_name().into_string().ok())?
        })
        .filter(|n| only.is_empty() || only.contains(n))
        .collect();
    names.sort();

    let mut failed = 0;
    let mut rows = Vec::new();
    for name in &names {
        let t = std::time::Instant::now();
        let outcome = run_model(&root.join(name));
        let secs = t.elapsed().as_secs_f64();
        let row = match &outcome {
            Outcome::Pass(d) => {
                format!("{name:<16} PASS      worst rel. diff {d:.1e}  ({secs:.1} s)")
            }
            Outcome::Mismatch(m) => format!("{name:<16} MISMATCH  {m}  ({secs:.1} s)"),
            Outcome::Failed(stage, e) => {
                let first = e.lines().next().unwrap_or("");
                format!("{name:<16} FAILED    at {stage}: {first}")
            }
        };
        if !matches!(outcome, Outcome::Pass(_)) {
            failed += 1;
            if let Outcome::Failed(_, e) = &outcome {
                eprintln!("--- {name}: full error ---\n{e}");
            }
        }
        println!("{row}");
        rows.push(row);
    }
    println!("\n== ONNX corpus on {} ==", teeny_runtime::BACKEND_NAME);
    for r in &rows {
        println!("{r}");
    }
    assert_eq!(
        failed,
        0,
        "{failed} of {} models did not match onnxruntime",
        names.len()
    );
}
