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

use teeny_core::graph::DtypeRepr;
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
    file: PathBuf,
}

fn specs(dir: &Path, list: &serde_json::Value) -> Vec<TensorSpec> {
    list.as_array()
        .expect("manifest list")
        .iter()
        .map(|t| TensorSpec {
            name: t["name"].as_str().unwrap().to_string(),
            dtype: t["dtype"].as_str().unwrap().to_string(),
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

/// A model output and its reference value.
struct Expected {
    name: String,
    dtype: DtypeRepr,
    bytes: Vec<u8>,
}

fn manifest_dtype(s: &str) -> DtypeRepr {
    match s {
        "i64" => DtypeRepr::I64,
        "i32" => DtypeRepr::I32,
        "bool" => DtypeRepr::Bool,
        _ => DtypeRepr::F32,
    }
}

/// The corpus front end: a model directory with `manifest.json`.
fn try_model(dir: &Path) -> Result<Outcome, Outcome> {
    let manifest: serde_json::Value = stage("manifest", || {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            dir.join("manifest.json"),
        )?)?)
    })?;
    let read = |t: &TensorSpec| {
        std::fs::read(&t.file)
            .map_err(|e| Outcome::Failed("inputs", format!("{}: {e}", t.file.display())))
    };
    let mut inputs = BTreeMap::new();
    for t in specs(dir, &manifest["inputs"]) {
        inputs.insert(t.name.clone(), read(&t)?);
    }
    let mut outputs = Vec::new();
    for t in specs(dir, &manifest["outputs"]) {
        outputs.push(Expected {
            name: t.name.clone(),
            dtype: manifest_dtype(&t.dtype),
            bytes: read(&t)?,
        });
    }
    try_onnx(
        &dir.join("model.onnx"),
        Some(dir),
        &inputs,
        &outputs,
        std::env::var("ONNX_CORPUS_FREEZE").is_ok(),
    )
}

/// Imports, materialises, lowers, compiles and runs `model` on `inputs` (raw bytes by
/// input name), and compares its outputs with `outputs`. With `trace_dir` and
/// ONNX_CORPUS_TRACE, also reports the first diverging intermediate.
fn try_onnx(
    model_path: &Path,
    trace_dir: Option<&Path>,
    inputs: &BTreeMap<String, Vec<u8>>,
    outputs: &[Expected],
    freeze_int_inputs: bool,
) -> Result<Outcome, Outcome> {
    let mut import = stage("import", || Ok(Onnx::import_from_path(model_path)?))?;
    // The conformance suite passes every operand as a graph input, including what real
    // models carry as constants: metadata (a Reduce's axes, a Reshape's shape, a Split's
    // sizes, a Resize's scales) and weights (a LayerNorm's scale, a Conv's kernel). Those
    // are re-imported as constants, as an exporter would have written them; the data an
    // op works on stays a runtime input, so its kernel is still what is checked.
    if freeze_int_inputs {
        use teeny_core::graph::Op;
        let weighted = |op: &Op| {
            matches!(
                op,
                Op::LayerNorm { .. }
                    | Op::BatchNorm1d { .. }
                    | Op::BatchNorm2d { .. }
                    | Op::BatchNorm3d { .. }
                    | Op::InstanceNorm1d { .. }
                    | Op::InstanceNorm2d { .. }
                    | Op::InstanceNorm3d { .. }
                    | Op::GroupNorm { .. }
                    | Op::Conv1d { .. }
                    | Op::Conv2d { .. }
                    | Op::Conv3d { .. }
                    | Op::ConvTranspose { .. }
            )
        };
        let mut freeze: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for n in &import.graph.nodes {
            let meta = n.op.shape_operand_positions();
            for (pos, &i) in n.inputs.iter().enumerate() {
                if meta.contains(&pos) || (pos > 0 && weighted(&n.op)) {
                    freeze.insert(i);
                }
            }
        }
        let mut constants = BTreeMap::new();
        for (g, name) in &import.graph.names {
            let node = &import.graph.nodes[*g];
            if !matches!(node.op, Op::Input) || import.tensors.contains_key(g) {
                continue;
            }
            let int = matches!(node.dtype, DtypeRepr::I64 | DtypeRepr::I32);
            if (int || freeze.contains(g))
                && let Some(bytes) = inputs.get(name)
            {
                constants.insert(
                    name.clone(),
                    teeny_onnx::TensorData {
                        dtype: node.dtype,
                        shape: node.shape.clone(),
                        bytes: bytes.clone(),
                    },
                );
            }
        }
        if !constants.is_empty() {
            import = stage("import", || {
                Ok(Onnx::import_from_path_with_constants(
                    model_path, &constants,
                )?)
            })?;
        }
    }
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
    // Elided constants have no DAG node, and their `graph_to_dag` entry is a leftover
    // 0, so they are skipped: inverting them too would make DAG node 0 (usually a model
    // input) look like the constant, and the constant's bytes would be fed to it.
    let elided = teeny_kernels::graph::elidable_constants(&import.graph);
    let mut dag_to_graph = vec![usize::MAX; dag.len()];
    for (g, &d) in graph_to_dag.iter().enumerate() {
        if d < dag.len() && !elided[g] {
            dag_to_graph[d] = g;
        }
    }

    // Feeds, in the executor's order (Input nodes in topological order): the model's
    // own inputs by name, everything else from the file's tensor values.
    let mut feeds: Vec<Vec<f32>> = Vec::new();
    for dag_idx in dag.topological_sort() {
        if !dag.node(dag_idx).value.as_ref().is_input() {
            continue;
        }
        let g = dag_to_graph[dag_idx];
        let name = import.graph.names.get(&g).cloned().unwrap_or_default();
        if let Some(bytes) = inputs.get(&name) {
            feeds.push(as_words(bytes));
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
    let slots: Vec<(String, usize, usize)> = model
        .param_slots()
        .map(|(slot, node, idx)| (format!(".{slot}"), node, idx))
        .collect();
    if std::env::var("ONNX_CORPUS_SLOTS").is_ok() {
        for (name, node, idx) in &slots {
            println!("slot {name} node {node} param {idx}");
        }
    }
    // Weight, bias and running-statistics slots hold the file's constants, in operand
    // order. Other named slots are an op's own metadata (a broadcast's `axes`, a
    // gather's `indices`), which its runtime initialises itself.
    let mut by_node: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (name, node_idx, param_idx) in slots {
        if [".weight", ".bias", ".running_mean", ".running_var"]
            .iter()
            .any(|w| name.ends_with(w))
        {
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
                anyhow::ensure!(
                    d < dag.len() && !elided[g],
                    "output '{}' was lowered away",
                    o.name
                );
                Ok(d)
            })
            .collect()
    })?;
    // With ONNX_CORPUS_TRACE and a trace/ from scripts/onnx_corpus/trace.py, every
    // traced intermediate is read back too, to find where a mismatch starts.
    let traced = match trace_dir {
        Some(dir) if std::env::var("ONNX_CORPUS_TRACE").is_ok() => {
            trace_targets(dir, &import, &graph_to_dag, &dag_to_graph, dag.len())
        }
        _ => Vec::new(),
    };
    let mut request = wanted.clone();
    request.extend(traced.iter().map(|t| t.dag));
    let refs: Vec<&[f32]> = feeds.iter().map(Vec::as_slice).collect();
    let mut got = stage("run", || model.run(&device, &refs, &request))?;
    if std::env::var("ONNX_CORPUS_DEBUG").is_ok() {
        println!(
            "DEBUG wanted {wanted:?} feeds {:?} got {:?}",
            feeds.iter().map(|f| f.len()).collect::<Vec<_>>(),
            got.iter().map(|g| &g[..g.len().min(6)]).collect::<Vec<_>>()
        );
        for (n, g) in &import.value_to_node {
            println!(
                "DEBUG value {n} -> graph {g} -> dag {:?}",
                graph_to_dag.get(*g)
            );
        }
    }
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
        match compare_output(o, g) {
            Ok(d) => worst = worst.max(d),
            Err(e) => return Ok(Outcome::Mismatch(format!("output '{}': {e}", o.name))),
        }
    }
    Ok(Outcome::Pass(worst))
}

/// Compares one output (read back as raw words) with its reference. Floats pass when
/// every element is within `max(1e-4, 1e-5 max|want|) + 1e-3 |want|` (NaN where the reference has NaN, the
/// same infinity where it has one), returning the worst relative difference; integers
/// and bools must match byte for byte.
fn compare_output(o: &Expected, got: &[f32]) -> Result<f32, String> {
    let ours: Vec<u8> = got.iter().flat_map(|w| w.to_le_bytes()).collect();
    let n = o.bytes.len();
    let Some(ours) = ours.get(..n) else {
        return Err(format!("{} bytes read back, {n} expected", ours.len()));
    };
    // The absolute part scales with the tensor's largest value, so a deep model's
    // small logits are held to the accuracy its large ones set, while the conformance
    // suite's unit-sized values keep the 1e-4 floor.
    let mut atol = 1e-4f64;
    let close = |g: f64, w: f64, atol: f64| {
        if w.is_nan() {
            g.is_nan()
        } else if w.is_infinite() {
            g == w
        } else {
            (g - w).abs() <= atol + 1e-3 * w.abs()
        }
    };
    let floats: Option<(Vec<f64>, Vec<f64>)> = match o.dtype {
        DtypeRepr::F32 => Some((
            ours.chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect(),
            o.bytes
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect(),
        )),
        DtypeRepr::F64 => Some((
            ours.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            o.bytes
                .chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )),
        _ => None,
    };
    match floats {
        Some((g, w)) => {
            let scale = w
                .iter()
                .filter(|v| v.is_finite())
                .fold(0.0f64, |m, v| m.max(v.abs()))
                .max(1e-6);
            atol = atol.max(1e-5 * scale);
            let mut worst = 0.0f64;
            for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
                if !close(a, b, atol) {
                    return Err(format!("element {i} is {a}, want {b}"));
                }
                if b.is_finite() {
                    worst = worst.max((a - b).abs() / scale);
                }
            }
            Ok(worst as f32)
        }
        None if ours == o.bytes.as_slice() => Ok(0.0),
        None => {
            let i = ours
                .iter()
                .zip(&o.bytes)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            Err(format!("{:?} bytes differ from byte {i}", o.dtype))
        }
    }
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

// ---------------------------------------------------------------------------
// ONNX backend conformance suite
// ---------------------------------------------------------------------------

/// The ONNX repository's backend test data, vendored with teeny-onnx.
fn conformance_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../support/teeny-onnx/onnx/onnx/backend/test/data")
}

/// `<prefix>_<i>.pb` files of a test data set, in index order.
fn numbered_pbs(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<(usize, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let p = e.ok()?.path();
            let stem = p.file_stem()?.to_str()?;
            let i: usize = stem.strip_prefix(prefix)?.strip_prefix('_')?.parse().ok()?;
            Some((i, p))
        })
        .collect();
    v.sort();
    v.into_iter().map(|(_, p)| p).collect()
}

/// One conformance test: its model, inputs by name and expected outputs.
fn conformance_case(dir: &Path) -> Result<Outcome, Outcome> {
    let data = dir.join("test_data_set_0");
    let read = |p: &Path| {
        let bytes = std::fs::read(p)
            .map_err(|e| Outcome::Failed("data", format!("{}: {e}", p.display())))?;
        teeny_onnx::read_tensor_proto(&bytes)
            .map_err(|e| Outcome::Failed("data", format!("{}: {e}", p.display())))
    };
    let mut inputs = BTreeMap::new();
    for p in numbered_pbs(&data, "input") {
        let (name, t) = read(&p)?;
        inputs.insert(name, t.bytes);
    }
    let mut outputs = Vec::new();
    for p in numbered_pbs(&data, "output") {
        let (name, t) = read(&p)?;
        outputs.push(Expected {
            name,
            dtype: t.dtype,
            bytes: t.bytes,
        });
    }
    try_onnx(&dir.join("model.onnx"), None, &inputs, &outputs, true)
}

/// Runs the ONNX backend node tests (one op per model, every attribute variant) and
/// reports which pass, per test and per op, with the stage and reason for the rest.
///
/// ONNX_CONFORMANCE_FILTER: comma-separated substrings of test names to run.
/// ONNX_CONFORMANCE_REPORT: where to write the per-test TSV (default: target dir).
/// ONNX_CONFORMANCE_STRICT: fail the test if any case fails.
#[test]
#[ignore = "runs the ONNX backend conformance suite (slow: ~1,800 models)"]
fn onnx_node_conformance() {
    let root = conformance_root().join("node");
    let filter: Vec<String> = std::env::var("ONNX_CONFORMANCE_FILTER")
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let mut tests: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("ONNX backend test data (git submodule support/teeny-onnx/onnx)")
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.join("model.onnx").exists())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy();
            filter.is_empty() || filter.iter().any(|f| n.contains(f.as_str()))
        })
        .collect();
    tests.sort();

    let report_path = std::env::var("ONNX_CONFORMANCE_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/onnx_conformance.tsv")
        });
    let mut rows = vec!["test\tresult\tstage\tdetail".to_string()];
    // Results per stage, and the first line of each failure reason with its count.
    let mut by_result: BTreeMap<String, usize> = BTreeMap::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let total = tests.len();
    for (i, dir) in tests.iter().enumerate() {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let outcome = match conformance_case(dir) {
            Ok(o) | Err(o) => o,
        };
        let (result, stage_name, detail) = match &outcome {
            Outcome::Pass(d) => ("PASS".to_string(), String::new(), format!("{d:.1e}")),
            Outcome::Mismatch(m) => ("MISMATCH".to_string(), "compare".to_string(), m.clone()),
            Outcome::Failed(st, e) => (
                "FAILED".to_string(),
                st.to_string(),
                e.lines().next().unwrap_or("").to_string(),
            ),
        };
        let key = if result == "PASS" {
            "PASS".to_string()
        } else {
            format!("{result} at {stage_name}")
        };
        *by_result.entry(key).or_default() += 1;
        if result != "PASS" {
            // Group reasons with numbers and quoted names stripped, so one gap counts once.
            let general: String = detail
                .split(['\'', '"'])
                .enumerate()
                .filter(|(k, _)| k % 2 == 0)
                .map(|(_, p)| p)
                .collect::<String>()
                .chars()
                .map(|c| if c.is_ascii_digit() { '#' } else { c })
                .take(140)
                .collect();
            *reasons
                .entry(format!("{stage_name}: {general}"))
                .or_default() += 1;
        }
        rows.push(format!(
            "{name}\t{result}\t{stage_name}\t{}",
            detail.replace('\t', " ")
        ));
        if (i + 1) % 100 == 0 {
            println!("  {}/{total}", i + 1);
        }
    }
    let _ = std::fs::write(&report_path, rows.join("\n") + "\n");

    let passed = by_result.get("PASS").copied().unwrap_or(0);
    println!(
        "\n== ONNX node conformance on {}: {passed}/{total} pass ==",
        teeny_runtime::BACKEND_NAME
    );
    for (k, v) in &by_result {
        println!("  {v:>5}  {k}");
    }
    println!("\n  most common reasons:");
    let mut top: Vec<_> = reasons.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    for (r, n) in top.iter().take(40) {
        println!("  {n:>5}  {r}");
    }
    println!("\n  per-test report: {}", report_path.display());
    if std::env::var("ONNX_CONFORMANCE_STRICT").is_ok() {
        assert_eq!(
            passed,
            total,
            "{} of {total} conformance tests failed",
            total - passed
        );
    }
}
