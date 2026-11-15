//! Constant and shape materialisation for imported ONNX graphs.
//!
//! Exported models carry two very different kinds of constant. A handful are real
//! weights, which can run to gigabytes and must stay where they are. The rest is
//! *shape plumbing*: thousands of tiny `Shape`/`Gather`/`Concat`/`Unsqueeze` chains
//! that an exporter emits to compute a reshape target or a broadcast size. Those
//! chains depend on nothing but the model itself, so they have a single answer, but
//! a lowering pass that meets them one node at a time cannot see it — it only sees
//! ops whose operands are not known until "runtime".
//!
//! This pass evaluates those chains ahead of lowering and rewrites each one to an
//! [`Op::Constant`] holding its value, which is what `Op::Constant`'s own
//! documentation asks for. Two properties keep it cheap:
//!
//! * It works entirely in memory on a [`Graph`], so a model that cannot be written
//!   back out through protobuf's 2 GB ceiling is not a problem.
//! * It only ever decodes tensors of at most [`MaterializeConfig::max_elements`]
//!   elements, so the weights are referenced but never touched. Folding stays
//!   proportional to the shape plumbing, not to the model size.
//!
//! Anything it cannot evaluate it leaves exactly as it was and counts in
//! [`MaterializeReport::unsupported`], so the pass is always safe to run and the
//! report says what the next op worth supporting is.
//!
//! # Shapes
//!
//! The pass also works out node shapes, in the same walk, because constant folding
//! and shape inference each need the other: `Shape` needs its operand's shape before
//! it can fold, and `Reshape` needs its target operand's *value* before its own shape
//! is known. Run separately, whichever goes first stalls. That is why ONNX's own
//! `shape_inference.infer_shapes_path` leaves thousands of symbolic dimensions on an
//! exported model: it cannot evaluate the shape arithmetic the exporter emitted.
//!
//! It is still worth running first, and cheap (3 seconds on a 1.75 GB model, where the
//! in-memory `infer_shapes` cannot run at all because of protobuf's 2 GB ceiling):
//!
//! ```text
//! onnx.shape_inference.infer_shapes_path(src, dst, strict_mode=False, data_prop=True)
//! ```
//!
//! It resolves the shapes that need no folding, which this pass then builds on. On the
//! FoundationStereo 480x640 export the two together take the nodes carrying a dynamic
//! dimension from 6,258 to 669, and the `Shape` nodes this pass must refuse from 883
//! to 3.
//!
//! Shape refinement only ever replaces a shape that has an unknown dimension, so it
//! cannot degrade one the model already stated. Where the op's fields and the operands
//! known so far do not determine the answer it declines rather than inventing one: a
//! `Reshape` whose target operand did not fold, for instance, has no shape to give.

use std::collections::{BTreeMap, BTreeSet};

use teeny_core::graph::{ConstantMap, DtypeRepr, Graph, Op, Shape};

use crate::onnx::TensorData;

/// Default element ceiling for a foldable tensor.
///
/// Shape plumbing deals in rank-0 and rank-1 tensors of a few dozen elements. The
/// default leaves generous headroom for things like a flattened position-embedding
/// index while staying far below any real weight.
pub const DEFAULT_MAX_ELEMENTS: usize = 65_536;

/// Tuning for [`materialize_constants`].
#[derive(Debug, Clone)]
pub struct MaterializeConfig {
    /// Largest tensor, in elements, the pass will decode or produce. A node whose
    /// value would exceed this is left alone and counted in
    /// [`MaterializeReport::skipped_too_large`].
    pub max_elements: usize,
}

impl Default for MaterializeConfig {
    fn default() -> Self {
        Self {
            max_elements: DEFAULT_MAX_ELEMENTS,
        }
    }
}

/// What [`materialize_constants`] did.
#[derive(Debug, Default, Clone)]
pub struct MaterializeReport {
    /// Nodes rewritten to [`Op::Constant`].
    pub folded: usize,
    /// Nodes left alone because their value would exceed the element ceiling.
    pub skipped_too_large: usize,
    /// Nodes left alone because the pass cannot evaluate the op, keyed by a
    /// human-readable op name with the number of occurrences.
    pub unsupported: BTreeMap<String, usize>,
    /// `Shape` nodes declined because the operand's rank is not known.
    ///
    /// [`Shape`] cannot tell a genuine rank-0 tensor from a tensor whose shape the
    /// importer never inferred: both are the empty vector. Folding the latter would
    /// silently produce an empty shape vector and corrupt everything downstream, so
    /// the pass refuses. A non-zero count here means the importer's shape coverage,
    /// not this pass, is what limits how much folds.
    pub shape_of_unknown_rank: usize,
    /// Reductions whose axes were moved from an operand onto the op. See
    /// `adopt_reduce_axes`.
    pub reduce_axes_adopted: usize,
    /// Cumulative ops whose single axis was moved from an operand onto the op. See
    /// `adopt_axis_operands`.
    pub axis_operands_adopted: usize,
    /// Nodes whose output shape the pass worked out, having been imported with an
    /// unknown rank or a dynamic dimension.
    ///
    /// Constant folding and shape inference each need the other, so running them
    /// separately stalls whichever goes first. This counts what interleaving them
    /// recovered.
    pub shapes_refined: usize,
    /// Nodes still carrying a dynamic dimension after the pass.
    ///
    /// Counts only shapes with an explicit `None` dimension. An empty shape is *not*
    /// counted, because `Shape` cannot distinguish a genuine rank-0 scalar from a rank
    /// that was never established, and a real model has many of the former.
    pub shapes_unresolved: usize,
    /// Nodes whose recorded shape disagrees with what this pass infers from the op's
    /// own semantics.
    ///
    /// The recorded one is kept, since where it came from the file it is authoritative
    /// and this pass is not. But a non-zero count means something upstream recorded a
    /// shape it could not know, so it is reported rather than swallowed.
    pub shapes_conflicting: usize,
    /// Nodes declined because an index or operand was out of range for the shapes
    /// the pass was given.
    ///
    /// On a model whose shapes are fully inferred this should be zero. A non-zero
    /// count means some folded value disagrees with a consumer about a tensor's
    /// real shape, which again points at importer shape coverage.
    pub declined_out_of_range: usize,
}

impl MaterializeReport {
    /// The unsupported ops, most frequent first. Useful for deciding what to
    /// implement next.
    pub fn unsupported_ranked(&self) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = self
            .unsupported
            .iter()
            .map(|(k, n)| (k.as_str(), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

/// Evaluates every foldable constant/shape subgraph in `graph` and rewrites it in
/// place to an [`Op::Constant`].
///
/// `tensors` is the value map from [`crate::Onnx::from_path_with_tensors`]: it seeds
/// the pass with initializer and `Constant` values. Tensors above the element
/// ceiling are skipped rather than decoded, so passing the full map of a multi-
/// gigabyte model is safe.
///
/// Returns a [`ConstantMap`] of the newly folded nodes, ready to hand to lowering,
/// alongside the report. `Op::Input` nodes are never rewritten: an initializer stays an input so
/// its weights can still be bound by name.
pub fn materialize_constants(
    graph: &mut Graph,
    tensors: &BTreeMap<usize, TensorData>,
    config: &MaterializeConfig,
) -> (ConstantMap, MaterializeReport) {
    let mut report = MaterializeReport::default();
    let mut values: BTreeMap<usize, Value> = BTreeMap::new();

    // Seed from the model's own tensors, skipping anything large. This is the step
    // that keeps the pass away from the weights.
    //
    // A `ConstantOfShape`'s seeded tensor is its scalar *fill value*, not its
    // output, so it goes in a separate table. Seeding it as a value would make the
    // node look already-known, and a scalar would stand in for the whole tensor.
    let mut fills: BTreeMap<usize, Value> = BTreeMap::new();
    for (&id, td) in tensors {
        if elem_count(&td.shape).is_none_or(|n| n > config.max_elements) {
            continue;
        }
        let Some(v) = Value::decode(td) else {
            continue;
        };
        if matches!(
            graph.nodes.get(id).map(|n| &n.op),
            Some(Op::ConstantOfShape { .. })
        ) {
            fills.insert(id, v);
        } else {
            values.insert(id, v);
        }
    }

    let order = graph.topological_sort();
    let mut folded: ConstantMap = ConstantMap::new();

    // Which nodes' ranks are actually known.
    //
    // `Shape` cannot say "unknown rank": an empty shape means both a rank-0 scalar and
    // a shape nobody has worked out (teenygrad#129). Collecting one into an
    // `Option<Vec<usize>>` yields `Some(vec![])`, so an operand nothing has resolved
    // arrives at inference looking like a scalar. For a broadcasting op that is not a
    // harmless imprecision, it is a wrong answer: broadcasting an unresolved operand
    // against a rank-1 bias of [3072] returns [3072], which is complete, static and
    // wrong, and because it is complete no later round revisits it. That single
    // confusion accounted for every missing shape in the transformer branch of the
    // FoundationStereo export.
    //
    // So rank-known is tracked beside the shapes. A node whose shape came from the
    // file is authoritative even when empty, because the file stated its rank: a
    // rank-0 initializer has no dims. Anything else is known only once this pass has
    // worked it out.
    let mut rank_known: BTreeSet<usize> = BTreeSet::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        let from_file = matches!(
            node.op,
            Op::Input | Op::Constant { .. } | Op::ConstantOfShape { .. }
        );
        if from_file || !node.shape.is_empty() {
            rank_known.insert(id);
        }
    }

    for id in order {
        if values.contains_key(&id) {
            // Already a known constant (an initializer or a `Constant` node). Only
            // record a rewrite for `Constant`, whose value now lives in the op.
            continue;
        }

        let node = &graph.nodes[id];
        if matches!(node.op, Op::Input) {
            continue;
        }

        // Work out this node's own shape before anything reads it. Operands are
        // earlier in topological order, so their shapes are already refined and their
        // values already folded, which is exactly what the shape-changing ops need:
        // `Reshape` reads its target operand's value, `Expand` its shape operand's.
        // Doing this here is what breaks the cycle that leaves a separate shape-
        // inference pass with thousands of unresolved dimensions.
        {
            let in_static: Vec<Option<Vec<usize>>> = graph.nodes[id]
                .inputs
                .iter()
                .map(|&i| {
                    let shape = &graph.nodes[i].shape;
                    // An empty shape is a rank-0 scalar only if something established
                    // that; otherwise its rank is simply not known yet, and saying so
                    // is what keeps a broadcast from inventing an answer.
                    if shape.is_empty() && !rank_known.contains(&i) {
                        return None;
                    }
                    shape.iter().copied().collect()
                })
                .collect();
            let in_values: Vec<Option<&Value>> = graph.nodes[id]
                .inputs
                .iter()
                .map(|i| values.get(i))
                .collect();
            let needs_shape = graph.nodes[id].shape.is_empty()
                || graph.nodes[id].shape.iter().any(Option::is_none);
            if let Some(inferred) = infer_static_shape(&graph.nodes[id].op, &in_static, &in_values)
            {
                if needs_shape {
                    graph.nodes[id].shape = inferred.into_iter().map(Some).collect();
                    report.shapes_refined += 1;
                    rank_known.insert(id);
                } else if graph.nodes[id]
                    .shape
                    .iter()
                    .copied()
                    .collect::<Option<Vec<usize>>>()
                    .is_some_and(|recorded| recorded != inferred)
                {
                    // The op's own semantics and the shape already recorded disagree.
                    // The recorded one is kept, because where it came from the file it
                    // is authoritative and this pass is not; but a non-zero count here
                    // means something upstream is manufacturing shapes, so it is
                    // reported rather than swallowed.
                    report.shapes_conflicting += 1;
                }
            }
        }

        let node = &graph.nodes[id];
        let input_shapes: Vec<&Shape> =
            node.inputs.iter().map(|&i| &graph.nodes[i].shape).collect();

        // `Shape` reads its operand's *shape*, not its value, so it folds over a
        // live activation. It is the root of nearly every plumbing chain, so this
        // case has to come before the all-operands-known test below.
        let args: Vec<&Value> = if matches!(node.op, Op::ShapeOf { .. }) {
            Vec::new()
        } else {
            // Every other op needs each operand's value to have a single answer.
            match node
                .inputs
                .iter()
                .map(|i| values.get(i))
                .collect::<Option<Vec<&Value>>>()
            {
                Some(args) => args,
                None => continue,
            }
        };

        let fill = fills.get(&id).filter(|f| f.len() > 0);
        match eval(&node.op, &args, &input_shapes, fill, config) {
            Eval::Ok(v) => {
                if v.len() > config.max_elements {
                    report.skipped_too_large += 1;
                    continue;
                }
                let td = v.encode();
                let shape: Shape = v.shape.iter().map(|&d| Some(d)).collect();
                let dtype = v.dtype;
                values.insert(id, v);
                folded.insert(id, td);

                let node = &mut graph.nodes[id];
                node.op = Op::Constant {
                    dtype,
                    shape: shape.clone(),
                };
                node.inputs.clear();
                node.dtype = dtype;
                node.shape = shape;
                report.folded += 1;
            }
            Eval::TooLarge => report.skipped_too_large += 1,
            Eval::UnknownRank => report.shape_of_unknown_rank += 1,
            Eval::OutOfRange => report.declined_out_of_range += 1,
            Eval::Unsupported => {
                *report.unsupported.entry(op_label(&node.op)).or_insert(0) += 1;
            }
        }
    }

    report.reduce_axes_adopted = adopt_reduce_axes(graph, &values);
    report.axis_operands_adopted = adopt_axis_operands(graph, &values);

    report.shapes_unresolved = graph
        .nodes
        .iter()
        .filter(|n| n.shape.iter().any(Option::is_none))
        .count();

    (folded, report)
}

/// Moves a reduction's axes from its operand onto the op, where everything downstream
/// reads them.
///
/// ONNX kept `axes` as an attribute until opset 13 for `ReduceSum` and opset 18 for the
/// rest, and as an operand after. The importer can only read the attribute form: an
/// operand is often a `Constant` node, whose value this pass is what computes. Left
/// alone, the op's `axes` stays empty, and an empty list means *every* axis unless
/// `noop_with_empty_axes` says otherwise, so a reduction over one axis silently becomes
/// a reduction over all of them. The FoundationStereo export is exactly that: all 10 of
/// its `ReduceSum` nodes carry an operand of `[2]` and no attribute.
///
/// The operand stays in `inputs`, where `Op::shape_operand_positions` marks it as
/// metadata so lowering elides it rather than allocating for it. Returns how many ops
/// were rewritten.
fn adopt_reduce_axes(graph: &mut Graph, values: &BTreeMap<usize, Value>) -> usize {
    let mut adopted = 0usize;
    for id in 0..graph.nodes.len() {
        let Some(&operand) = graph.nodes[id].inputs.get(1) else {
            continue;
        };
        // Only when the op has no axes of its own: an attribute, if the file wrote one,
        // is what it means.
        let has_axes = match reduce_axes_mut(&mut graph.nodes[id].op) {
            Some(axes) => !axes.is_empty(),
            None => continue,
        };
        if has_axes {
            continue;
        }
        let Some(value) = values.get(&operand) else {
            continue;
        };
        // A float axes operand would be a malformed file; `as_ints` on one would
        // truncate silently, so an integer dtype is required.
        if !matches!(value.data, Data::Int(_)) {
            continue;
        }
        let read = value.as_ints();
        if let Some(axes) = reduce_axes_mut(&mut graph.nodes[id].op) {
            *axes = read;
            adopted += 1;
        }
    }
    adopted
}

/// Moves a cumulative op's axis from its operand onto the op.
///
/// ONNX has always carried `CumSum`'s axis as an operand rather than an attribute, so
/// the importer cannot see it: it is usually a `Constant` node, whose value this pass
/// is what computes. Left alone the op keeps the 0 the importer wrote, and the kernel
/// accumulates along the innermost axis whatever the file asked for. Returns how many
/// ops were rewritten.
fn adopt_axis_operands(graph: &mut Graph, values: &BTreeMap<usize, Value>) -> usize {
    let mut adopted = 0usize;
    for id in 0..graph.nodes.len() {
        let Some(&operand) = graph.nodes[id].inputs.get(1) else {
            continue;
        };
        if cum_axis_mut(&mut graph.nodes[id].op).is_none() {
            continue;
        }
        let Some(value) = values.get(&operand) else {
            continue;
        };
        if !matches!(value.data, Data::Int(_)) || value.len() != 1 {
            continue;
        }
        let read = value.int_at(0);
        if let Some(axis) = cum_axis_mut(&mut graph.nodes[id].op) {
            *axis = read;
            adopted += 1;
        }
    }
    adopted
}

/// A cumulative op's `axis` field, to be written.
fn cum_axis_mut(op: &mut Op) -> Option<&mut i64> {
    match op {
        Op::CumSum { axis, .. } | Op::CumProd { axis, .. } => Some(axis),
        _ => None,
    }
}

/// A reduction's `axes` field, to be written.
fn reduce_axes_mut(op: &mut Op) -> Option<&mut Vec<i64>> {
    match op {
        Op::ReduceSum { axes, .. }
        | Op::ReduceMean { axes, .. }
        | Op::ReduceMax { axes, .. }
        | Op::ReduceMin { axes, .. }
        | Op::ReduceProd { axes, .. }
        | Op::ReduceL1 { axes, .. }
        | Op::ReduceL2 { axes, .. }
        | Op::ReduceLogSum { axes, .. }
        | Op::ReduceLogSumExp { axes, .. }
        | Op::ReduceSumSquare { axes, .. } => Some(axes),
        _ => None,
    }
}

/// Collects every constant value that lowering may need, for
/// `TritonLowering::with_constants`.
///
/// Two maps hold constant values after an import-and-materialise, and lowering needs
/// both. `tensors`, from [`crate::Onnx::from_path_with_tensors`], holds the values the
/// model itself carries; `folded`, from [`materialize_constants`], holds the ones the
/// pass computed. Passing only the latter leaves lowering unable to find a literal
/// that was in the file all along, which is a confusing failure because the pass
/// reports success.
///
/// Only [`Op::Constant`] nodes are collected, so the weights are not copied: an
/// initializer is an [`Op::Input`] and is bound by name from a checkpoint instead.
pub fn lowering_constants(
    graph: &Graph,
    tensors: &ConstantMap,
    folded: &ConstantMap,
) -> ConstantMap {
    let mut out = ConstantMap::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, Op::Constant { .. }) {
            continue;
        }
        // A folded value supersedes the file's, since folding rewrote the node.
        if let Some(v) = folded.get(&id).or_else(|| tensors.get(&id)) {
            out.insert(id, v.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A materialised tensor, held in whichever of three lanes its dtype belongs to.
///
/// Integers use `i64` and floats `f64` so that every ONNX dtype the pass meets
/// round-trips without loss: shape arithmetic is exact in `i64`, and `f64` holds
/// every `f32` exactly.
#[derive(Debug, Clone, PartialEq)]
struct Value {
    dtype: DtypeRepr,
    shape: Vec<usize>,
    data: Data,
}

#[derive(Debug, Clone, PartialEq)]
enum Data {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
}

/// The outcome of trying to evaluate one node.
enum Eval {
    Ok(Value),
    /// Evaluable, but the result would exceed the element ceiling.
    TooLarge,
    /// The pass does not implement this op, or an operand combination it needs.
    Unsupported,
    /// A `Shape` whose operand's rank the IR does not pin down.
    UnknownRank,
    /// Evaluable in principle, but an index or operand was out of range.
    OutOfRange,
}

impl Value {
    fn len(&self) -> usize {
        match &self.data {
            Data::Int(v) => v.len(),
            Data::Float(v) => v.len(),
            Data::Bool(v) => v.len(),
        }
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Reads element `i` as an `i64`, for shape arithmetic. Floats truncate.
    fn int_at(&self, i: usize) -> i64 {
        match &self.data {
            Data::Int(v) => v[i],
            Data::Float(v) => v[i] as i64,
            Data::Bool(v) => v[i] as i64,
        }
    }

    fn float_at(&self, i: usize) -> f64 {
        match &self.data {
            Data::Int(v) => v[i] as f64,
            Data::Float(v) => v[i],
            Data::Bool(v) => v[i] as i64 as f64,
        }
    }

    fn bool_at(&self, i: usize) -> bool {
        match &self.data {
            Data::Int(v) => v[i] != 0,
            Data::Float(v) => v[i] != 0.0,
            Data::Bool(v) => v[i],
        }
    }

    /// The whole tensor as `i64`s. Shape vectors arrive this way.
    fn as_ints(&self) -> Vec<i64> {
        (0..self.len()).map(|i| self.int_at(i)).collect()
    }

    fn is_float(&self) -> bool {
        matches!(self.data, Data::Float(_))
    }

    /// Rebuilds a value with the same dtype and lane but a new shape, keeping the
    /// element order. Used by the reshaping ops, which never move data.
    fn with_shape(&self, shape: Vec<usize>) -> Value {
        Value {
            dtype: self.dtype,
            shape,
            data: self.data.clone(),
        }
    }

    /// Picks the listed flat source indices out of this value, into `shape`.
    fn gather_flat(&self, idx: &[usize], shape: Vec<usize>) -> Value {
        let data = match &self.data {
            Data::Int(v) => Data::Int(idx.iter().map(|&i| v[i]).collect()),
            Data::Float(v) => Data::Float(idx.iter().map(|&i| v[i]).collect()),
            Data::Bool(v) => Data::Bool(idx.iter().map(|&i| v[i]).collect()),
        };
        Value {
            dtype: self.dtype,
            shape,
            data,
        }
    }

    fn decode(td: &TensorData) -> Option<Value> {
        let shape: Vec<usize> = td.shape.iter().copied().collect::<Option<Vec<usize>>>()?;
        let n = shape.iter().product::<usize>();
        let b = &td.bytes;

        macro_rules! ints {
            ($ty:ty, $w:expr) => {{
                if b.len() < n * $w {
                    return None;
                }
                Data::Int(
                    b.as_chunks::<$w>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| <$ty>::from_le_bytes(*c) as i64)
                        .collect(),
                )
            }};
        }

        let data = match td.dtype {
            DtypeRepr::Bool => {
                if b.len() < n {
                    return None;
                }
                Data::Bool(b.iter().take(n).map(|&x| x != 0).collect())
            }
            DtypeRepr::I8 => ints!(i8, 1),
            DtypeRepr::U8 => ints!(u8, 1),
            DtypeRepr::I16 => ints!(i16, 2),
            DtypeRepr::U16 => ints!(u16, 2),
            DtypeRepr::I32 => ints!(i32, 4),
            DtypeRepr::U32 => ints!(u32, 4),
            DtypeRepr::I64 => ints!(i64, 8),
            DtypeRepr::U64 => {
                if b.len() < n * 8 {
                    return None;
                }
                Data::Int(
                    b.as_chunks::<8>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| u64::from_le_bytes(*c) as i64)
                        .collect(),
                )
            }
            DtypeRepr::F32 => {
                if b.len() < n * 4 {
                    return None;
                }
                Data::Float(
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| f32::from_le_bytes(*c) as f64)
                        .collect(),
                )
            }
            DtypeRepr::F64 => {
                if b.len() < n * 8 {
                    return None;
                }
                Data::Float(
                    b.as_chunks::<8>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| f64::from_le_bytes(*c))
                        .collect(),
                )
            }
            // 16-bit floats have no native Rust type here. Shape plumbing never uses
            // them, so leaving them unfolded costs nothing.
            DtypeRepr::F16 | DtypeRepr::BF16 => return None,
        };

        Some(Value {
            dtype: td.dtype,
            shape,
            data,
        })
    }

    fn encode(&self) -> TensorData {
        let mut bytes = Vec::new();
        match (&self.data, self.dtype) {
            (_, DtypeRepr::Bool) => {
                for i in 0..self.len() {
                    bytes.push(self.bool_at(i) as u8);
                }
            }
            (_, DtypeRepr::I8) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as i8).to_le_bytes());
                }
            }
            (_, DtypeRepr::U8) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u8).to_le_bytes());
                }
            }
            (_, DtypeRepr::I16) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as i16).to_le_bytes());
                }
            }
            (_, DtypeRepr::U16) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u16).to_le_bytes());
                }
            }
            (_, DtypeRepr::I32) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as i32).to_le_bytes());
                }
            }
            (_, DtypeRepr::U32) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u32).to_le_bytes());
                }
            }
            (_, DtypeRepr::I64) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&self.int_at(i).to_le_bytes());
                }
            }
            (_, DtypeRepr::U64) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u64).to_le_bytes());
                }
            }
            (_, DtypeRepr::F32) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.float_at(i) as f32).to_le_bytes());
                }
            }
            (_, DtypeRepr::F64) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&self.float_at(i).to_le_bytes());
                }
            }
            (_, DtypeRepr::F16 | DtypeRepr::BF16) => {}
        }
        TensorData {
            dtype: self.dtype,
            shape: self.shape.iter().map(|&d| Some(d)).collect(),
            bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

fn elem_count(shape: &Shape) -> Option<usize> {
    shape
        .iter()
        .copied()
        .collect::<Option<Vec<usize>>>()
        .map(|d| d.iter().product())
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Right-aligned numpy/ONNX broadcasting of two shapes.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = dim_from_right(a, rank, i);
        let db = dim_from_right(b, rank, i);
        out.push(match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        });
    }
    Some(out)
}

fn dim_from_right(shape: &[usize], rank: usize, i: usize) -> usize {
    let pad = rank - shape.len();
    if i < pad { 1 } else { shape[i - pad] }
}

/// Maps a coordinate in the broadcast output back to a flat index in `shape`.
fn broadcast_offset(coords: &[usize], shape: &[usize]) -> usize {
    let rank = coords.len();
    let strides = contiguous_strides(shape);
    let pad = rank - shape.len();
    let mut off = 0;
    for (ax, &d) in shape.iter().enumerate() {
        // A size-1 axis is stretched: every output coordinate reads index 0.
        if d != 1 {
            off += coords[ax + pad] * strides[ax];
        }
    }
    off
}

fn coords_of(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = contiguous_strides(shape);
    let mut c = vec![0usize; shape.len()];
    for ax in 0..shape.len() {
        c[ax] = flat / strides[ax];
        flat -= c[ax] * strides[ax];
    }
    c
}

/// Normalises a possibly-negative axis against `rank`.
fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..=r).contains(&a).then_some(a as usize)
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn eval(
    op: &Op,
    args: &[&Value],
    input_shapes: &[&Shape],
    fill: Option<&Value>,
    config: &MaterializeConfig,
) -> Eval {
    match op {
        // --- plumbing sources ---
        Op::ShapeOf { start, end } => eval_shape_of(*start, *end, input_shapes),
        Op::ConstantOfShape { dtype, .. } => eval_constant_of_shape(*dtype, args, fill, config),
        Op::Range => eval_range(args, config),

        // --- pure reshapes: element order never changes ---
        Op::Identity => one(args).map_or(Eval::Unsupported, |v| Eval::Ok(v.clone())),
        Op::Reshape => eval_reshape(args),
        Op::Squeeze { axes } => eval_squeeze(args, axes),
        Op::Unsqueeze { axes } => eval_unsqueeze(args, axes),

        // --- data movement ---
        Op::Concat { axis } => eval_concat(args, *axis, config),
        Op::Gather { axis } => eval_gather(args, *axis, config),
        Op::GatherND { batch_dims } => eval_gather_nd(args, *batch_dims, config),
        Op::GatherElements { axis } => eval_gather_elements(args, *axis, config),
        Op::Slice => eval_slice(args),
        Op::Transpose { perm } => eval_transpose(args, perm),
        Op::Expand => eval_expand(args, config),

        // --- arithmetic and comparison ---
        Op::Add => binary(args, BinKind::Add),
        Op::Sub => binary(args, BinKind::Sub),
        Op::Mul => binary(args, BinKind::Mul),
        Op::Div => binary(args, BinKind::Div),
        Op::Pow => binary(args, BinKind::Pow),
        Op::Mod { fmod } => binary(args, BinKind::Mod { fmod: *fmod }),
        Op::Equal => binary(args, BinKind::Equal),
        Op::Greater => binary(args, BinKind::Greater),
        Op::GreaterOrEqual => binary(args, BinKind::GreaterOrEqual),
        Op::Less => binary(args, BinKind::Less),
        Op::LessOrEqual => binary(args, BinKind::LessOrEqual),
        Op::NonZero => eval_nonzero(args, config),
        Op::And => binary(args, BinKind::And),
        Op::Or => binary(args, BinKind::Or),
        Op::Xor => binary(args, BinKind::Xor),
        Op::Where => eval_where(args),
        Op::Neg => unary(args, UnKind::Neg),
        Op::Abs => unary(args, UnKind::Abs),
        Op::Sqrt => unary(args, UnKind::Sqrt),
        Op::Floor => unary(args, UnKind::Floor),
        Op::Ceil => unary(args, UnKind::Ceil),
        Op::Not => unary(args, UnKind::Not),
        Op::Cast { to } => eval_cast(args, *to),

        _ => Eval::Unsupported,
    }
}

fn one<'a>(args: &[&'a Value]) -> Option<&'a Value> {
    (args.len() == 1).then(|| args[0])
}

fn eval_shape_of(start: i64, end: i64, input_shapes: &[&Shape]) -> Eval {
    let Some(shape) = input_shapes.first() else {
        return Eval::Unsupported;
    };
    if shape.is_empty() {
        // Ambiguous: either a real rank-0 tensor or a shape the importer never
        // inferred. See `MaterializeReport::shape_of_unknown_rank`.
        return Eval::UnknownRank;
    }
    let rank = shape.len() as i64;
    // ONNX clamps both ends into range; `end` defaults to the rank.
    let lo = (if start < 0 { start + rank } else { start }).clamp(0, rank) as usize;
    let hi = (if end < 0 { end + rank } else { end }).clamp(0, rank) as usize;
    if lo > hi {
        return Eval::Unsupported;
    }
    // A dynamic dim has no single answer, so the whole `Shape` stays unfolded.
    let Some(dims) = shape[lo..hi]
        .iter()
        .copied()
        .collect::<Option<Vec<usize>>>()
    else {
        return Eval::Unsupported;
    };
    Eval::Ok(Value {
        dtype: DtypeRepr::I64,
        shape: vec![dims.len()],
        data: Data::Int(dims.iter().map(|&d| d as i64).collect()),
    })
}

fn eval_constant_of_shape(
    dtype: DtypeRepr,
    args: &[&Value],
    fill: Option<&Value>,
    config: &MaterializeConfig,
) -> Eval {
    // args[0] is the target shape. `fill` is the node's `value` attribute, a scalar;
    // ONNX defaults it to zero when the attribute is absent.
    let Some(shape_v) = args.first() else {
        return Eval::Unsupported;
    };
    let dims = shape_v.as_ints();
    if dims.iter().any(|&d| d < 0) {
        return Eval::Unsupported;
    }
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    if n > config.max_elements {
        return Eval::TooLarge;
    }
    let data = match dtype {
        DtypeRepr::F32 | DtypeRepr::F64 => {
            Data::Float(vec![fill.map_or(0.0, |f| f.float_at(0)); n])
        }
        DtypeRepr::Bool => Data::Bool(vec![fill.is_some_and(|f| f.bool_at(0)); n]),
        DtypeRepr::F16 | DtypeRepr::BF16 => return Eval::Unsupported,
        _ => Data::Int(vec![fill.map_or(0, |f| f.int_at(0)); n]),
    };
    Eval::Ok(Value { dtype, shape, data })
}

/// Where a tensor's non-zero elements are: one row per axis, one column per element
/// found, as ONNX's `NonZero` defines it.
///
/// The output's width is the number of non-zero elements, which no shape inference can
/// work out, so a node like this one is why the frontier of unresolved shapes exists at
/// all. What makes it resolvable here is the operand being known: when it is, the count
/// is a fact about the data rather than about the shapes, and evaluating the node
/// removes a hole that would otherwise spread to everything downstream of it.
///
/// A rank-0 operand is declined. ONNX would give it a shape of `[0, n]`, a tensor with
/// no axes to report coordinates along, and nothing in an export has needed it.
fn eval_nonzero(args: &[&Value], config: &MaterializeConfig) -> Eval {
    let Some(v) = one(args) else {
        return Eval::Unsupported;
    };
    let rank = v.rank();
    if rank == 0 {
        return Eval::Unsupported;
    }

    // Row-major order, which is the order ONNX reports the coordinates in.
    let mut columns: Vec<Vec<usize>> = Vec::new();
    for i in 0..v.len() {
        // A bool reads as its byte, an int as itself and a float against zero, so
        // this is the same reading of "true" that the logical kernels use.
        if v.bool_at(i) {
            columns.push(coords_of(i, &v.shape));
        }
    }
    if rank * columns.len() > config.max_elements {
        return Eval::TooLarge;
    }

    let mut data = Vec::with_capacity(rank * columns.len());
    for axis in 0..rank {
        for c in &columns {
            data.push(c[axis] as i64);
        }
    }
    Eval::Ok(Value {
        dtype: DtypeRepr::I64,
        shape: vec![rank, columns.len()],
        data: Data::Int(data),
    })
}

fn eval_range(args: &[&Value], config: &MaterializeConfig) -> Eval {
    if args.len() != 3 || args.iter().any(|a| a.len() != 1) {
        return Eval::Unsupported;
    }
    let (start, limit, delta) = (args[0], args[1], args[2]);
    let dtype = start.dtype;
    if start.is_float() || limit.is_float() || delta.is_float() {
        let (s, l, d) = (start.float_at(0), limit.float_at(0), delta.float_at(0));
        if d == 0.0 {
            return Eval::Unsupported;
        }
        let n = ((l - s) / d).ceil().max(0.0);
        if !n.is_finite() || n as usize > config.max_elements {
            return Eval::TooLarge;
        }
        let n = n as usize;
        let vals: Vec<f64> = (0..n).map(|i| s + i as f64 * d).collect();
        return Eval::Ok(Value {
            dtype,
            shape: vec![n],
            data: Data::Float(vals),
        });
    }
    let (s, l, d) = (start.int_at(0), limit.int_at(0), delta.int_at(0));
    if d == 0 {
        return Eval::Unsupported;
    }
    // Ceiling division of the span by the step, clamped at zero.
    let span = l - s;
    let n = if (span > 0) == (d > 0) {
        ((span + d - d.signum()) / d).max(0)
    } else {
        0
    };
    if n as usize > config.max_elements {
        return Eval::TooLarge;
    }
    let vals: Vec<i64> = (0..n).map(|i| s + i * d).collect();
    Eval::Ok(Value {
        dtype,
        shape: vec![n as usize],
        data: Data::Int(vals),
    })
}

fn eval_reshape(args: &[&Value]) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, target) = (args[0], args[1]);
    let dims = target.as_ints();
    let n = data.len();

    let mut out: Vec<usize> = Vec::with_capacity(dims.len());
    let mut wildcard = None;
    for (ax, &d) in dims.iter().enumerate() {
        match d {
            // `0` means "keep the input's dim at this axis".
            0 => match data.shape.get(ax) {
                Some(&keep) => out.push(keep),
                None => return Eval::Unsupported,
            },
            -1 => {
                if wildcard.is_some() {
                    return Eval::Unsupported;
                }
                wildcard = Some(ax);
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            _ => return Eval::Unsupported,
        }
    }
    if let Some(ax) = wildcard {
        let known: usize = out.iter().product();
        if known == 0 || !n.is_multiple_of(known) {
            return Eval::Unsupported;
        }
        out[ax] = n / known;
    }
    if out.iter().product::<usize>() != n {
        return Eval::Unsupported;
    }
    Eval::Ok(data.with_shape(out))
}

fn eval_squeeze(args: &[&Value], axes: &[i64]) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let rank = v.rank();
    let mut drop = vec![false; rank];
    if axes.is_empty() {
        for (ax, &d) in v.shape.iter().enumerate() {
            drop[ax] = d == 1;
        }
    } else {
        for &a in axes {
            match norm_axis(a, rank) {
                Some(ax) if ax < rank && v.shape[ax] == 1 => drop[ax] = true,
                _ => return Eval::Unsupported,
            }
        }
    }
    let out: Vec<usize> = v
        .shape
        .iter()
        .enumerate()
        .filter(|(ax, _)| !drop[*ax])
        .map(|(_, &d)| d)
        .collect();
    Eval::Ok(v.with_shape(out))
}

fn eval_unsqueeze(args: &[&Value], axes: &[i64]) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    if axes.is_empty() {
        // An Unsqueeze with no axes is not an identity, it is an unrecorded one: see
        // the opset-13 axes-operand handling in `crate::onnx`. Folding it would pass
        // the value through and silently drop an axis.
        return Eval::Unsupported;
    }
    let out_rank = v.rank() + axes.len();
    let mut insert = vec![false; out_rank];
    for &a in axes {
        match norm_axis(a, out_rank) {
            Some(ax) if ax < out_rank && !insert[ax] => insert[ax] = true,
            _ => return Eval::Unsupported,
        }
    }
    let mut out = Vec::with_capacity(out_rank);
    let mut src = v.shape.iter();
    for &inserted in &insert {
        if inserted {
            out.push(1);
        } else {
            match src.next() {
                Some(&d) => out.push(d),
                None => return Eval::Unsupported,
            }
        }
    }
    Eval::Ok(v.with_shape(out))
}

fn eval_concat(args: &[&Value], axis: i64, config: &MaterializeConfig) -> Eval {
    if args.is_empty() {
        return Eval::Unsupported;
    }
    // Exporters build shape vectors by concatenating scalars, mixing rank-0 and
    // rank-1 operands in one node. Runtimes read a rank-0 operand as `[1]`; do the
    // same rather than decline, since this is most of the plumbing we are here for.
    let promoted: Vec<Value> = args
        .iter()
        .map(|a| {
            if a.rank() == 0 {
                a.with_shape(vec![1])
            } else {
                (*a).clone()
            }
        })
        .collect();
    let args: Vec<&Value> = promoted.iter().collect();
    let args = args.as_slice();

    let rank = args[0].rank();
    let Some(ax) = norm_axis(axis, rank).filter(|&ax| ax < rank) else {
        return Eval::Unsupported;
    };
    // Every operand must agree on every axis but the concatenated one.
    let mut out_shape = args[0].shape.clone();
    let mut concat_dim = 0;
    for a in args {
        if a.rank() != rank {
            return Eval::Unsupported;
        }
        for (i, (&d, &d0)) in a.shape.iter().zip(args[0].shape.iter()).enumerate() {
            if i != ax && d != d0 {
                return Eval::Unsupported;
            }
        }
        concat_dim += a.shape[ax];
    }
    out_shape[ax] = concat_dim;
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }

    // Walk the output and pull each element from whichever operand owns that slot.
    let mut offsets = Vec::with_capacity(args.len());
    let mut acc = 0;
    for a in args {
        offsets.push(acc);
        acc += a.shape[ax];
    }

    let mut ints = Vec::with_capacity(total);
    let mut floats = Vec::with_capacity(total);
    let mut bools = Vec::with_capacity(total);
    let float_out = args[0].is_float();
    let bool_out = matches!(args[0].data, Data::Bool(_));

    for flat in 0..total {
        let mut c = coords_of(flat, &out_shape);
        let which = offsets.iter().rposition(|&o| o <= c[ax]).unwrap_or(0);
        c[ax] -= offsets[which];
        let src = args[which];
        let off: usize = c
            .iter()
            .zip(contiguous_strides(&src.shape))
            .map(|(&ci, s)| ci * s)
            .sum();
        if bool_out {
            bools.push(src.bool_at(off));
        } else if float_out {
            floats.push(src.float_at(off));
        } else {
            ints.push(src.int_at(off));
        }
    }

    let data = if bool_out {
        Data::Bool(bools)
    } else if float_out {
        Data::Float(floats)
    } else {
        Data::Int(ints)
    };
    Eval::Ok(Value {
        dtype: args[0].dtype,
        shape: out_shape,
        data,
    })
}

/// The output shape of a `GatherND`, and the slice length each index addresses.
///
/// `indices`' last axis holds a coordinate of `k` numbers, which names a position in
/// `data` after the `batch_dims` leading axes the two share. Everything of `data` past
/// those `batch_dims + k` axes comes along as a slice, so the output is `indices`'
/// shape with its last axis replaced by that slice's shape.
///
/// `None` if the two shapes do not describe a `GatherND` at all: an empty `indices`
/// shape, a coordinate wider than the axes left to address, or batch axes that
/// disagree between the operands.
fn gather_nd_shape(
    data: &[usize],
    indices: &[usize],
    batch_dims: i64,
) -> Option<(Vec<usize>, usize)> {
    let b = usize::try_from(batch_dims).ok()?;
    let k = *indices.last()?;
    if b > data.len() || b + 1 > indices.len() || b + k > data.len() || k == 0 {
        return None;
    }
    if data[..b] != indices[..b] {
        return None;
    }
    let mut out: Vec<usize> = indices[..indices.len() - 1].to_vec();
    out.extend_from_slice(&data[b + k..]);
    let slice_len: usize = data[b + k..].iter().product();
    Some((out, slice_len))
}

/// `GatherND`: each row of `indices` names a position in `data`, and the rest of
/// `GatherElements(data, indices, axis)`: the output has the indices' shape, and each
/// element is `data` at its own position with the `axis` coordinate replaced by the
/// index there.
fn eval_gather_elements(args: &[&Value], axis: i64, config: &MaterializeConfig) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, indices) = (args[0], args[1]);
    let rank = data.rank();
    if indices.rank() != rank {
        return Eval::Unsupported;
    }
    let Some(ax) = norm_axis(axis, rank).filter(|&ax| ax < rank) else {
        return Eval::Unsupported;
    };
    let total: usize = indices.shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }
    let dim = data.shape[ax] as i64;
    let idx = indices.as_ints();
    // Row-major strides of both tensors.
    let strides = |shape: &[usize]| {
        let mut st = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            st[i] = st[i + 1] * shape[i + 1];
        }
        st
    };
    let (ds, is) = (strides(&data.shape), strides(&indices.shape));
    let mut flat = Vec::with_capacity(total);
    for (p, &raw) in idx.iter().enumerate() {
        let i = if raw < 0 { raw + dim } else { raw };
        if i < 0 || i >= dim {
            return Eval::OutOfRange;
        }
        let mut src = 0;
        for d in 0..rank {
            let coord = (p / is[d]) % indices.shape[d];
            if coord >= data.shape[d] && d != ax {
                return Eval::OutOfRange;
            }
            src += if d == ax { i as usize } else { coord } * ds[d];
        }
        flat.push(src);
    }
    Eval::Ok(data.gather_flat(&flat, indices.shape.clone()))
}

/// `data`'s axes come along as a slice.
fn eval_gather_nd(args: &[&Value], batch_dims: i64, config: &MaterializeConfig) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, indices) = (args[0], args[1]);
    let Some((out_shape, slice_len)) = gather_nd_shape(&data.shape, &indices.shape, batch_dims)
    else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }

    let b = batch_dims as usize;
    let k = *indices.shape.last().expect("checked by gather_nd_shape");
    let coord_shape = &indices.shape[..indices.shape.len() - 1];
    let n_coords: usize = coord_shape.iter().product();
    let strides = contiguous_strides(&data.shape);
    let idx = indices.as_ints();

    let mut flat = Vec::with_capacity(total);
    for c in 0..n_coords {
        // The leading `batch_dims` of this coordinate index `data`'s own batch axes;
        // the rest of the position comes from the index row.
        let coords = coords_of(c, coord_shape);
        let mut offset = 0usize;
        for (axis, &co) in coords.iter().take(b).enumerate() {
            offset += co * strides[axis];
        }
        for j in 0..k {
            let dim = data.shape[b + j] as i64;
            // ONNX counts a negative index from the end of its own axis.
            let i = idx[c * k + j];
            let i = if i < 0 { i + dim } else { i };
            if i < 0 || i >= dim {
                return Eval::OutOfRange;
            }
            offset += i as usize * strides[b + j];
        }
        for n in 0..slice_len {
            flat.push(offset + n);
        }
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_gather(args: &[&Value], axis: i64, config: &MaterializeConfig) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, indices) = (args[0], args[1]);
    let rank = data.rank();
    let Some(ax) = norm_axis(axis, rank).filter(|&ax| ax < rank) else {
        return Eval::Unsupported;
    };
    let dim = data.shape[ax] as i64;

    // Output replaces the gathered axis with the indices' own shape.
    let mut out_shape = Vec::with_capacity(rank - 1 + indices.rank());
    out_shape.extend_from_slice(&data.shape[..ax]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[ax + 1..]);
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }

    let idx: Vec<i64> = indices
        .as_ints()
        .iter()
        .map(|&i| if i < 0 { i + dim } else { i })
        .collect();
    if idx.iter().any(|&i| i < 0 || i >= dim) {
        return Eval::OutOfRange;
    }

    // `data` is contiguous, so the gathered axis sits at a stride of `inner` inside
    // each of the `outer` leading blocks.
    let outer: usize = data.shape[..ax].iter().product();
    let inner: usize = data.shape[ax + 1..].iter().product();
    let block = dim as usize * inner;

    let mut flat = Vec::with_capacity(total);
    for o in 0..outer {
        for &i in &idx {
            for n in 0..inner {
                flat.push(o * block + i as usize * inner + n);
            }
        }
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_slice(args: &[&Value]) -> Eval {
    if !(3..=5).contains(&args.len()) {
        return Eval::Unsupported;
    }
    let data = args[0];
    let rank = data.rank();
    let starts = args[1].as_ints();
    let ends = args[2].as_ints();
    let axes: Vec<i64> = match args.get(3) {
        Some(a) => a.as_ints(),
        None => (0..starts.len() as i64).collect(),
    };
    let steps: Vec<i64> = match args.get(4) {
        Some(a) => a.as_ints(),
        None => vec![1; starts.len()],
    };
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Eval::Unsupported;
    }

    // Per-axis index lists, defaulting to the full extent.
    let mut picks: Vec<Vec<usize>> = data
        .shape
        .iter()
        .map(|&d| (0..d).collect::<Vec<usize>>())
        .collect();

    for k in 0..starts.len() {
        let Some(ax) = norm_axis(axes[k], rank).filter(|&ax| ax < rank) else {
            return Eval::Unsupported;
        };
        let dim = data.shape[ax] as i64;
        let step = steps[k];
        if step == 0 {
            return Eval::Unsupported;
        }
        let mut s = if starts[k] < 0 {
            starts[k] + dim
        } else {
            starts[k]
        };
        let mut e = if ends[k] < 0 { ends[k] + dim } else { ends[k] };
        let mut out = Vec::new();
        if step > 0 {
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
            let mut i = s;
            while i < e {
                out.push(i as usize);
                i += step;
            }
        } else {
            // Reverse slicing: ONNX clamps the start one short of the end of the axis.
            s = s.clamp(-1, dim - 1);
            e = e.clamp(-1, dim - 1);
            let mut i = s;
            while i > e {
                if i < 0 {
                    break;
                }
                out.push(i as usize);
                i += step;
            }
        }
        picks[ax] = out;
    }

    let out_shape: Vec<usize> = picks.iter().map(|p| p.len()).collect();
    let total: usize = out_shape.iter().product();
    let strides = contiguous_strides(&data.shape);
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let off: usize = c
            .iter()
            .enumerate()
            .map(|(ax, &ci)| picks[ax][ci] * strides[ax])
            .sum();
        flat.push(off);
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_transpose(args: &[&Value], perm: &[usize]) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let rank = v.rank();
    if perm.len() != rank || perm.iter().any(|&p| p >= rank) {
        return Eval::Unsupported;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| v.shape[p]).collect();
    let src_strides = contiguous_strides(&v.shape);
    let total: usize = out_shape.iter().product();
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let off: usize = c
            .iter()
            .zip(perm.iter())
            .map(|(&ci, &p)| ci * src_strides[p])
            .sum();
        flat.push(off);
    }
    Eval::Ok(v.gather_flat(&flat, out_shape))
}

fn eval_expand(args: &[&Value], config: &MaterializeConfig) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, target) = (args[0], args[1]);
    let dims = target.as_ints();
    if dims.iter().any(|&d| d < 0) {
        return Eval::Unsupported;
    }
    let want: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    // `Expand` broadcasts both ways: the result can be wider than the target.
    let Some(out_shape) = broadcast_shapes(&data.shape, &want) else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        flat.push(broadcast_offset(&c, &data.shape));
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_cast(args: &[&Value], to: DtypeRepr) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let n = v.len();
    let data = match to {
        DtypeRepr::Bool => Data::Bool((0..n).map(|i| v.bool_at(i)).collect()),
        DtypeRepr::F32 => Data::Float((0..n).map(|i| v.float_at(i) as f32 as f64).collect()),
        DtypeRepr::F64 => Data::Float((0..n).map(|i| v.float_at(i)).collect()),
        DtypeRepr::F16 | DtypeRepr::BF16 => return Eval::Unsupported,
        // Integer casts truncate toward zero, as ONNX specifies, and then wrap to the
        // target width so the value round-trips through `encode`.
        _ => {
            let ints: Vec<i64> = (0..n)
                .map(|i| {
                    let x = if v.is_float() {
                        v.float_at(i).trunc() as i64
                    } else {
                        v.int_at(i)
                    };
                    narrow(x, to)
                })
                .collect();
            Data::Int(ints)
        }
    };
    Eval::Ok(Value {
        dtype: to,
        shape: v.shape.clone(),
        data,
    })
}

fn narrow(x: i64, to: DtypeRepr) -> i64 {
    match to {
        DtypeRepr::I8 => x as i8 as i64,
        DtypeRepr::U8 => x as u8 as i64,
        DtypeRepr::I16 => x as i16 as i64,
        DtypeRepr::U16 => x as u16 as i64,
        DtypeRepr::I32 => x as i32 as i64,
        DtypeRepr::U32 => x as u32 as i64,
        _ => x,
    }
}

#[derive(Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod { fmod: bool },
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    And,
    Or,
    Xor,
}

impl BinKind {
    fn is_comparison(self) -> bool {
        matches!(
            self,
            BinKind::Equal
                | BinKind::Greater
                | BinKind::GreaterOrEqual
                | BinKind::Less
                | BinKind::LessOrEqual
        )
    }

    /// Takes two booleans and returns one.
    fn is_logical(self) -> bool {
        matches!(self, BinKind::And | BinKind::Or | BinKind::Xor)
    }
}

fn binary(args: &[&Value], kind: BinKind) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (a, b) = (args[0], args[1]);
    let Some(out_shape) = broadcast_shapes(&a.shape, &b.shape) else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    let float_math = a.is_float() || b.is_float();

    let mut ints = Vec::new();
    let mut floats = Vec::new();
    let mut bools = Vec::new();

    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let ia = broadcast_offset(&c, &a.shape);
        let ib = broadcast_offset(&c, &b.shape);

        if kind.is_logical() {
            // ONNX defines these on bool tensors only, and a Value holds a bool as
            // one byte, so anything non-zero is true: the same reading the kernel
            // uses, so folding a node and running it agree.
            let (x, y) = (a.bool_at(ia), b.bool_at(ib));
            bools.push(match kind {
                BinKind::And => x && y,
                BinKind::Or => x || y,
                BinKind::Xor => x != y,
                _ => unreachable!("non-logical kind in logical branch"),
            });
            continue;
        }

        if kind.is_comparison() {
            let ord = if float_math {
                a.float_at(ia).partial_cmp(&b.float_at(ib))
            } else {
                Some(a.int_at(ia).cmp(&b.int_at(ib)))
            };
            let Some(ord) = ord else {
                return Eval::Unsupported; // NaN comparison; leave the node alone.
            };
            bools.push(match kind {
                BinKind::Equal => ord.is_eq(),
                BinKind::Greater => ord.is_gt(),
                BinKind::GreaterOrEqual => ord.is_ge(),
                BinKind::Less => ord.is_lt(),
                BinKind::LessOrEqual => ord.is_le(),
                _ => unreachable!("non-comparison kind in comparison branch"),
            });
            continue;
        }

        if float_math {
            let (x, y) = (a.float_at(ia), b.float_at(ib));
            floats.push(match kind {
                BinKind::Add => x + y,
                BinKind::Sub => x - y,
                BinKind::Mul => x * y,
                BinKind::Div => x / y,
                BinKind::Pow => x.powf(y),
                // ONNX `Mod` on floats requires fmod=1; its semantics are C's fmod.
                BinKind::Mod { .. } => x % y,
                _ => unreachable!("comparison kind in arithmetic branch"),
            });
        } else {
            let (x, y) = (a.int_at(ia), b.int_at(ib));
            let v = match kind {
                BinKind::Add => x.wrapping_add(y),
                BinKind::Sub => x.wrapping_sub(y),
                BinKind::Mul => x.wrapping_mul(y),
                BinKind::Div | BinKind::Mod { .. } if y == 0 => return Eval::Unsupported,
                // Integer division truncates toward zero in ONNX, as it does in Rust.
                BinKind::Div => x / y,
                BinKind::Pow => {
                    let Ok(e) = u32::try_from(y) else {
                        return Eval::Unsupported;
                    };
                    x.wrapping_pow(e)
                }
                // With fmod=0 the sign follows the divisor (Python); with 1 it follows
                // the dividend (C).
                BinKind::Mod { fmod } => {
                    if fmod {
                        x % y
                    } else {
                        let r = x % y;
                        if r != 0 && (r < 0) != (y < 0) { r + y } else { r }
                    }
                }
                _ => unreachable!("comparison kind in arithmetic branch"),
            };
            ints.push(narrow(v, a.dtype));
        }
    }

    let (dtype, data) = if kind.is_comparison() || kind.is_logical() {
        (DtypeRepr::Bool, Data::Bool(bools))
    } else if float_math {
        (
            if a.is_float() { a.dtype } else { b.dtype },
            Data::Float(floats),
        )
    } else {
        (a.dtype, Data::Int(ints))
    };
    Eval::Ok(Value {
        dtype,
        shape: out_shape,
        data,
    })
}

#[derive(Clone, Copy)]
enum UnKind {
    Neg,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Not,
}

fn unary(args: &[&Value], kind: UnKind) -> Eval {
    let Some(v) = one(args) else {
        return Eval::Unsupported;
    };
    let n = v.len();
    if matches!(kind, UnKind::Not) {
        return Eval::Ok(Value {
            dtype: DtypeRepr::Bool,
            shape: v.shape.clone(),
            data: Data::Bool((0..n).map(|i| !v.bool_at(i)).collect()),
        });
    }
    // `Sqrt` is float-only in ONNX; the rest keep the operand's lane so integer
    // shape arithmetic stays exact.
    if v.is_float() || matches!(kind, UnKind::Sqrt) {
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                let x = v.float_at(i);
                match kind {
                    UnKind::Neg => -x,
                    UnKind::Abs => x.abs(),
                    UnKind::Sqrt => x.sqrt(),
                    UnKind::Floor => x.floor(),
                    UnKind::Ceil => x.ceil(),
                    UnKind::Not => unreachable!("Not handled above"),
                }
            })
            .collect();
        if vals.iter().any(|x| x.is_nan()) {
            // A NaN here means a negative `Sqrt`; leave the node for the runtime.
            return Eval::Unsupported;
        }
        let dtype = if v.is_float() {
            v.dtype
        } else {
            DtypeRepr::F32
        };
        return Eval::Ok(Value {
            dtype,
            shape: v.shape.clone(),
            data: Data::Float(vals),
        });
    }
    let vals: Vec<i64> = (0..n)
        .map(|i| {
            let x = v.int_at(i);
            match kind {
                UnKind::Neg => x.wrapping_neg(),
                UnKind::Abs => x.wrapping_abs(),
                // Already integral.
                UnKind::Floor | UnKind::Ceil => x,
                UnKind::Sqrt | UnKind::Not => unreachable!("handled above"),
            }
        })
        .map(|x| narrow(x, v.dtype))
        .collect();
    Eval::Ok(Value {
        dtype: v.dtype,
        shape: v.shape.clone(),
        data: Data::Int(vals),
    })
}

fn eval_where(args: &[&Value]) -> Eval {
    if args.len() != 3 {
        return Eval::Unsupported;
    }
    let (cond, x, y) = (args[0], args[1], args[2]);
    let Some(s1) = broadcast_shapes(&cond.shape, &x.shape) else {
        return Eval::Unsupported;
    };
    let Some(out_shape) = broadcast_shapes(&s1, &y.shape) else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    let float_out = x.is_float() || y.is_float();
    let bool_out = matches!(x.data, Data::Bool(_)) && matches!(y.data, Data::Bool(_));

    let mut ints = Vec::new();
    let mut floats = Vec::new();
    let mut bools = Vec::new();
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let take_x = cond.bool_at(broadcast_offset(&c, &cond.shape));
        let (src, off) = if take_x {
            (x, broadcast_offset(&c, &x.shape))
        } else {
            (y, broadcast_offset(&c, &y.shape))
        };
        if bool_out {
            bools.push(src.bool_at(off));
        } else if float_out {
            floats.push(src.float_at(off));
        } else {
            ints.push(src.int_at(off));
        }
    }
    let data = if bool_out {
        Data::Bool(bools)
    } else if float_out {
        Data::Float(floats)
    } else {
        Data::Int(ints)
    };
    Eval::Ok(Value {
        dtype: x.dtype,
        shape: out_shape,
        data,
    })
}

/// A short, stable name for an op, for the unsupported-op tally.
fn op_label(op: &Op) -> String {
    let dbg = format!("{op:?}");
    // `Op`'s Debug prints either `Name` or `Name { .. }`; keep just the name.
    dbg.split([' ', '('])
        .next()
        .unwrap_or("Unknown")
        .to_string()
}

// ---------------------------------------------------------------------------
// Shape refinement
// ---------------------------------------------------------------------------

/// Infers a node's fully-static output shape from its operands' shapes and values.
///
/// This exists because constant folding and shape inference each need the other.
/// `Shape` needs its operand's shape before it can fold; `Reshape` needs its target
/// operand's *value* before its own shape is known. Running them as separate passes
/// stalls whichever goes first, which is what leaves ONNX's own
/// `infer_shapes_path` with thousands of symbolic dimensions on an exported model:
/// it cannot evaluate the shape arithmetic the exporter emitted. Interleaving them in
/// one topological walk resolves the cycle, because by the time a node is visited both
/// its operands' shapes and their values are already settled.
///
/// Returns `None` when the shape is not fully static, which includes every op this
/// does not implement. A `None` leaves the imported shape untouched, so the function
/// can only improve a shape, never degrade one.
fn infer_static_shape(
    op: &Op,
    in_shapes: &[Option<Vec<usize>>],
    vals: &[Option<&Value>],
) -> Option<Vec<usize>> {
    let first = || in_shapes.first().cloned().flatten();
    let shape_at = |i: usize| in_shapes.get(i).cloned().flatten();
    let val = |i: usize| vals.get(i).copied().flatten();
    let ints = |i: usize| val(i).map(|v| v.as_ints());

    match op {
        // --- shape unchanged ---
        Op::Relu
        | Op::Sigmoid
        | Op::Tanh
        | Op::Erf
        | Op::Softmax { .. }
        | Op::Clip { .. }
        | Op::Identity
        | Op::Neg
        | Op::Abs
        | Op::Sqrt
        | Op::Exp
        | Op::Log
        | Op::Sin
        | Op::Cos
        | Op::Floor
        | Op::Ceil
        | Op::Reciprocal
        | Op::Not
        | Op::Sign
        | Op::LeakyRelu { .. }
        | Op::Elu { .. }
        | Op::Selu
        | Op::Gelu
        | Op::GeluErf
        | Op::Mish
        | Op::Silu
        | Op::Swish
        | Op::Softsign
        | Op::Softplus { .. }
        | Op::PRelu
        | Op::ThresholdedRelu { .. }
        | Op::Shrink { .. }
        | Op::Cast { .. }
        | Op::CastLike
        | Op::BitCast { .. }
        | Op::ScatterND
        | Op::ScatterElements { .. }
        | Op::Trilu { .. }
        | Op::CumSum { .. }
        | Op::CumProd { .. }
        | Op::ReverseSequence { .. }
        // Element-wise activations and maths that keep their input's shape.
        | Op::Hardswish
        | Op::Hardsigmoid
        | Op::Hardtanh { .. }
        | Op::Relu6
        | Op::Hardshrink { .. }
        | Op::Softshrink { .. }
        | Op::Tanhshrink
        | Op::Logsigmoid
        | Op::Celu { .. }
        | Op::Round
        | Op::Tan
        | Op::Asin
        | Op::Acos
        | Op::Atan
        | Op::Sinh
        | Op::Cosh
        | Op::Asinh
        | Op::Acosh
        | Op::Atanh
        | Op::IsNaN
        | Op::IsInf { .. }
        | Op::BitwiseNot
        | Op::LogSoftmax { .. }
        | Op::Hardmax { .. }
        | Op::LpNormalization { .. }
        | Op::Dropout { .. } => first(),

        // Normalisation keeps its input's shape; the scale and bias operands are
        // per-channel and do not change it.
        // ONNX Flatten: [d0..d(axis-1), d(axis)..] -> [prod before axis, prod from axis].
        Op::Flatten { axis } => {
            let s = first()?;
            let axis = (*axis).min(s.len());
            Some(vec![s[..axis].iter().product(), s[axis..].iter().product()])
        }

        Op::BatchNorm1d { .. }
        | Op::BatchNorm2d { .. }
        | Op::BatchNorm3d { .. }
        | Op::InstanceNorm1d { .. }
        | Op::InstanceNorm2d { .. }
        | Op::InstanceNorm3d { .. }
        | Op::LayerNorm { .. }
        | Op::GroupNorm { .. } => first(),

        // --- broadcasting ---
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Pow
        | Op::Mod { .. }
        | Op::Equal
        | Op::Greater
        | Op::GreaterOrEqual
        | Op::Less
        | Op::LessOrEqual
        | Op::And
        | Op::Or
        | Op::Xor => {
            let a = in_shapes.first().cloned().flatten()?;
            let b = in_shapes.get(1).cloned().flatten()?;
            broadcast_shapes(&a, &b)
        }
        Op::Where => {
            let c = in_shapes.first().cloned().flatten()?;
            let x = in_shapes.get(1).cloned().flatten()?;
            let y = in_shapes.get(2).cloned().flatten()?;
            broadcast_shapes(&broadcast_shapes(&c, &x)?, &y)
        }

        // --- driven by an operand's value ---
        Op::ShapeOf { start, end } => {
            let rank = first()?.len() as i64;
            let lo = (if *start < 0 { start + rank } else { *start }).clamp(0, rank);
            let hi = (if *end < 0 { end + rank } else { *end }).clamp(0, rank);
            Some(vec![(hi - lo).max(0) as usize])
        }
        Op::SizeOf => Some(vec![]),
        Op::Reshape => {
            let input = first()?;
            let n: usize = input.iter().product();
            let dims = ints(1)?;
            let mut out = Vec::with_capacity(dims.len());
            let mut wildcard = None;
            for (ax, &d) in dims.iter().enumerate() {
                match d {
                    0 => out.push(*input.get(ax)?),
                    -1 => {
                        if wildcard.is_some() {
                            return None;
                        }
                        wildcard = Some(ax);
                        out.push(1);
                    }
                    d if d > 0 => out.push(d as usize),
                    _ => return None,
                }
            }
            if let Some(ax) = wildcard {
                let known: usize = out.iter().product();
                if known == 0 || !n.is_multiple_of(known) {
                    return None;
                }
                out[ax] = n / known;
            }
            (out.iter().product::<usize>() == n).then_some(out)
        }
        Op::Expand => {
            let input = first()?;
            let want: Vec<usize> = ints(1)?
                .iter()
                .map(|&d| usize::try_from(d).ok())
                .collect::<Option<_>>()?;
            broadcast_shapes(&input, &want)
        }
        Op::Unsqueeze { axes } => {
            let input = first()?;
            if axes.is_empty() {
                return None;
            }
            let out_rank = input.len() + axes.len();
            let mut insert = vec![false; out_rank];
            for &a in axes {
                let ax = norm_axis(a, out_rank)?;
                if ax >= out_rank || insert[ax] {
                    return None;
                }
                insert[ax] = true;
            }
            let mut out = Vec::with_capacity(out_rank);
            let mut src = input.iter();
            for &inserted in &insert {
                if inserted {
                    out.push(1);
                } else {
                    out.push(*src.next()?);
                }
            }
            Some(out)
        }
        Op::Squeeze { axes } => {
            let input = first()?;
            let mut drop = vec![false; input.len()];
            if axes.is_empty() {
                for (ax, &d) in input.iter().enumerate() {
                    drop[ax] = d == 1;
                }
            } else {
                for &a in axes {
                    let ax = norm_axis(a, input.len())?;
                    if ax >= input.len() || input[ax] != 1 {
                        return None;
                    }
                    drop[ax] = true;
                }
            }
            Some(
                input
                    .iter()
                    .enumerate()
                    .filter(|(ax, _)| !drop[*ax])
                    .map(|(_, &d)| d)
                    .collect(),
            )
        }
        Op::Concat { axis } => {
            let first_shape = first()?;
            let rank = first_shape.len();
            let ax = norm_axis(*axis, rank).filter(|&ax| ax < rank)?;
            let mut out = first_shape.clone();
            let mut total = 0usize;
            for s in in_shapes {
                let s = s.as_ref()?;
                if s.len() != rank {
                    return None;
                }
                for (i, (&d, &d0)) in s.iter().zip(first_shape.iter()).enumerate() {
                    if i != ax && d != d0 {
                        return None;
                    }
                }
                total += s[ax];
            }
            out[ax] = total;
            Some(out)
        }
        Op::Transpose { perm } => {
            let input = first()?;
            if perm.len() != input.len() || perm.iter().any(|&p| p >= input.len()) {
                return None;
            }
            Some(perm.iter().map(|&p| input[p]).collect())
        }
        Op::Gather { axis } => {
            let input = first()?;
            let rank = input.len();
            let ax = norm_axis(*axis, rank).filter(|&ax| ax < rank)?;
            let idx_shape = in_shapes.get(1).cloned().flatten()?;
            let mut out = Vec::with_capacity(rank - 1 + idx_shape.len());
            out.extend_from_slice(&input[..ax]);
            out.extend_from_slice(&idx_shape);
            out.extend_from_slice(&input[ax + 1..]);
            Some(out)
        }
        Op::GatherND { batch_dims } => {
            let data = first()?;
            let indices = in_shapes.get(1).cloned().flatten()?;
            Some(gather_nd_shape(&data, &indices, *batch_dims)?.0)
        }
        // GatherElements: the output takes the indices' shape.
        Op::GatherElements { .. } => in_shapes.get(1).cloned().flatten(),
        Op::ConstantOfShape { .. } => ints(0)?
            .iter()
            .map(|&d| usize::try_from(d).ok())
            .collect::<Option<_>>(),
        Op::Range => {
            let (s, l, d) = (val(0)?, val(1)?, val(2)?);
            if s.len() != 1 || l.len() != 1 || d.len() != 1 {
                return None;
            }
            if s.is_float() || l.is_float() || d.is_float() {
                let (s, l, d) = (s.float_at(0), l.float_at(0), d.float_at(0));
                if d == 0.0 {
                    return None;
                }
                let n = ((l - s) / d).ceil().max(0.0);
                n.is_finite().then(|| vec![n as usize])
            } else {
                let (s, l, d) = (s.int_at(0), l.int_at(0), d.int_at(0));
                if d == 0 {
                    return None;
                }
                let span = l - s;
                let n = if (span > 0) == (d > 0) {
                    ((span + d - d.signum()) / d).max(0)
                } else {
                    0
                };
                Some(vec![n as usize])
            }
        }
        Op::Tile => {
            let input = first()?;
            let repeats = ints(1)?;
            if repeats.len() != input.len() || repeats.iter().any(|&r| r < 0) {
                return None;
            }
            Some(
                input
                    .iter()
                    .zip(repeats)
                    .map(|(&d, r)| d * r as usize)
                    .collect(),
            )
        }
        Op::Slice => {
            // `slice_plan` is shared with the kernel lowering, so the extents inferred
            // here and the mapping the kernel copies under cannot drift apart.
            let plan = teeny_core::graph::slice_plan(
                &first()?,
                &ints(1)?,
                &ints(2)?,
                ints(3).as_deref(),
                ints(4).as_deref(),
            )?;
            Some(plan.out_shape)
        }
        Op::Pad { .. } => {
            let input = first()?;
            let pads = ints(1)?;
            let rank = input.len();
            // `pads` is [begin per axis..., end per axis...], optionally restricted to
            // the axes named by operand 3.
            let axes = match ints(3) {
                Some(a) => a,
                None => (0..rank as i64).collect(),
            };
            if pads.len() != axes.len() * 2 {
                return None;
            }
            let mut out = input.clone();
            for (k, &a) in axes.iter().enumerate() {
                let ax = norm_axis(a, rank).filter(|&ax| ax < rank)?;
                let total = pads[k] + pads[k + axes.len()];
                let d = out[ax] as i64 + total;
                if d < 0 {
                    return None;
                }
                out[ax] = d as usize;
            }
            Some(out)
        }
        // `Split(data, split?)`. The importer records one graph node per ONNX node, so
        // this is the shape of output 0: either the first entry of the `split` operand,
        // or an equal share of the axis.
        Op::Split { axis, num_outputs } => {
            let input = first()?;
            let rank = input.len();
            let ax = norm_axis(*axis, rank).filter(|&ax| ax < rank)?;
            let mut out = input.clone();
            out[ax] = match ints(1) {
                Some(split) => usize::try_from(*split.first()?).ok()?,
                None => {
                    if *num_outputs == 0 || !input[ax].is_multiple_of(*num_outputs) {
                        return None;
                    }
                    input[ax] / num_outputs
                }
            };
            Some(out)
        }
        Op::MatMul => {
            let a = in_shapes.first().cloned().flatten()?;
            let b = in_shapes.get(1).cloned().flatten()?;
            matmul_shape(&a, &b)
        }

        // --- convolution and pooling ---
        Op::Conv2d {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ..
        } => {
            let x = first()?;
            if x.len() != 4 {
                return None;
            }
            let oh = conv_out(x[2], *kernel_h, *stride_h, *padding_h)?;
            let ow = conv_out(x[3], *kernel_w, *stride_w, *padding_w)?;
            Some(vec![x[0], *out_channels, oh, ow])
        }
        Op::ConvTranspose {
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            ..
        } => {
            let x = first()?;
            if x.len() != 4 {
                return None;
            }
            // A transposed convolution runs the forward extent formula backwards:
            // (in - 1) * stride - 2 * pad + kernel + output_pad.
            let oh = conv_transpose_out(x[2], *kernel_h, *stride_h, *padding_h, *output_padding_h)?;
            let ow = conv_transpose_out(x[3], *kernel_w, *stride_w, *padding_w, *output_padding_w)?;
            Some(vec![x[0], *out_channels, oh, ow])
        }
        Op::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            let x = first()?;
            if x.len() != 4 {
                return None;
            }
            let oh = conv_out(x[2], *kernel_h, *stride_h, *pad_h)?;
            let ow = conv_out(x[3], *kernel_w, *stride_w, *pad_w)?;
            Some(vec![x[0], x[1], oh, ow])
        }
        // Now that the IR carries the padding, the extent follows the same rule as
        // every other windowed op. This used to decline, because inferring without the
        // pads would have produced a confidently wrong shape for the 32 of 34
        // AveragePool nodes in the FoundationStereo export that pad by 1.
        Op::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            ..
        } => {
            let x = first()?;
            if x.len() != 4 {
                return None;
            }
            let oh = conv_out(x[2], *kernel_h, *stride_h, *pad_h)?;
            let ow = conv_out(x[3], *kernel_w, *stride_w, *pad_w)?;
            Some(vec![x[0], x[1], oh, ow])
        }
        Op::GlobalAvgPool | Op::GlobalMaxPool => {
            let x = first()?;
            if x.len() < 2 {
                return None;
            }
            let mut out = vec![x[0], x[1]];
            out.extend(core::iter::repeat_n(1usize, x.len() - 2));
            Some(out)
        }

        // `GridSample` takes its spatial extent from the grid: [N, C, H_out, W_out]
        // where the grid is [N, H_out, W_out, 2].
        Op::GridSample { .. } => {
            let x = first()?;
            let grid = in_shapes.get(1).cloned().flatten()?;
            if x.len() != 4 || grid.len() != 4 {
                return None;
            }
            Some(vec![x[0], x[1], grid[1], grid[2]])
        }

        // `Resize(x, roi, scales, sizes)`: prefer the explicit sizes operand.
        // Every reduction has the same shape rule, differing only in the arithmetic,
        // which the shape does not depend on.
        Op::ReduceSum {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceMean {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceMax {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceMin {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceProd {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceL1 {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceL2 {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceLogSum {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceLogSumExp {
            keepdims,
            noop_with_empty_axes,
            axes,
        }
        | Op::ReduceSumSquare {
            keepdims,
            noop_with_empty_axes,
            axes,
        } => {
            let input = first()?;
            let rank = input.len();
            // `axes` is an attribute in older opsets and operand 1 in newer ones, and
            // an export states it one way or the other, so both are read. The operand
            // wins when present, since a model that supplies it does not set the
            // attribute.
            let axes = match ints(1) {
                Some(operand) => operand,
                None => axes.clone(),
            };
            let axes: Vec<usize> = if axes.is_empty() {
                // An empty list means every axis, except where the op says it means
                // none. Reducing nothing is the identity, which has to be said
                // explicitly or the output comes back rank-0.
                if *noop_with_empty_axes {
                    return Some(input.clone());
                }
                (0..rank).collect()
            } else {
                axes.iter()
                    .map(|&a| norm_axis(a, rank).filter(|&a| a < rank))
                    .collect::<Option<_>>()?
            };
            Some(
                (0..rank)
                    .filter_map(|d| match (axes.contains(&d), *keepdims) {
                        (true, true) => Some(1),
                        (true, false) => None,
                        (false, _) => Some(input[d]),
                    })
                    .collect(),
            )
        }

        Op::Einsum { equation } => {
            // Planned by the same function the lowering uses, so the extents recorded
            // here and the strides the contraction runs under come from one reading of
            // the equation.
            let plan = teeny_core::graph::einsum_plan(equation, &first()?, &shape_at(1)?).ok()?;
            Some(plan.out_shape)
        }

        Op::Resize { .. } => {
            let x = first()?;
            if let Some(sizes) = ints(3)
                && sizes.len() == x.len()
            {
                return sizes
                    .iter()
                    .map(|&d| usize::try_from(d).ok())
                    .collect::<Option<_>>();
            }
            let scales = val(2)?;
            if scales.len() != x.len() {
                return None;
            }
            Some(
                x.iter()
                    .enumerate()
                    .map(|(i, &d)| (d as f64 * scales.float_at(i)).floor() as usize)
                    .collect(),
            )
        }

        _ => None,
    }
}

/// Output extent of a convolution or pooling window.
///
/// This follows the IR's own representation of the op, which has a single padding per
/// spatial axis and no dilation. A model using dilated or asymmetrically padded
/// convolution has already lost that on import, so the node is mis-imported whatever
/// this returns; the shape here is the one the IR describes.
fn conv_out(extent: usize, kernel: usize, stride: usize, pad: usize) -> Option<usize> {
    if stride == 0 {
        return None;
    }
    let padded = extent + 2 * pad;
    (padded >= kernel).then(|| (padded - kernel) / stride + 1)
}

/// Output extent of a transposed convolution, the forward formula inverted. As with
/// [`conv_out`], this follows the IR's representation: one padding per axis, no
/// dilation.
fn conv_transpose_out(
    extent: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    output_pad: usize,
) -> Option<usize> {
    let grown = (extent.max(1) - 1) * stride + kernel + output_pad;
    grown.checked_sub(2 * pad)
}

/// ONNX `MatMul`: the last two dimensions contract, the rest broadcast. A 1-D operand
/// is promoted for the contraction and the promoted axis is then removed.
fn matmul_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (a2, a_promoted) = if a.len() == 1 {
        (vec![1, a[0]], true)
    } else {
        (a.to_vec(), false)
    };
    let (b2, b_promoted) = if b.len() == 1 {
        (vec![b[0], 1], true)
    } else {
        (b.to_vec(), false)
    };
    let (m, k1) = (a2[a2.len() - 2], a2[a2.len() - 1]);
    let (k2, n) = (b2[b2.len() - 2], b2[b2.len() - 1]);
    if k1 != k2 {
        return None;
    }
    let batch = broadcast_shapes(&a2[..a2.len() - 2], &b2[..b2.len() - 2])?;
    let mut out = batch;
    if !a_promoted {
        out.push(m);
    }
    if !b_promoted {
        out.push(n);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_td(shape: &[usize], vals: &[i64]) -> TensorData {
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        TensorData {
            dtype: DtypeRepr::I64,
            shape: shape.iter().map(|&d| Some(d)).collect(),
            bytes,
        }
    }

    fn ints(v: &Value) -> Vec<i64> {
        v.as_ints()
    }

    fn val(shape: &[usize], vals: &[i64]) -> Value {
        Value {
            dtype: DtypeRepr::I64,
            shape: shape.to_vec(),
            data: Data::Int(vals.to_vec()),
        }
    }

    fn bools(shape: &[usize], vals: &[bool]) -> Value {
        Value {
            dtype: DtypeRepr::Bool,
            shape: shape.to_vec(),
            data: Data::Bool(vals.to_vec()),
        }
    }

    fn nonzero(v: &Value) -> Value {
        match eval_nonzero(&[v], &MaterializeConfig::default()) {
            Eval::Ok(out) => out,
            _ => panic!("expected a folded NonZero"),
        }
    }

    /// One row per axis: a vector reports a single row of positions.
    #[test]
    fn test_nonzero_folds_a_vector() {
        let v = nonzero(&bools(&[5], &[false, true, false, true, true]));
        assert_eq!(v.dtype, DtypeRepr::I64);
        assert_eq!(v.shape, vec![1, 3]);
        assert_eq!(ints(&v), vec![1, 3, 4]);
    }

    /// A matrix reports a row of row-indices and a row of column-indices, in the order
    /// the elements appear row-major, which is the order ONNX specifies.
    #[test]
    fn test_nonzero_folds_a_matrix_in_row_major_order() {
        //  0 1 0
        //  1 0 1
        let v = nonzero(&val(&[2, 3], &[0, 1, 0, 1, 0, 1]));
        assert_eq!(v.shape, vec![2, 3]);
        // (0,1), (1,0), (1,2)
        assert_eq!(ints(&v), vec![0, 1, 1, 1, 0, 2]);
    }

    /// Nothing found is an empty result, not an absent one: the shape is still known,
    /// which is what lets the nodes downstream of it resolve.
    #[test]
    fn test_nonzero_of_all_zeros_is_empty_but_shaped() {
        let v = nonzero(&val(&[2, 2], &[0, 0, 0, 0]));
        assert_eq!(v.shape, vec![2, 0]);
        assert!(ints(&v).is_empty());
    }

    /// Floats count by value, so a negative or fractional element is non-zero.
    #[test]
    fn test_nonzero_reads_a_float_against_zero() {
        let v = Value {
            dtype: DtypeRepr::F32,
            shape: vec![4],
            data: Data::Float(vec![0.0, -1.5, 0.0, 0.25]),
        };
        assert_eq!(ints(&nonzero(&v)), vec![1, 3]);
    }

    /// A scalar has no axes to report coordinates along.
    #[test]
    fn test_nonzero_declines_a_scalar() {
        let v = Value {
            dtype: DtypeRepr::I64,
            shape: vec![],
            data: Data::Int(vec![1]),
        };
        assert!(matches!(
            eval_nonzero(&[&v], &MaterializeConfig::default()),
            Eval::Unsupported
        ));
    }

    /// The result is as large as the data makes it, so it is held to the same size
    /// limit as every other folded value.
    #[test]
    fn test_nonzero_declines_a_result_above_the_limit() {
        let v = val(&[4], &[1, 1, 1, 1]);
        let config = MaterializeConfig { max_elements: 3 };
        assert!(matches!(eval_nonzero(&[&v], &config), Eval::TooLarge));
    }

    fn gather_nd(data: &Value, indices: &Value, batch_dims: i64) -> Eval {
        eval_gather_nd(&[data, indices], batch_dims, &MaterializeConfig::default())
    }

    fn folded(e: Eval) -> Value {
        match e {
            Eval::Ok(v) => v,
            _ => panic!("expected a folded value"),
        }
    }

    /// The export's own form: a vector of values and a column of positions into it.
    #[test]
    fn test_gather_nd_folds_a_column_of_positions() {
        let data = val(&[5], &[10, 11, 12, 13, 14]);
        let indices = val(&[3, 1], &[4, 0, 2]);
        let v = folded(gather_nd(&data, &indices, 0));
        assert_eq!(v.shape, vec![3]);
        assert_eq!(ints(&v), vec![14, 10, 12]);
    }

    /// A coordinate narrower than the data's rank gathers whole slices: the axes it
    /// does not name come along.
    #[test]
    fn test_gather_nd_gathers_slices_when_the_coordinate_is_short() {
        //  data[2][3] = [[0,1,2],[3,4,5]]
        let data = val(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let indices = val(&[2, 1], &[1, 0]);
        let v = folded(gather_nd(&data, &indices, 0));
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(ints(&v), vec![3, 4, 5, 0, 1, 2]);
    }

    /// A full coordinate names one element, so nothing comes along.
    #[test]
    fn test_gather_nd_with_a_full_coordinate_picks_elements() {
        let data = val(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let indices = val(&[2, 2], &[0, 2, 1, 1]);
        let v = folded(gather_nd(&data, &indices, 0));
        assert_eq!(v.shape, vec![2]);
        assert_eq!(ints(&v), vec![2, 4]);
    }

    /// With a batch axis the two operands walk it together: row 0 of the indices
    /// addresses row 0 of the data, and never another row.
    #[test]
    fn test_gather_nd_keeps_its_batch_axes_aligned() {
        //  data[2][3] = [[0,1,2],[3,4,5]], one index per batch row
        let data = val(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let indices = val(&[2, 1, 1], &[2, 0]);
        let v = folded(gather_nd(&data, &indices, 1));
        assert_eq!(v.shape, vec![2, 1]);
        assert_eq!(ints(&v), vec![2, 3]);
    }

    #[test]
    fn test_gather_nd_counts_a_negative_index_from_the_end() {
        let data = val(&[4], &[7, 8, 9, 10]);
        let indices = val(&[2, 1], &[-1, -4]);
        assert_eq!(ints(&folded(gather_nd(&data, &indices, 0))), vec![10, 7]);
    }

    #[test]
    fn test_gather_nd_declines_an_index_past_the_end() {
        let data = val(&[4], &[7, 8, 9, 10]);
        let indices = val(&[1, 1], &[4]);
        assert!(matches!(gather_nd(&data, &indices, 0), Eval::OutOfRange));
    }

    /// Shapes that do not describe a GatherND are declined rather than read anyway: a
    /// coordinate wider than the axes left to address, and batch axes that disagree.
    #[test]
    fn test_gather_nd_declines_shapes_it_cannot_mean() {
        let data = val(&[4], &[7, 8, 9, 10]);
        let too_wide = val(&[1, 2], &[0, 0]);
        assert!(matches!(gather_nd(&data, &too_wide, 0), Eval::Unsupported));

        let data = val(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let wrong_batch = val(&[3, 1, 1], &[0, 0, 0]);
        assert!(matches!(
            gather_nd(&data, &wrong_batch, 1),
            Eval::Unsupported
        ));
    }

    /// Inference and folding read the shapes through the same function, so a node that
    /// folds and a node that only gets a shape agree on what that shape is.
    #[test]
    fn test_gather_nd_infers_the_shape_it_folds_to() {
        let data = val(&[2, 3, 4], &[0; 24]);
        let indices = val(&[5, 2], &[0; 10]);
        let folded_shape = folded(gather_nd(&data, &indices, 0)).shape;
        assert_eq!(folded_shape, vec![5, 4]);
        assert_eq!(
            infer_static_shape(
                &Op::GatherND { batch_dims: 0 },
                &[stat(&[2, 3, 4]), stat(&[5, 2])],
                &[None, None]
            ),
            Some(folded_shape)
        );
    }

    #[test]
    fn test_shape_of_folds_a_static_shape() {
        let shapes = [vec![Some(1), Some(3), Some(480), Some(640)]];
        let refs: Vec<&Shape> = shapes.iter().collect();
        let Eval::Ok(v) = eval_shape_of(0, i64::MAX, &refs) else {
            panic!("expected a folded shape");
        };
        assert_eq!(ints(&v), vec![1, 3, 480, 640]);
        assert_eq!(v.shape, vec![4]);
    }

    #[test]
    fn test_shape_of_declines_a_dynamic_dim() {
        let shapes = [vec![None, Some(3)]];
        let refs: Vec<&Shape> = shapes.iter().collect();
        assert!(matches!(
            eval_shape_of(0, i64::MAX, &refs),
            Eval::Unsupported
        ));
    }

    /// An empty recorded shape is ambiguous — a real rank-0 tensor, or one the
    /// importer never inferred — so folding it would risk an empty shape vector.
    #[test]
    fn test_shape_of_refuses_an_unknown_rank() {
        let shapes = [Shape::new()];
        let refs: Vec<&Shape> = shapes.iter().collect();
        assert!(matches!(
            eval_shape_of(0, i64::MAX, &refs),
            Eval::UnknownRank
        ));
    }

    /// Exporters concatenate scalars to build a shape vector; rank-0 reads as `[1]`.
    #[test]
    fn test_concat_promotes_scalar_operands() {
        let a = val(&[], &[1]);
        let b = val(&[], &[64]);
        let c = val(&[1], &[-1]);
        let Eval::Ok(v) = eval_concat(&[&a, &b, &c], 0, &MaterializeConfig::default()) else {
            panic!("expected scalars to concatenate");
        };
        assert_eq!(ints(&v), vec![1, 64, -1]);
        assert_eq!(v.shape, vec![3]);
    }

    #[test]
    fn test_gather_reports_an_out_of_range_index() {
        let data = val(&[4], &[1, 3, 480, 640]);
        let idx = val(&[], &[4]);
        assert!(matches!(
            eval_gather(&[&data, &idx], 0, &MaterializeConfig::default()),
            Eval::OutOfRange
        ));
    }

    #[test]
    fn test_gather_picks_one_dim_out_of_a_shape_vector() {
        let data = val(&[4], &[1, 3, 480, 640]);
        let idx = val(&[], &[2]);
        let Eval::Ok(v) = eval_gather(&[&data, &idx], 0, &MaterializeConfig::default()) else {
            panic!("expected a folded gather");
        };
        assert_eq!(ints(&v), vec![480]);
        assert!(v.shape.is_empty(), "a scalar index yields a scalar");
    }

    #[test]
    fn test_concat_builds_a_reshape_target() {
        let a = val(&[1], &[1]);
        let b = val(&[1], &[-1]);
        let c = val(&[2], &[480, 640]);
        let Eval::Ok(v) = eval_concat(&[&a, &b, &c], 0, &MaterializeConfig::default()) else {
            panic!("expected a folded concat");
        };
        assert_eq!(ints(&v), vec![1, -1, 480, 640]);
    }

    #[test]
    fn test_binary_ops_broadcast_and_divide_toward_zero() {
        let a = val(&[3], &[7, -7, 9]);
        let b = val(&[], &[2]);
        let Eval::Ok(v) = binary(&[&a, &b], BinKind::Div) else {
            panic!("expected a folded div");
        };
        assert_eq!(ints(&v), vec![3, -3, 4]);
        assert_eq!(v.shape, vec![3]);
    }

    #[test]
    fn test_slice_handles_negative_and_stepped_ranges() {
        let data = val(&[6], &[0, 1, 2, 3, 4, 5]);
        let starts = val(&[1], &[1]);
        let ends = val(&[1], &[-1]);
        let axes = val(&[1], &[0]);
        let steps = val(&[1], &[2]);
        let Eval::Ok(v) = eval_slice(&[&data, &starts, &ends, &axes, &steps]) else {
            panic!("expected a folded slice");
        };
        assert_eq!(ints(&v), vec![1, 3]);
    }

    #[test]
    fn test_transpose_moves_elements() {
        let data = val(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let Eval::Ok(v) = eval_transpose(&[&data], &[1, 0]) else {
            panic!("expected a folded transpose");
        };
        assert_eq!(v.shape, vec![3, 2]);
        assert_eq!(ints(&v), vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn test_reshape_resolves_a_wildcard() {
        let data = val(&[6], &[0, 1, 2, 3, 4, 5]);
        let target = val(&[2], &[3, -1]);
        let Eval::Ok(v) = eval_reshape(&[&data, &target]) else {
            panic!("expected a folded reshape");
        };
        assert_eq!(v.shape, vec![3, 2]);
    }

    #[test]
    fn test_range_counts_up_and_down() {
        let cfg = MaterializeConfig::default();
        let (s, l, d) = (val(&[], &[0]), val(&[], &[5]), val(&[], &[2]));
        let Eval::Ok(v) = eval_range(&[&s, &l, &d], &cfg) else {
            panic!("expected a folded range");
        };
        assert_eq!(ints(&v), vec![0, 2, 4]);

        let (s, l, d) = (val(&[], &[5]), val(&[], &[0]), val(&[], &[-2]));
        let Eval::Ok(v) = eval_range(&[&s, &l, &d], &cfg) else {
            panic!("expected a folded descending range");
        };
        assert_eq!(ints(&v), vec![5, 3, 1]);
    }

    #[test]
    fn test_expand_stretches_size_one_axes() {
        let data = val(&[1, 3], &[7, 8, 9]);
        let target = val(&[2], &[2, 3]);
        let Eval::Ok(v) = eval_expand(&[&data, &target], &MaterializeConfig::default()) else {
            panic!("expected a folded expand");
        };
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(ints(&v), vec![7, 8, 9, 7, 8, 9]);
    }

    #[test]
    fn test_unary_sqrt_promotes_integers_to_float() {
        let v = val(&[2], &[4, 9]);
        let Eval::Ok(r) = unary(&[&v], UnKind::Sqrt) else {
            panic!("expected a folded sqrt");
        };
        assert_eq!(r.dtype, DtypeRepr::F32);
        assert_eq!(
            (0..2).map(|i| r.float_at(i)).collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
    }

    #[test]
    fn test_unary_neg_stays_integral() {
        let v = val(&[2], &[4, -9]);
        let Eval::Ok(r) = unary(&[&v], UnKind::Neg) else {
            panic!("expected a folded neg");
        };
        assert_eq!(ints(&r), vec![-4, 9]);
        assert_eq!(r.dtype, DtypeRepr::I64);
    }

    /// `ConstantOfShape` emits one element per slot in its shape operand, filled
    /// with its `value` attribute. Standing a scalar fill in for the whole tensor
    /// is what made exporter-emitted pad lists come out one element short.
    #[test]
    fn test_constant_of_shape_expands_to_its_shape_operand() {
        let shape = val(&[1], &[2]);
        let fill = val(&[1], &[0]);
        let Eval::Ok(v) = eval_constant_of_shape(
            DtypeRepr::I64,
            &[&shape],
            Some(&fill),
            &MaterializeConfig::default(),
        ) else {
            panic!("expected a folded ConstantOfShape");
        };
        assert_eq!(v.shape, vec![2]);
        assert_eq!(ints(&v), vec![0, 0]);
    }

    #[test]
    fn test_constant_of_shape_honours_a_non_zero_fill() {
        let shape = val(&[1], &[3]);
        let fill = val(&[1], &[7]);
        let Eval::Ok(v) = eval_constant_of_shape(
            DtypeRepr::I64,
            &[&shape],
            Some(&fill),
            &MaterializeConfig::default(),
        ) else {
            panic!("expected a folded ConstantOfShape");
        };
        assert_eq!(ints(&v), vec![7, 7, 7]);
    }

    /// The exporter's `F.pad` pattern: a literal prefix concatenated with a
    /// `ConstantOfShape` tail, reshaped into (begin, end) pairs. The whole chain
    /// must fold, and the pad list must come out even.
    #[test]
    fn test_folds_an_exporter_pad_list() {
        let mut g = Graph::new();
        let shape_c = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(1)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        let cos = g.add_node(
            Op::ConstantOfShape {
                dtype: DtypeRepr::I64,
                // The folder reads the fill from the model's tensors, not from here.
                fill: Vec::new(),
            },
            vec![shape_c],
            DtypeRepr::I64,
            vec![Some(2)],
        );
        let prefix = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(6)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(6)],
        );
        let cat = g.add_node(
            Op::Concat { axis: 0 },
            vec![prefix, cos],
            DtypeRepr::I64,
            vec![Some(8)],
        );
        let tgt = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(2)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(2)],
        );
        let rs = g.add_node(
            Op::Reshape,
            vec![cat, tgt],
            DtypeRepr::I64,
            vec![Some(4), Some(2)],
        );

        let mut tensors = BTreeMap::new();
        tensors.insert(shape_c, int_td(&[1], &[2]));
        tensors.insert(cos, int_td(&[1], &[0])); // the fill value
        tensors.insert(prefix, int_td(&[6], &[0, 0, 0, 0, 1, 1]));
        tensors.insert(tgt, int_td(&[2], &[-1, 2]));

        let (folded, report) =
            materialize_constants(&mut g, &tensors, &MaterializeConfig::default());
        assert!(
            report.unsupported.is_empty(),
            "the pad chain should fold whole: {:?}",
            report.unsupported_ranked()
        );
        let v = Value::decode(&folded[&rs]).expect("reshape result decodes");
        assert_eq!(v.shape, vec![4, 2], "an 8-element pad list, not 7");
        assert_eq!(v.as_ints(), vec![0, 0, 0, 0, 1, 1, 0, 0]);
    }

    /// An `Unsqueeze` whose axes never made it into the IR must not fold: passing
    /// the value through would silently drop an axis.
    #[test]
    fn test_unsqueeze_refuses_empty_axes() {
        let v = val(&[2], &[1, 2]);
        assert!(matches!(eval_unsqueeze(&[&v], &[]), Eval::Unsupported));
    }

    // --- shape refinement ---

    fn stat(dims: &[usize]) -> Option<Vec<usize>> {
        Some(dims.to_vec())
    }

    #[test]
    fn test_broadcasting_shapes_are_inferred() {
        let shapes = [stat(&[1, 3, 1]), stat(&[4, 1, 5])];
        assert_eq!(
            infer_static_shape(&Op::Add, &shapes, &[None, None]),
            Some(vec![4, 3, 5])
        );
        // A comparison broadcasts the same way; only its dtype differs.
        assert_eq!(
            infer_static_shape(&Op::Equal, &shapes, &[None, None]),
            Some(vec![4, 3, 5])
        );
        // Incompatible extents have no answer.
        let bad = [stat(&[3]), stat(&[4])];
        assert_eq!(infer_static_shape(&Op::Mul, &bad, &[None, None]), None);
    }

    /// A broadcast against an operand whose rank is unknown has no answer.
    ///
    /// This is the fix that mattered most on a real export. An unresolved shape and a
    /// rank-0 scalar are the same empty vector, so passing an unresolved operand as
    /// `Some(vec![])` makes the broadcast return the *other* operand's shape, which is
    /// complete, static and wrong. Because it is complete, `needs_shape` is then false
    /// and no later round revisits it. On the FoundationStereo export that one
    /// confusion put the qkv bias's shape, rank-1 [3072], on the transformer's
    /// attention output and left 10,000 downstream nodes unshaped.
    #[test]
    fn test_a_broadcast_against_an_unknown_rank_has_no_answer() {
        let bias = [None, Some(vec![3072])];
        assert_eq!(infer_static_shape(&Op::Add, &bias, &[None, None]), None);
        // Either side, and for every broadcasting op.
        let other = [Some(vec![3072]), None];
        assert_eq!(infer_static_shape(&Op::Mul, &other, &[None, None]), None);
        assert_eq!(
            infer_static_shape(&Op::Where, &[None, None, None], &[None; 3]),
            None
        );

        // A genuine rank-0 scalar still broadcasts, which is why the two cases have to
        // be told apart rather than both refused.
        let scalar = [Some(vec![2, 3]), Some(vec![])];
        assert_eq!(
            infer_static_shape(&Op::Add, &scalar, &[None, None]),
            Some(vec![2, 3])
        );
    }

    /// Reductions drop or keep the axes they reduce, and read their axes from either
    /// the attribute or the operand depending on the opset.
    #[test]
    fn test_reductions_drop_or_keep_the_reduced_axes() {
        let shapes = [stat(&[2, 8, 12, 4800])];

        // keepdims, the default, leaves a unit axis behind.
        let op = Op::ReduceL2 {
            keepdims: true,
            noop_with_empty_axes: false,
            axes: vec![-1],
        };
        assert_eq!(
            infer_static_shape(&op, &shapes, &[None]),
            Some(vec![2, 8, 12, 1])
        );

        // Without it the axis is removed.
        let op = Op::ReduceMean {
            keepdims: false,
            noop_with_empty_axes: false,
            axes: vec![1],
        };
        assert_eq!(
            infer_static_shape(&op, &shapes, &[None]),
            Some(vec![2, 12, 4800])
        );

        // Several axes at once, one of them negative.
        let op = Op::ReduceMax {
            keepdims: false,
            noop_with_empty_axes: false,
            axes: vec![0, -1],
        };
        assert_eq!(infer_static_shape(&op, &shapes, &[None]), Some(vec![8, 12]));

        // An axis outside the rank has no answer.
        let op = Op::ReduceSum {
            keepdims: true,
            noop_with_empty_axes: false,
            axes: vec![9],
        };
        assert_eq!(infer_static_shape(&op, &shapes, &[None]), None);
    }

    /// An empty axes list means every axis, or none, depending on the op's own flag.
    /// Reading it the wrong way turns a full reduction into the identity or back.
    #[test]
    fn test_an_empty_axes_list_follows_the_noop_flag() {
        let shapes = [stat(&[2, 3, 4])];

        let reduce_all = Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: false,
            axes: vec![],
        };
        assert_eq!(
            infer_static_shape(&reduce_all, &shapes, &[None]),
            Some(vec![])
        );

        let reduce_all_keepdims = Op::ReduceSum {
            keepdims: true,
            noop_with_empty_axes: false,
            axes: vec![],
        };
        assert_eq!(
            infer_static_shape(&reduce_all_keepdims, &shapes, &[None]),
            Some(vec![1, 1, 1])
        );

        let identity = Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: true,
            axes: vec![],
        };
        assert_eq!(
            infer_static_shape(&identity, &shapes, &[None]),
            Some(vec![2, 3, 4])
        );
    }

    /// Opset 13 moved some reductions' axes to an operand and opset 18 the rest, so an
    /// export states it either way. The operand wins when supplied.
    #[test]
    fn test_a_reduction_reads_its_axes_from_the_operand_too() {
        let shapes = [stat(&[2, 3, 4]), stat(&[1])];
        let op = Op::ReduceSum {
            keepdims: false,
            noop_with_empty_axes: false,
            // Nothing in the attribute: an opset-13 export puts it in the operand.
            axes: vec![],
        };
        let axes = val(&[1], &[1]);
        assert_eq!(
            infer_static_shape(&op, &shapes, &[None, Some(&axes)]),
            Some(vec![2, 4])
        );
    }

    #[test]
    fn test_reshape_resolves_its_target_from_a_value() {
        let shapes = [stat(&[2, 3, 4]), stat(&[2])];
        let target = val(&[2], &[4, -1]);
        assert_eq!(
            infer_static_shape(&Op::Reshape, &shapes, &[None, Some(&target)]),
            Some(vec![4, 6])
        );
        // A 0 copies the input's extent at that axis.
        let keep = val(&[3], &[0, 0, -1]);
        assert_eq!(
            infer_static_shape(&Op::Reshape, &shapes, &[None, Some(&keep)]),
            Some(vec![2, 3, 4])
        );
        // Without the target's value there is nothing to infer from.
        assert_eq!(
            infer_static_shape(&Op::Reshape, &shapes, &[None, None]),
            None
        );
    }

    #[test]
    fn test_conv_and_pool_extents_follow_the_window() {
        let conv = Op::Conv2d {
            in_channels: 3,
            out_channels: 16,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            padding_h: 1,
            padding_w: 1,
            groups: 1,
            has_bias: true,
        };
        let shapes = [stat(&[1, 3, 480, 640])];
        assert_eq!(
            infer_static_shape(&conv, &shapes, &[None]),
            Some(vec![1, 16, 240, 320])
        );

        // A transposed convolution runs the same formula backwards.
        let up = Op::ConvTranspose {
            in_channels: 16,
            out_channels: 8,
            kernel_h: 4,
            kernel_w: 4,
            stride_h: 2,
            stride_w: 2,
            padding_h: 1,
            padding_w: 1,
            output_padding_h: 0,
            output_padding_w: 0,
            groups: 1,
            has_bias: false,
        };
        assert_eq!(
            infer_static_shape(&up, &[stat(&[1, 16, 240, 320])], &[None]),
            Some(vec![1, 8, 480, 640])
        );
    }

    /// `AvgPool2d`'s extent now follows from its own fields, padding included. This
    /// replaces a test that asserted it had to decline, which was right while the IR
    /// dropped the padding and would have been wrong to keep afterwards.
    #[test]
    fn test_avg_pool_accounts_for_its_padding() {
        let padded = Op::AvgPool2d {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
            count_include_pad: true,
        };
        // The shape every padded AveragePool in the FoundationStereo export has.
        assert_eq!(
            infer_static_shape(&padded, &[stat(&[1, 128, 120, 160])], &[None]),
            Some(vec![1, 128, 60, 80])
        );

        // Without the padding the same window gives a smaller output, which is the
        // difference that made inferring unsafe before.
        let unpadded = Op::AvgPool2d {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            count_include_pad: true,
        };
        assert_eq!(
            infer_static_shape(&unpadded, &[stat(&[1, 128, 120, 160])], &[None]),
            Some(vec![1, 128, 59, 79])
        );

        // `count_include_pad` changes the values, not the extent.
        let excluded = Op::AvgPool2d {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
            count_include_pad: false,
        };
        assert_eq!(
            infer_static_shape(&excluded, &[stat(&[1, 128, 120, 160])], &[None]),
            infer_static_shape(&padded, &[stat(&[1, 128, 120, 160])], &[None])
        );
    }

    #[test]
    fn test_matmul_contracts_and_broadcasts_the_batch() {
        let m = infer_static_shape(
            &Op::MatMul,
            &[stat(&[2, 3, 4]), stat(&[2, 4, 5])],
            &[None, None],
        );
        assert_eq!(m, Some(vec![2, 3, 5]));
        // The batch dimensions broadcast.
        let b = infer_static_shape(
            &Op::MatMul,
            &[stat(&[1, 3, 4]), stat(&[5, 4, 2])],
            &[None, None],
        );
        assert_eq!(b, Some(vec![5, 3, 2]));
        // A 1-D operand is promoted for the contraction, then its axis removed.
        let v = infer_static_shape(&Op::MatMul, &[stat(&[3, 4]), stat(&[4])], &[None, None]);
        assert_eq!(v, Some(vec![3]));
        // Mismatched contraction extents have no answer.
        assert_eq!(
            infer_static_shape(&Op::MatMul, &[stat(&[3, 4]), stat(&[5, 2])], &[None, None]),
            None
        );
    }

    #[test]
    fn test_slice_extents_account_for_steps_and_clamping() {
        let shapes = [stat(&[10]), stat(&[1]), stat(&[1]), stat(&[1]), stat(&[1])];
        let (starts, ends, axes, steps) = (
            val(&[1], &[1]),
            val(&[1], &[-1]),
            val(&[1], &[0]),
            val(&[1], &[2]),
        );
        assert_eq!(
            infer_static_shape(
                &Op::Slice,
                &shapes,
                &[None, Some(&starts), Some(&ends), Some(&axes), Some(&steps)]
            ),
            // indices 1,3,5,7 of 0..9 up to (exclusive) 9
            Some(vec![4])
        );
        // An end past the axis clamps rather than overruns.
        let big_end = val(&[1], &[99]);
        let one = val(&[1], &[1]);
        let zero = val(&[1], &[0]);
        assert_eq!(
            infer_static_shape(
                &Op::Slice,
                &shapes,
                &[None, Some(&zero), Some(&big_end), Some(&axes), Some(&one)]
            ),
            Some(vec![10])
        );
    }

    #[test]
    fn test_grid_sample_takes_its_extent_from_the_grid() {
        let op = Op::GridSample {
            mode: "bilinear".to_string(),
            padding_mode: "zeros".to_string(),
            align_corners: false,
        };
        assert_eq!(
            infer_static_shape(
                &op,
                &[stat(&[2, 8, 30, 40]), stat(&[2, 60, 80, 2])],
                &[None, None]
            ),
            Some(vec![2, 8, 60, 80])
        );
    }

    #[test]
    fn test_split_describes_its_first_output() {
        let op = Op::Split {
            axis: 1,
            num_outputs: 3,
        };
        // An equal share when there is no split operand.
        assert_eq!(
            infer_static_shape(&op, &[stat(&[2, 9, 4])], &[None]),
            Some(vec![2, 3, 4])
        );
        // An indivisible axis has no equal share.
        assert_eq!(infer_static_shape(&op, &[stat(&[2, 8, 4])], &[None]), None);
        // An explicit split operand gives output 0's extent.
        let split = val(&[3], &[5, 2, 2]);
        assert_eq!(
            infer_static_shape(&op, &[stat(&[2, 9, 4]), stat(&[3])], &[None, Some(&split)]),
            Some(vec![2, 5, 4])
        );
    }

    #[test]
    fn test_an_unimplemented_op_declines_rather_than_guessing() {
        // NonZero's output extent depends on the data, so no inference can resolve it.
        // Evaluating the node is the only thing that can, which is what the pass does
        // when the operand is known: see `test_nonzero_resolves_what_inference_cannot`.
        assert_eq!(
            infer_static_shape(&Op::NonZero, &[stat(&[4, 4])], &[None]),
            None
        );
    }

    /// A `NonZero` whose operand is known, and a consumer that could not be shaped
    /// until it was.
    ///
    /// Nothing about the shapes says how wide a `NonZero` is, so inference leaves it
    /// and everything downstream unresolved. Folding it closes the hole: here the
    /// `Transpose` behind it gets a shape only because the node in front of it stopped
    /// being a mystery. This is the shape of the whole tail of the FoundationStereo
    /// export, where a chain of constants runs And -> NonZero -> Transpose.
    #[test]
    fn test_nonzero_resolves_what_inference_cannot() {
        let mut g = Graph::new();
        let mask = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::Bool,
                shape: vec![Some(5)],
            },
            vec![],
            DtypeRepr::Bool,
            vec![Some(5)],
        );
        let nz = g.add_node(Op::NonZero, vec![mask], DtypeRepr::I64, vec![]);
        let t = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![nz],
            DtypeRepr::I64,
            vec![],
        );

        let mut tensors = BTreeMap::new();
        tensors.insert(
            mask,
            TensorData {
                dtype: DtypeRepr::Bool,
                shape: vec![Some(5)],
                bytes: vec![0, 1, 0, 1, 1],
            },
        );

        let (folded, _) = materialize_constants(&mut g, &tensors, &MaterializeConfig::default());

        // Three non-zero positions, reported as one row of coordinates.
        assert_eq!(g.nodes[nz].shape, vec![Some(1), Some(3)]);
        assert_eq!(
            folded.get(&nz).map(|v| v.shape.clone()),
            Some(vec![Some(1), Some(3)])
        );
        // And the consumer is shaped, which it could not be before.
        assert_eq!(g.nodes[t].shape, vec![Some(3), Some(1)]);
    }

    /// The point of the pass: a `Shape -> Gather -> Unsqueeze -> Concat -> Reshape`
    /// chain over a live activation. Shape inference alone cannot resolve the Reshape,
    /// because its target is computed; constant folding alone cannot fold the Shape,
    /// because the activation's shape is what it needs. Interleaved, both resolve.
    #[test]
    fn test_interleaving_resolves_what_neither_pass_can_alone() {
        let mut g = Graph::new();
        // An activation whose shape is known but whose values are not.
        let x = g.add_node(
            Op::Input,
            vec![],
            DtypeRepr::F32,
            vec![Some(2), Some(3), Some(4)],
        );
        let shape = g.add_node(
            Op::ShapeOf {
                start: 0,
                end: i64::MAX,
            },
            vec![x],
            DtypeRepr::I64,
            vec![],
        );
        let idx = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![],
            },
            vec![],
            DtypeRepr::I64,
            vec![],
        );
        // Pick dimension 0 (= 2).
        let dim = g.add_node(
            Op::Gather { axis: 0 },
            vec![shape, idx],
            DtypeRepr::I64,
            vec![],
        );
        let unsq = g.add_node(
            Op::Unsqueeze { axes: vec![0] },
            vec![dim],
            DtypeRepr::I64,
            vec![],
        );
        let tail = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(1)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        let target = g.add_node(
            Op::Concat { axis: 0 },
            vec![unsq, tail],
            DtypeRepr::I64,
            vec![],
        );
        // Reshape the activation to [2, -1], with the target computed above. Imported
        // with no shape at all, as an exporter leaves it.
        let rs = g.add_node(Op::Reshape, vec![x, target], DtypeRepr::F32, vec![]);

        let mut tensors = BTreeMap::new();
        tensors.insert(idx, int_td(&[], &[0]));
        tensors.insert(tail, int_td(&[1], &[-1]));

        let (folded, report) =
            materialize_constants(&mut g, &tensors, &MaterializeConfig::default());

        // The chain folded to the reshape target [2, -1] ...
        let t = Value::decode(&folded[&target]).expect("target decodes");
        assert_eq!(t.as_ints(), vec![2, -1]);
        // ... and the Reshape's own shape followed from it, which is the part a
        // separate shape-inference pass cannot reach.
        assert_eq!(g.nodes[rs].shape, vec![Some(2), Some(12)]);
        assert!(report.shapes_refined > 0);
        assert_eq!(report.shapes_unresolved, 0);
        // The Reshape itself is not folded: its data operand is a live activation.
        assert!(!folded.contains_key(&rs));
    }

    #[test]
    fn test_values_round_trip_through_encode() {
        let v = val(&[2, 2], &[1, -2, 3, -4]);
        let decoded = Value::decode(&v.encode()).expect("i64 decodes");
        assert_eq!(decoded, v);
    }

    /// The pass folds a whole `Shape -> Gather -> Unsqueeze -> Concat` chain, the
    /// shape of exporter-emitted plumbing, and leaves the weight input alone.
    #[test]
    fn test_folds_an_exporter_shape_chain_end_to_end() {
        let mut g = Graph::new();
        let x = g.add_node(
            Op::Input,
            vec![],
            DtypeRepr::F32,
            vec![Some(1), Some(64), Some(480), Some(640)],
        );
        let shape = g.add_node(
            Op::ShapeOf {
                start: 0,
                end: i64::MAX,
            },
            vec![x],
            DtypeRepr::I64,
            vec![Some(4)],
        );
        let idx = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![],
            },
            vec![],
            DtypeRepr::I64,
            vec![],
        );
        let dim = g.add_node(
            Op::Gather { axis: 0 },
            vec![shape, idx],
            DtypeRepr::I64,
            vec![],
        );
        let unsq = g.add_node(
            Op::Unsqueeze { axes: vec![0] },
            vec![dim],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        let tail = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(1)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        let cat = g.add_node(
            Op::Concat { axis: 0 },
            vec![unsq, tail],
            DtypeRepr::I64,
            vec![Some(2)],
        );

        let mut tensors = BTreeMap::new();
        tensors.insert(idx, int_td(&[], &[2])); // gather dim 2 (= 480)
        tensors.insert(tail, int_td(&[1], &[-1]));

        let (folded, report) =
            materialize_constants(&mut g, &tensors, &MaterializeConfig::default());

        assert_eq!(report.folded, 4, "Shape, Gather, Unsqueeze and Concat fold");
        assert!(
            report.unsupported.is_empty(),
            "unexpected unsupported ops: {:?}",
            report.unsupported_ranked()
        );
        // The chain collapses to the reshape target [480, -1].
        let cat_v = Value::decode(&folded[&cat]).expect("concat result decodes");
        assert_eq!(cat_v.as_ints(), vec![480, -1]);
        assert!(matches!(g.nodes[cat].op, Op::Constant { .. }));
        assert!(g.nodes[cat].inputs.is_empty());
        // The weight stays an input, so it can still be bound by name.
        assert!(matches!(g.nodes[x].op, Op::Input));
    }

    #[test]
    fn test_large_tensors_are_never_decoded() {
        let mut g = Graph::new();
        let w = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(1024)]);
        let id = g.add_node(Op::Identity, vec![w], DtypeRepr::F32, vec![Some(1024)]);

        let mut tensors = BTreeMap::new();
        tensors.insert(
            w,
            TensorData {
                dtype: DtypeRepr::F32,
                shape: vec![Some(1024)],
                bytes: vec![0u8; 4096],
            },
        );

        let cfg = MaterializeConfig { max_elements: 16 };
        let (folded, report) = materialize_constants(&mut g, &tensors, &cfg);
        assert!(folded.is_empty(), "nothing above the ceiling is folded");
        assert_eq!(report.folded, 0);
        // The `Identity` never became evaluable, because its operand was never decoded.
        assert!(matches!(g.nodes[id].op, Op::Identity));
    }
}
