/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

use protobuf::{CodedInputStream, Enum, Message};
use teeny_core::graph::{DtypeRepr, Graph, Op, Shape};

use crate::{
    errors::{Error, Result},
    onnx::onnx_proto3::{
        GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto,
    },
};

include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));

/// Parses ONNX protobuf models into a `teeny-core` [`Graph`].
pub struct Onnx {}

/// One element's width in bytes for a dtype as stored in a tensor's `raw_data`.
fn dtype_width(dtype: DtypeRepr) -> usize {
    match dtype {
        DtypeRepr::Bool | DtypeRepr::I8 | DtypeRepr::U8 => 1,
        DtypeRepr::I16 | DtypeRepr::U16 | DtypeRepr::F16 | DtypeRepr::BF16 => 2,
        DtypeRepr::I32 | DtypeRepr::U32 | DtypeRepr::F32 => 4,
        DtypeRepr::I64 | DtypeRepr::U64 | DtypeRepr::F64 => 8,
    }
}

/// Keeps the low `width` bytes of each value's little-endian encoding.
///
/// Narrowing is the right operation here rather than a lossy cast to check: the
/// carrier field's upper bytes are padding the exporter wrote, and a value that does
/// not fit its own dtype is a malformed model, not something to silently reinterpret.
fn narrow<T: Copy + Into<i128>>(values: &[T], width: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * width);
    for &v in values {
        let bytes = Into::<i128>::into(v).to_le_bytes();
        out.extend_from_slice(&bytes[..width]);
    }
    out
}

/// Reads a serialised `TensorProto` (a `.pb` file of the ONNX backend test data, say),
/// returning its name and values.
pub fn read_tensor_proto(bytes: &[u8]) -> Result<(String, TensorData)> {
    let t = TensorProto::parse_from_bytes(bytes)
        .map_err(|e| Error::InvalidModel(format!("parsing TensorProto: {e}")))?;
    let data = tensor_data(&t, &t.name)?;
    Ok((t.name.clone(), data))
}

/// Extracts a `TensorProto`'s values.
///
/// ONNX stores values either in `raw_data` or in a dtype-specific repeated field; both
/// are handled. Tensors whose data lives outside the model file are rejected rather
/// than silently returned empty.
fn tensor_data(t: &TensorProto, what: &str) -> Result<TensorData> {
    let dtype = tensor_dtype_to_dtype(t.data_type)?;
    let shape = dims_to_shape(&t.dims);

    if t.data_location.enum_value_or_default() == tensor_proto::DataLocation::EXTERNAL {
        let location = t
            .external_data
            .iter()
            .find(|e| e.key == "location")
            .map(|e| e.value.as_str())
            .unwrap_or("<unspecified>");
        return Err(Error::InvalidModel(format!(
            "{what}: values are stored externally (location '{location}'); \
             load the model with its external data resolved first"
        ))
        .into());
    }

    let bytes = if !t.raw_data.is_empty() {
        t.raw_data.clone()
    } else if !t.float_data.is_empty() {
        t.float_data.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !t.double_data.is_empty() {
        t.double_data.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !t.int64_data.is_empty() {
        t.int64_data.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !t.int32_data.is_empty() {
        // `int32_data` is a carrier field, not a dtype: ONNX packs bool, int8, uint8,
        // int16, uint16, float16 and bfloat16 into it one *int32* per element. Writing
        // all four bytes would store every element at four times its true stride, so
        // narrow each value to the dtype's own width.
        narrow(&t.int32_data, dtype_width(dtype).min(4))
    } else if !t.uint64_data.is_empty() {
        // Likewise for uint32, which shares this field with uint64.
        narrow(&t.uint64_data, dtype_width(dtype).min(8))
    } else {
        // A genuinely empty tensor is legal; anything else means a field this function
        // does not yet read.
        Vec::new()
    };

    Ok(TensorData {
        dtype,
        shape,
        bytes,
    })
}

/// A constant tensor's values. Alias of [`teeny_core::graph::ConstantTensor`], which
/// is where the type lives so that lowering can read materialised values without
/// depending on the ONNX crate.
pub type TensorData = teeny_core::graph::ConstantTensor;

/// Reads a Squeeze/Unsqueeze `axes` operand (input 1) out of the constants seen so
/// far, `None` if it is not a constant, or an error if it is a constant that cannot be
/// read as integers.
///
/// Opset 13 moved `axes` from an attribute to an operand. Almost always that operand
/// is a `Constant` or an initializer, and resolving it is what keeps the op's meaning
/// intact. It can legitimately be a graph input instead, as several models in the
/// ONNX conformance suite do: `axes` then changes the output *rank* at runtime, which
/// this graph IR records statically and so cannot represent. `None` reports that, and
/// the node keeps an empty `axes` as it did before; `crate::materialize` refuses to
/// fold an `Unsqueeze` in that state rather than treat it as an identity.
fn axes_operand(
    inputs: &[usize],
    tensors: &BTreeMap<usize, TensorData>,
    node: &str,
) -> Result<Option<Vec<i64>>> {
    let Some(&src) = inputs.get(1) else {
        return Ok(None);
    };
    let Some(t) = tensors.get(&src) else {
        return Ok(None);
    };
    let axes = t.to_i64_vec().map_err(|reason| Error::UnreadableConstant {
        context: format!("node '{node}': axes operand (node {src})"),
        reason,
    })?;
    Ok(Some(axes))
}

/// Everything an ONNX import produced.
///
/// [`Onnx::from_path_with_tensors`] returns only the graph and its tensor values,
/// which is all most callers need. An `If` needs more: its branch bodies cannot be
/// imported until the condition is known, and the condition is usually only known
/// after constant folding, so they are carried here for [`crate::branches`].
pub struct Import {
    /// The imported graph.
    pub graph: Graph,
    /// Tensor values for initializers and `Constant` nodes, keyed by node index.
    pub tensors: BTreeMap<usize, TensorData>,
    /// Tensor name -> the index of the node producing it.
    pub value_to_node: BTreeMap<String, usize>,
    /// Dtype and shape by tensor name, as the file declared them.
    pub value_info_map: BTreeMap<String, (DtypeRepr, Shape)>,
    /// Unresolved `If` nodes, keyed by node index. See [`crate::branches`].
    pub branches: BTreeMap<usize, IfBranches>,
    /// The file's ONNX opset version for the default domain, or zero when it named
    /// none. Some attributes changed their default between versions, so a later pass
    /// that imports a subgraph needs the same number the first pass used.
    pub opset: i64,
}

/// The two branch bodies of an `If` node, kept unimported.
///
/// Only the taken branch may be imported. The branches of a traced guard are often
/// valid only under their own condition, so importing both and selecting afterwards
/// would put an operation in the graph that cannot run: the squeeze in
/// `if dim == 1 { squeeze(x, 1) } else { x }` is an error on the path where the
/// condition is false.
pub struct IfBranches {
    then_branch: GraphProto,
    else_branch: GraphProto,
}

impl IfBranches {
    /// Reads both branch subgraphs off an `If` node, or `None` if either is absent.
    fn from_node(node: &NodeProto) -> Option<Self> {
        let branch = |name: &str| {
            node.attribute
                .iter()
                .find(|a| a.name == name)
                .and_then(|a| a.g.as_ref().cloned())
        };
        Some(Self {
            then_branch: branch("then_branch")?,
            else_branch: branch("else_branch")?,
        })
    }

    /// The body that runs when the condition is `cond`.
    pub(crate) fn taken(&self, cond: bool) -> &GraphProto {
        if cond {
            &self.then_branch
        } else {
            &self.else_branch
        }
    }
}

/// Everything the per-node import step reads or writes.
///
/// Bundled into a context because a node is imported from two places: the model's own
/// graph, and the body of an `If` branch that turns out to be taken (see
/// [`crate::branches`]). A branch body resolves names it does not define against the
/// enclosing graph, which is exactly ONNX's scoping rule, so sharing `value_to_node`
/// between the two is what makes outer-scope capture work rather than a special case.
pub(crate) struct ImportCtx<'a> {
    pub(crate) graph: Graph,
    pub(crate) tensors: BTreeMap<usize, TensorData>,
    /// Tensor name -> the index of the node producing it. Doubles as the enclosing
    /// scope when a subgraph is imported.
    pub(crate) value_to_node: BTreeMap<String, usize>,
    /// Later outputs of multi-output nodes, which are not computed: tensor name -> its
    /// producer, for the error a consumer of one gets.
    dropped_outputs: BTreeMap<String, String>,
    pub(crate) value_info_map: BTreeMap<String, (DtypeRepr, Shape)>,
    initializer_map: BTreeMap<&'a str, &'a TensorProto>,
    /// The file's ONNX opset version, which decides what some absent attributes mean.
    /// Zero when the model did not say, where the newest meaning is assumed.
    pub(crate) opset: i64,
    /// The shared [`Op::Omitted`] node, once something has needed one.
    omitted: Option<usize>,
}

impl<'a> ImportCtx<'a> {
    /// Rewrites a `Split` into one `Slice` per output, or `None` if the sizes are not
    /// known at import time.
    ///
    /// Each output is a contiguous run along `axis`, so a slice is exactly what it is,
    /// and the slice path is one the lowering already covers. The sizes come from the
    /// `split` operand when there is one, which is where opset 13 moved them from an
    /// attribute, and otherwise from equal shares of the axis.
    ///
    /// The new nodes get no shape. Shapes are not known during import of an export
    /// that carries no value_info, and `Slice`'s own inference works them out later
    /// from the same operands this builds; recording the axis as it was given, negative
    /// or not, is what keeps this from needing the rank.
    fn expand_split(
        &mut self,
        inputs: &[usize],
        axis: i64,
        num_outputs: usize,
        name: &str,
    ) -> Result<Option<Vec<usize>>> {
        let Some(&data) = inputs.first() else {
            return Ok(None);
        };

        // The explicit sizes, if the operand is one of the tensors already collected.
        let explicit: Option<Vec<usize>> = inputs
            .get(1)
            .and_then(|src| self.tensors.get(src))
            .map(|td| td.to_i64_vec())
            .transpose()
            .map_err(|reason| Error::UnreadableConstant {
                context: format!("Split '{name}': split sizes operand"),
                reason,
            })?
            .filter(|values| !values.is_empty())
            .and_then(|values| values.iter().map(|&v| usize::try_from(v).ok()).collect());

        let sizes = match explicit {
            Some(sizes) => sizes,
            None => {
                // Equal shares, which is what ONNX does with no sizes given. This is
                // the one case that needs the extent, and so the shape.
                let shape: Vec<usize> = match self.graph.nodes[data].shape.iter().copied().collect()
                {
                    Some(s) => s,
                    None => return Ok(None),
                };
                let rank = shape.len();
                let ax = if axis < 0 { axis + rank as i64 } else { axis };
                let Ok(ax) = usize::try_from(ax).map(|a| a.min(rank)) else {
                    return Ok(None);
                };
                if ax >= rank || num_outputs == 0 || !shape[ax].is_multiple_of(num_outputs) {
                    return Ok(None);
                }
                vec![shape[ax] / num_outputs; num_outputs]
            }
        };
        if sizes.is_empty() {
            return Ok(None);
        }

        let dtype = self.graph.nodes[data].dtype;
        let mut ids = Vec::with_capacity(sizes.len());
        let mut start = 0usize;
        for (which, &len) in sizes.iter().enumerate() {
            let starts = self.scalar_i64_constant(start as i64);
            let ends = self.scalar_i64_constant((start + len) as i64);
            let axes = self.scalar_i64_constant(axis);
            let id =
                self.graph
                    .add_node(Op::Slice, vec![data, starts, ends, axes], dtype, Vec::new());
            if !name.is_empty() {
                self.graph.names.insert(id, format!("{name}:{which}"));
            }
            ids.push(id);
            start += len;
        }
        Ok(Some(ids))
    }

    /// A length-1 I64 `Constant` node holding `value`, with its bytes recorded so the
    /// constant folder and the lowering can both read it.
    fn scalar_i64_constant(&mut self, value: i64) -> usize {
        let id = self.graph.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(1)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        self.tensors.insert(
            id,
            TensorData {
                dtype: DtypeRepr::I64,
                shape: vec![Some(1)],
                bytes: value.to_le_bytes().to_vec(),
            },
        );
        id
    }

    /// Builds an empty context whose name metadata comes from `onnx_graph`.
    fn new(onnx_graph: &'a GraphProto, opset: i64) -> Result<Self> {
        let mut value_info_map: BTreeMap<String, (DtypeRepr, Shape)> = BTreeMap::new();
        for info in onnx_graph
            .input
            .iter()
            .chain(&onnx_graph.value_info)
            .chain(&onnx_graph.output)
        {
            if let Some(meta) = value_info_meta(info)? {
                value_info_map.insert(info.name.clone(), meta);
            }
        }
        Ok(Self {
            graph: Graph::new(),
            tensors: BTreeMap::new(),
            value_to_node: BTreeMap::new(),
            dropped_outputs: BTreeMap::new(),
            value_info_map,
            opset,
            initializer_map: onnx_graph
                .initializer
                .iter()
                .map(|t| (t.name.as_str(), t))
                .collect(),
            omitted: None,
        })
    }

    /// Rebuilds a context around an existing graph, so a subgraph's nodes can be
    /// imported into it.
    ///
    /// `value_to_node` carries the enclosing scope, which is what lets a branch body
    /// reference tensors it does not define. The subgraph's own `value_info` and
    /// initializers take precedence for the names it does define, as ONNX scoping
    /// says they should.
    pub(crate) fn reopen(
        graph: Graph,
        tensors: BTreeMap<usize, TensorData>,
        value_to_node: BTreeMap<String, usize>,
        mut value_info_map: BTreeMap<String, (DtypeRepr, Shape)>,
        subgraph: &'a GraphProto,
        opset: i64,
    ) -> Result<Self> {
        for info in subgraph
            .input
            .iter()
            .chain(&subgraph.value_info)
            .chain(&subgraph.output)
        {
            if let Some(meta) = value_info_meta(info)? {
                value_info_map.insert(info.name.clone(), meta);
            }
        }
        Ok(Self {
            graph,
            tensors,
            value_to_node,
            dropped_outputs: BTreeMap::new(),
            value_info_map,
            opset,
            initializer_map: subgraph
                .initializer
                .iter()
                .map(|t| (t.name.as_str(), t))
                .collect(),
            omitted: None,
        })
    }

    /// Adds `graph`'s initializers and inputs as `Op::Input` nodes.
    pub(crate) fn add_initializers_and_inputs(&mut self, onnx_graph: &GraphProto) -> Result<()> {
        let initializer_names: BTreeSet<&str> = onnx_graph
            .initializer
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        for init in &onnx_graph.initializer {
            let dtype = tensor_dtype_to_dtype(init.data_type)?;
            let shape = dims_to_shape(&init.dims);
            let id = self.graph.add_node(Op::Input, vec![], dtype, shape);
            // Record the initializer's ONNX name. Without it an imported graph has no
            // way to say which `Op::Input` is which weight, so nothing downstream can
            // bind tensor data to it — the node carries dtype and shape only.
            self.graph.names.insert(id, init.name.clone());
            self.tensors.insert(
                id,
                tensor_data(init, &format!("initializer '{}'", init.name))?,
            );
            self.value_to_node.insert(init.name.clone(), id);
        }

        for input in &onnx_graph.input {
            if initializer_names.contains(input.name.as_str()) {
                continue;
            }
            let (dtype, shape) = self
                .value_info_map
                .get(&input.name)
                .cloned()
                .unwrap_or((DtypeRepr::F32, vec![]));
            let id = self.graph.add_node(Op::Input, vec![], dtype, shape);
            // Name the model's own inputs too, so every Op::Input in an imported graph
            // is identifiable, whether it is a weight or a feed.
            self.graph.names.insert(id, input.name.clone());
            self.value_to_node.insert(input.name.clone(), id);
        }
        Ok(())
    }

    /// The graph's single `Op::Omitted` node, created on first use.
    ///
    /// One node serves every omitted operand in the model: it carries no value and no
    /// shape, so there is nothing to distinguish one from another, and sharing it
    /// keeps a model with hundreds of them from growing hundreds of nodes.
    fn omitted_node(&mut self) -> usize {
        match self.omitted {
            Some(id) => id,
            None => {
                let id = self
                    .graph
                    .add_node(Op::Omitted, vec![], DtypeRepr::F32, Vec::new());
                self.omitted = Some(id);
                id
            }
        }
    }

    /// Imports one ONNX node, returning its index in the graph.
    pub(crate) fn import_node(&mut self, node: &NodeProto) -> Result<usize> {
        let mut inputs = Vec::with_capacity(node.input.len());
        for input_name in &node.input {
            if input_name.is_empty() {
                // An omitted optional operand. ONNX writes an empty name and keeps the
                // positions of the operands after it, and those positions are the only
                // thing that says what each operand means. Skipping the entry shifts
                // every later operand one place left, which is not a missing-feature
                // kind of wrong: `Clip(x, "", max)` becomes `Clip(x, max)`, so the
                // upper bound is read as the lower one and the tensor is clamped to
                // the wrong constant with nothing to indicate it.
                //
                // Trailing omissions could be dropped safely, but they are not worth a
                // special case: a consumer that handles `Omitted` in one position
                // handles it in all of them.
                inputs.push(self.omitted_node());
                continue;
            }
            let input_id = match self.value_to_node.get(input_name) {
                Some(&id) => id,
                None => {
                    if let Some(producer) = self.dropped_outputs.get(input_name) {
                        return Err(Error::Unsupported(format!(
                            "Node '{}' ({}) reads '{input_name}', a later output of {producer}; \
                             only a node's first output is computed",
                            node.name, node.op_type
                        ))
                        .into());
                    }
                    return Err(Error::InvalidModel(format!(
                        "Node '{}' references unknown input '{}'",
                        node.name, input_name
                    ))
                    .into());
                }
            };
            inputs.push(input_id);
        }

        // Some ops (Constant, SequenceEmpty, OptionalHasElement) have zero
        // tensor inputs by design — allow them through.
        let zero_input_allowed = matches!(
            node.op_type.as_str(),
            "Constant" | "SequenceEmpty" | "OptionalHasElement"
        );
        if inputs.is_empty() && !zero_input_allowed {
            return Err(Error::InvalidModel(format!(
                "Node '{}' (op='{}') has no resolvable inputs",
                node.name, node.op_type
            ))
            .into());
        }

        let first_output = node
            .output
            .iter()
            .find(|name| !name.is_empty())
            .ok_or_else(|| {
                Error::InvalidModel(format!("Node '{}' has no output names", node.name))
            })?;

        // A `Constant`'s dtype and shape are stated exactly by its own `value`
        // attribute, so take them from there rather than from `value_info` or the
        // fallback below. Without this a `Constant` with no `value_info` entry
        // becomes an F32 scalar whatever it really holds, which puts the node's
        // dtype and shape at odds with `Op::Constant`'s own fields and with the
        // value in the tensor map.
        let constant_type = (node.op_type == "Constant")
            .then(|| node.attribute.iter().find(|a| a.name == "value"))
            .flatten()
            .map(|attr| {
                anyhow::Ok((
                    tensor_dtype_to_dtype(attr.t.data_type)?,
                    dims_to_shape(&attr.t.dims),
                ))
            })
            .transpose()?;

        let op = map_node_op(node, &self.initializer_map, self.opset)?;

        // Opset 13 moved `axes` from an attribute to input 1 for Squeeze and
        // Unsqueeze. `map_node_op` can only read the attribute, so resolve the
        // operand here, where the constant values collected so far are in scope.
        // Without this, every Squeeze/Unsqueeze in an opset-13-or-later export
        // silently gets `axes: []`, which is a different operation: for Squeeze
        // it means "squeeze every size-1 axis", and for Unsqueeze it is not an
        // operation at all.
        let op = match op {
            Op::Squeeze { axes } if axes.is_empty() && node.input.len() > 1 => Op::Squeeze {
                axes: axes_operand(&inputs, &self.tensors, &node.name)?.unwrap_or(axes),
            },
            Op::Unsqueeze { axes } if axes.is_empty() && node.input.len() > 1 => Op::Unsqueeze {
                axes: axes_operand(&inputs, &self.tensors, &node.name)?.unwrap_or(axes),
            },
            other => other,
        };

        // A logical operator's output is a bool whatever its operands are, and
        // nothing downstream can tell a wrongly recorded dtype from a real one: it
        // decides how wide the output buffer is and how the kernel reads it. An
        // export without value_info otherwise leaves these carrying their first
        // operand's dtype, which in FoundationStereo makes every `And` an I64 node
        // over two Bool operands.
        //
        // The comparisons are left as they are on purpose, though ONNX says their
        // output is a bool too: their kernels write 0 or 1 in the operands' own
        // dtype, and lowering reads the node's dtype to pick the instantiation, so
        // calling them Bool here would stop them lowering at all. That wants fixing
        // together, in the kernels and here.
        // Comparisons return bools too. Their kernels write 0/1 in the operands' dtype,
        // so `rewrite::compare_then_cast` later computes them at that dtype and casts.
        let logical_output = matches!(
            op,
            Op::And
                | Op::Or
                | Op::Xor
                | Op::Equal
                | Op::Greater
                | Op::GreaterOrEqual
                | Op::Less
                | Op::LessOrEqual
        );

        let (dtype, shape) = constant_type
            .or_else(|| self.value_info_map.get(first_output).cloned())
            .map(|(d, s)| (if logical_output { DtypeRepr::Bool } else { d }, s))
            .unwrap_or_else(|| {
                // The shape is left to shape inference, as for every other
                // broadcasting op: these take the broadcast of their operands, so
                // copying the first one's would be wrong whenever they differ.
                if logical_output {
                    return (DtypeRepr::Bool, Vec::new());
                }
                if inputs.is_empty() {
                    // Zero-input op with no value_info — fall back to F32 scalar.
                    return (DtypeRepr::F32, vec![]);
                }
                let input_node = &self.graph.nodes[inputs[0]];
                // Copying the first input's shape is right for the elementwise and
                // normalisation ops and wrong for everything else, and a wrong
                // shape is worse than an absent one: nothing downstream can tell
                // it is wrong, so it propagates silently. For the rest, record the
                // dtype, which the first input does determine for most ops, and
                // leave the shape unknown for a later pass to work out (see
                // `crate::materialize`).
                if op.preserves_input_shape() {
                    (input_node.dtype, input_node.shape.clone())
                } else {
                    (input_node.dtype, Vec::new())
                }
            });
        // A Cast's output has the dtype it casts to, whatever its operand's is. Without
        // value_info the fallback above copies the operand's, which makes every
        // converting Cast look like a no-op copy at the wrong width.
        // Likewise a Where's output has its values' dtype (operand 1), not its
        // condition's, which is Bool.
        let dtype = match &op {
            Op::Cast { to } => *to,
            Op::Where if dtype == DtypeRepr::Bool => {
                inputs.get(1).map_or(dtype, |&i| self.graph.nodes[i].dtype)
            }
            _ => dtype,
        };

        // A multi-output op cannot be one node: every output name would map to the
        // same node, and so to the same buffer, and a consumer of output 1 would
        // silently read output 0. Split is the one the exports use, and each of its
        // outputs is a slice of the input along one axis, so it becomes that: one
        // Slice per output, each with its own node and its own name.
        if let Op::Split { axis, num_outputs } = &op
            && node.output.len() > 1
            && let Some(ids) = self.expand_split(&inputs, *axis, *num_outputs, &node.name)?
        {
            for (output_name, id) in node.output.iter().zip(ids) {
                if !output_name.is_empty() {
                    self.value_to_node.insert(output_name.clone(), id);
                }
            }
            return Ok(*self.value_to_node.get(&node.output[0]).unwrap_or(&0));
        }

        let node_id = self.graph.add_node(op, inputs, dtype, shape);

        if !node.name.is_empty() {
            self.graph.names.insert(node_id, node.name.clone());
        }

        // `Constant` and `ConstantOfShape` carry their values in a `value` attribute,
        // which map_node_op reads for dtype and shape only. Keep the values here so the
        // nodes can be materialised (see `crate::materialize`). For `ConstantOfShape`
        // the attribute is the scalar fill value rather than the whole tensor.
        if matches!(node.op_type.as_str(), "Constant" | "ConstantOfShape")
            && let Some(attr) = node.attribute.iter().find(|a| a.name == "value")
        {
            self.tensors.insert(
                node_id,
                tensor_data(&attr.t, &format!("{} '{}'", node.op_type, node.name))?,
            );
        }

        // The node computes its first output only. A multi-output op's later outputs
        // (LayerNormalization's Mean and InvStdDev, MaxPool's Indices, TopK's indices)
        // are left without a producer, so a consumer of one fails to import rather than
        // silently reading the first output's buffer. Split, whose outputs are all data,
        // is expanded into one node per output above.
        if let Some(first) = node.output.first().filter(|n| !n.is_empty()) {
            self.value_to_node.insert(first.clone(), node_id);
        }
        for (k, name) in node.output.iter().enumerate().skip(1) {
            if !name.is_empty() {
                self.dropped_outputs.insert(
                    name.clone(),
                    format!("node '{}' ({}, output #{k})", node.name, node.op_type),
                );
            }
        }
        Ok(node_id)
    }
}

impl Onnx {
    /// Reads and parses the `.onnx` file at `path`.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Graph> {
        let mut file = File::open(path)?;
        Self::from_reader(&mut file)
    }

    /// As [`Onnx::from_path`], but also returns the model's tensor values.
    ///
    /// The graph alone describes topology: every weight is an `Op::Input` and every
    /// `Constant` carries only a dtype and shape, so a graph on its own cannot be run.
    /// The returned map supplies the missing values, keyed by graph node index, for
    /// both initializers and `Constant` nodes.
    pub fn from_path_with_tensors(
        path: impl AsRef<Path>,
    ) -> Result<(Graph, BTreeMap<usize, TensorData>)> {
        let mut file = File::open(path)?;
        Self::from_reader_with_tensors(&mut file)
    }

    /// Parses an ONNX model (`ModelProto`, protobuf binary format) from `reader` into a `Graph`.
    ///
    /// Errors if the bytes aren't valid ONNX protobuf, the model has no graph, or the graph uses
    /// an op this crate doesn't support yet.
    pub fn from_reader(reader: &mut impl Read) -> Result<Graph> {
        Self::from_reader_with_tensors(reader).map(|(graph, _)| graph)
    }

    /// As [`Onnx::from_reader`], but also returns the model's tensor values.
    /// See [`Onnx::from_path_with_tensors`].
    pub fn from_reader_with_tensors(
        reader: &mut impl Read,
    ) -> Result<(Graph, BTreeMap<usize, TensorData>)> {
        Self::import_from_reader(reader).map(|i| (i.graph, i.tensors))
    }

    /// As [`Onnx::from_path_with_tensors`], but returns everything the import learned,
    /// including the `If` branch bodies that only a later pass can resolve.
    /// See [`Import`].
    pub fn import_from_path(path: impl AsRef<Path>) -> Result<Import> {
        let mut file = File::open(path)?;
        Self::import_from_reader(&mut file)
    }

    /// As [`Onnx::import_from_path`], with `constants` giving values for named graph
    /// inputs, which are then imported as if they were initializers.
    ///
    /// For a model whose metadata or weights arrive as inputs (the ONNX backend test
    /// suite passes every operand that way) this is what an exporter would have written:
    /// ops resolved during import, such as Split's sizes, see the values.
    pub fn import_from_path_with_constants(
        path: impl AsRef<Path>,
        constants: &BTreeMap<String, TensorData>,
    ) -> Result<Import> {
        let mut file = File::open(path)?;
        Self::import_inner(&mut file, Some(constants))
    }

    /// As [`Onnx::import_from_path`], reading from `reader`.
    pub fn import_from_reader(reader: &mut impl Read) -> Result<Import> {
        Self::import_inner(reader, None)
    }

    fn import_inner(
        reader: &mut impl Read,
        constants: Option<&BTreeMap<String, TensorData>>,
    ) -> Result<Import> {
        let mut reader = CodedInputStream::new(reader);
        let model = ModelProto::parse_from(&mut reader)?;
        let onnx_graph = model
            .graph
            .as_ref()
            .ok_or_else(|| Error::InvalidModel("Model is missing graph field".to_string()))?;

        // The default ONNX domain's version. An empty domain name is "ai.onnx", and a
        // model that names no version leaves this zero, where the newest meaning of an
        // absent attribute is the better guess.
        let opset = model
            .opset_import
            .iter()
            .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
            .map(|o| o.version)
            .unwrap_or(0);
        let mut ctx = ImportCtx::new(onnx_graph, opset)?;
        let mut branches: BTreeMap<usize, IfBranches> = BTreeMap::new();
        ctx.add_initializers_and_inputs(onnx_graph)?;
        for (name, value) in constants.into_iter().flatten() {
            if let Some(&id) = ctx.value_to_node.get(name) {
                ctx.tensors.insert(id, value.clone());
            }
        }

        for node in &onnx_graph.node {
            let node_id = ctx.import_node(node)?;
            // An `If`'s branch bodies are kept rather than imported: which one applies
            // depends on the condition, and that is not known until constants are
            // folded. Discarding them here would make the decision unrecoverable, so
            // they are carried out of the import for `crate::branches` to resolve.
            if node.op_type == "If"
                && let Some(b) = IfBranches::from_node(node)
            {
                branches.insert(node_id, b);
            }
        }

        Ok(Import {
            graph: ctx.graph,
            tensors: ctx.tensors,
            value_to_node: ctx.value_to_node,
            value_info_map: ctx.value_info_map,
            branches,
            opset,
        })
    }
}

fn value_info_meta(info: &ValueInfoProto) -> Result<Option<(DtypeRepr, Shape)>> {
    let ty = info
        .type_
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("ValueInfoProto has no type".to_string()))?;
    if !ty.has_tensor_type() {
        return Ok(None);
    }
    let tensor = ty.tensor_type();
    let dtype = tensor_dtype_to_dtype(tensor.elem_type)?;
    let shape = tensor.shape.as_ref().map_or_else(Vec::new, |shape| {
        shape
            .dim
            .iter()
            .map(|d| {
                if d.has_dim_value() && d.dim_value() >= 0 {
                    Some(d.dim_value() as usize)
                } else {
                    None
                }
            })
            .collect()
    });

    Ok(Some((dtype, shape)))
}

fn dims_to_shape(dims: &[i64]) -> Shape {
    dims.iter()
        .map(|d| if *d >= 0 { Some(*d as usize) } else { None })
        .collect()
}

fn tensor_dtype_to_dtype(dtype: i32) -> Result<DtypeRepr> {
    Ok(match tensor_proto::DataType::from_i32(dtype) {
        Some(tensor_proto::DataType::BOOL) => DtypeRepr::Bool,
        Some(tensor_proto::DataType::INT8) => DtypeRepr::I8,
        Some(tensor_proto::DataType::INT16) => DtypeRepr::I16,
        Some(tensor_proto::DataType::INT32) => DtypeRepr::I32,
        Some(tensor_proto::DataType::INT64) => DtypeRepr::I64,
        Some(tensor_proto::DataType::UINT8) => DtypeRepr::U8,
        Some(tensor_proto::DataType::UINT16) => DtypeRepr::U16,
        Some(tensor_proto::DataType::UINT32) => DtypeRepr::U32,
        Some(tensor_proto::DataType::UINT64) => DtypeRepr::U64,
        Some(tensor_proto::DataType::FLOAT16) => DtypeRepr::F16,
        Some(tensor_proto::DataType::BFLOAT16) => DtypeRepr::BF16,
        Some(tensor_proto::DataType::DOUBLE) => DtypeRepr::F64,
        Some(tensor_proto::DataType::FLOAT) | None => DtypeRepr::F32,
        // STRING tensors — represent as U8 (byte storage) for graph purposes.
        Some(tensor_proto::DataType::STRING) => DtypeRepr::U8,
        // Sub-byte and float8 types — map to nearest supported type.
        Some(tensor_proto::DataType::FLOAT8E4M3FN)
        | Some(tensor_proto::DataType::FLOAT8E4M3FNUZ)
        | Some(tensor_proto::DataType::FLOAT8E5M2)
        | Some(tensor_proto::DataType::FLOAT8E5M2FNUZ) => DtypeRepr::F16,
        Some(_) => {
            // Unknown or newer sub-byte types (INT4, UINT4, INT2, UINT2, FLOAT4…).
            // Map to I8/U8 as a safe placeholder for graph-building purposes.
            if dtype % 2 == 0 {
                DtypeRepr::U8
            } else {
                DtypeRepr::I8
            }
        }
    })
}

fn get_attr_int(node: &NodeProto, name: &str, default: i64) -> i64 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

/// Reads a float attribute, distinguishing absent from zero.
///
/// `get_attr_float`'s default cannot do that, and for a bound it matters: a `Clip`
/// with no `min` leaves values alone, while one with `min = 0` is a ReLU.
fn get_attr_float_opt(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

fn get_attr_float(node: &NodeProto, name: &str, default: f32) -> f32 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.f)
        .unwrap_or(default)
}

fn get_attr_ints<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [i64]> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.as_slice())
}

fn get_attr_string<'a>(node: &'a NodeProto, name: &str, default: &'a str) -> &'a str {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| std::str::from_utf8(&a.s).ok())
        .unwrap_or(default)
}

/// Refuses pooling and convolution attributes the IR cannot express, so they fail to
/// import instead of being computed as if absent: `auto_pad` other than NOTSET/VALID
/// (its padding depends on the input's size, which the importer does not know yet),
/// `ceil_mode`, `dilations` other than 1, and, when `pads_supported` is false or the
/// pads are asymmetric, any padding at all.
fn refuse_window_attrs(node: &NodeProto, what: &str, pads_supported: bool) -> Result<()> {
    let refuse = |why: String| -> Result<()> {
        Err(Error::Unsupported(format!("Node '{}' ({what}): {why}", node.name)).into())
    };
    let auto_pad = get_attr_string(node, "auto_pad", "NOTSET");
    if !matches!(auto_pad, "NOTSET" | "VALID" | "") {
        return refuse(format!("auto_pad {auto_pad} is not supported"));
    }
    if what != "Conv" && get_attr_int(node, "ceil_mode", 0) != 0 {
        return refuse("ceil_mode is not supported".into());
    }
    if let Some(d) = get_attr_ints(node, "dilations")
        && d.iter().any(|&v| v != 1)
    {
        return refuse(format!("dilations {d:?} are not supported"));
    }
    if let Some(p) = get_attr_ints(node, "pads") {
        let half = p.len() / 2;
        if p.iter().any(|&v| v != 0) && (!pads_supported || p[..half] != p[half..]) {
            return refuse(format!(
                "pads {p:?} are not supported (only symmetric padding, and only in 2-D)"
            ));
        }
    }
    Ok(())
}

fn map_node_op(
    node: &NodeProto,
    initializers: &BTreeMap<&str, &TensorProto>,
    opset: i64,
) -> Result<Op> {
    let op = match node.op_type.as_str() {
        // ---------------------------------------------------------------
        // Existing ops (previously supported)
        // ---------------------------------------------------------------
        "Relu" => Op::Relu,
        "Sigmoid" => Op::Sigmoid,
        "Tanh" => Op::Tanh,
        "Add" => Op::Add,
        "Flatten" => {
            // A negative axis counts from the end, which needs the input's rank; the
            // importer has no shapes yet, so refuse it rather than guess.
            let axis = get_attr_int(node, "axis", 1);
            if axis < 0 {
                return Err(Error::Unsupported(format!(
                    "Flatten '{}' with negative axis {axis} is not supported",
                    node.name
                ))
                .into());
            }
            Op::Flatten {
                axis: axis as usize,
            }
        }
        "Softmax" => {
            // ONNX changed the default in opset 13, from 1 to -1, so the file's own
            // opset decides what an absent attribute means. The axis is kept as
            // written: a negative one is resolved against the rank at lowering, where
            // the rank is known, and clamping it to 0 normalized the wrong axis.
            let default = if opset >= 13 { -1 } else { 1 };
            Op::Softmax {
                dim: get_attr_int(node, "axis", default),
            }
        }

        "Conv" => {
            refuse_window_attrs(node, "Conv", true)?;
            let weight_name = node.input.get(1).map(|s| s.as_str()).unwrap_or("");
            // Weight dims: [out_ch, in_ch/groups, k...] if from an initializer,
            // otherwise inferred from the kernel_shape attribute.
            let weight_dims: Vec<i64> = if let Some(w) = initializers.get(weight_name) {
                w.dims.clone()
            } else {
                let ks = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
                let mut d = vec![0i64, 0i64]; // out/in channels unknown
                d.extend_from_slice(ks);
                d
            };

            let has_bias = node.input.get(2).is_some_and(|s| !s.is_empty());
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            let groups = get_attr_int(node, "group", 1).max(1) as usize;
            let weight = &weight_dims[..];
            let ndim = weight.len();

            if ndim == 3 {
                Op::Conv1d {
                    in_channels: weight.get(1).copied().unwrap_or(0) as usize * groups,
                    out_channels: weight.first().copied().unwrap_or(0) as usize,
                    kernel_l: weight.get(2).copied().unwrap_or(1) as usize,
                    stride: strides.first().copied().unwrap_or(1).max(1) as usize,
                    padding: pads.first().copied().unwrap_or(0).max(0) as usize,
                    has_bias,
                }
            } else if ndim == 4 {
                Op::Conv2d {
                    in_channels: weight.get(1).copied().unwrap_or(0) as usize * groups,
                    out_channels: weight.first().copied().unwrap_or(0) as usize,
                    kernel_h: weight.get(2).copied().unwrap_or(1) as usize,
                    kernel_w: weight.get(3).copied().unwrap_or(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    padding_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                    padding_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                    groups,
                    has_bias,
                }
            } else if ndim == 5 {
                Op::Conv3d {
                    in_channels: weight.get(1).copied().unwrap_or(0) as usize * groups,
                    out_channels: weight.first().copied().unwrap_or(0) as usize,
                    kernel_d: weight.get(2).copied().unwrap_or(1) as usize,
                    kernel_h: weight.get(3).copied().unwrap_or(1) as usize,
                    kernel_w: weight.get(4).copied().unwrap_or(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(1).max(1) as usize,
                    padding_d: pads.first().copied().unwrap_or(0).max(0) as usize,
                    padding_h: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                    padding_w: pads.get(2).copied().unwrap_or(0).max(0) as usize,
                    has_bias,
                }
            } else {
                return Err(Error::Unsupported(format!(
                    "Conv node '{}': unsupported weight rank {}",
                    node.name, ndim
                ))
                .into());
            }
        }

        "ConvTranspose" => {
            let weight_name = node.input.get(1).map(|s| s.as_str()).unwrap_or("");
            let (out_channels, kernel_h, kernel_w) = if let Some(w) = initializers.get(weight_name)
            {
                let nd = w.dims.len();
                if nd >= 4 {
                    (w.dims[1] as usize, w.dims[2] as usize, w.dims[3] as usize)
                } else {
                    (0, 1, 1)
                }
            } else {
                (0, 1, 1)
            };
            let has_bias = node.input.get(2).is_some_and(|s| !s.is_empty());
            let strides = get_attr_ints(node, "strides").unwrap_or(&[1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            let out_pads = get_attr_ints(node, "output_padding").unwrap_or(&[0, 0]);
            let groups = get_attr_int(node, "group", 1).max(1) as usize;
            let in_channels = if let Some(w) = initializers.get(weight_name) {
                w.dims.first().copied().unwrap_or(0) as usize * groups
            } else {
                0
            };
            Op::ConvTranspose {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                padding_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                padding_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                output_padding_h: out_pads.first().copied().unwrap_or(0).max(0) as usize,
                output_padding_w: out_pads.get(1).copied().unwrap_or(0).max(0) as usize,
                groups,
                has_bias,
            }
        }

        "MaxPool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            refuse_window_attrs(node, "MaxPool", kernel.len() == 2)?;
            if node.output.len() > 1 && !node.output[1].is_empty() {
                return Err(Error::Unsupported(format!(
                    "Node '{}' (MaxPool): the Indices output is not supported",
                    node.name
                ))
                .into());
            }
            // ONNX's default stride is 1 along each axis (PyTorch's is the kernel size).
            let ones = vec![1i64; kernel.len()];
            let strides = get_attr_ints(node, "strides").unwrap_or(&ones);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            match kernel.len() {
                1 => Op::MaxPool1d {
                    kernel_l: kernel[0].max(1) as usize,
                    stride: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                },
                3 => Op::MaxPool3d {
                    kernel_d: kernel[0].max(1) as usize,
                    kernel_h: kernel[1].max(1) as usize,
                    kernel_w: kernel[2].max(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(kernel[1]).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(kernel[2]).max(1) as usize,
                },
                _ => Op::MaxPool2d {
                    kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                    kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    pad_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                    pad_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                },
            }
        }

        "AveragePool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            refuse_window_attrs(node, "AveragePool", kernel.len() == 2)?;
            // ONNX's default stride is 1 along each axis (PyTorch's is the kernel size).
            let ones = vec![1i64; kernel.len()];
            let strides = get_attr_ints(node, "strides").unwrap_or(&ones);
            let pads = get_attr_ints(node, "pads").unwrap_or(&[0, 0, 0, 0]);
            let count_include_pad = get_attr_int(node, "count_include_pad", 0) != 0;
            // `pads` is [h_begin, w_begin, h_end, w_end] and may be asymmetric, which
            // the IR cannot express and the kernel cannot do. Refusing is the point:
            // keeping only the begins would shift the window by the difference and
            // average a different set of pixels with nothing to show it.
            if kernel.len() == 2 && pads.len() == 4 && (pads[0] != pads[2] || pads[1] != pads[3]) {
                return Err(Error::Unsupported(format!(
                    "Node '{}' (AveragePool) has asymmetric pads {pads:?}; only \
                     symmetric padding is supported, and treating these as symmetric \
                     would average a different window",
                    node.name
                ))
                .into());
            }
            // `ceil_mode` rounds the output extent up, which adds a partial window at
            // the far edge whose interaction with `count_include_pad` ONNX does not
            // pin down. Refused rather than guessed at.
            if get_attr_int(node, "ceil_mode", 0) != 0 {
                return Err(Error::Unsupported(format!(
                    "Node '{}' (AveragePool) sets ceil_mode, which is not supported",
                    node.name
                ))
                .into());
            }
            match kernel.len() {
                1 => Op::AvgPool1d {
                    kernel_l: kernel[0].max(1) as usize,
                    stride: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                },
                3 => Op::AvgPool3d {
                    kernel_d: kernel[0].max(1) as usize,
                    kernel_h: kernel[1].max(1) as usize,
                    kernel_w: kernel[2].max(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(kernel[1]).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(kernel[2]).max(1) as usize,
                },
                _ => Op::AvgPool2d {
                    kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                    kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    pad_h: pads.first().copied().unwrap_or(0).max(0) as usize,
                    pad_w: pads.get(1).copied().unwrap_or(0).max(0) as usize,
                    count_include_pad,
                },
            }
        }

        "GlobalAveragePool" => Op::GlobalAvgPool,
        "GlobalMaxPool" => Op::GlobalMaxPool,

        "LpPool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(kernel);
            let p = get_attr_int(node, "p", 2) as f64;
            match kernel.len() {
                1 => Op::LpPool1d {
                    kernel_l: kernel[0].max(1) as usize,
                    stride: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    p,
                },
                3 => Op::LpPool3d {
                    kernel_d: kernel[0].max(1) as usize,
                    kernel_h: kernel[1].max(1) as usize,
                    kernel_w: kernel[2].max(1) as usize,
                    stride_d: strides.first().copied().unwrap_or(kernel[0]).max(1) as usize,
                    stride_h: strides.get(1).copied().unwrap_or(kernel[1]).max(1) as usize,
                    stride_w: strides.get(2).copied().unwrap_or(kernel[2]).max(1) as usize,
                    p,
                },
                _ => Op::LpPool2d {
                    kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                    kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                    stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                    stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
                    p,
                },
            }
        }

        "MaxUnpool" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            let strides = get_attr_ints(node, "strides").unwrap_or(kernel);
            Op::MaxUnpool {
                kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
                stride_h: strides.first().copied().unwrap_or(1).max(1) as usize,
                stride_w: strides.get(1).copied().unwrap_or(1).max(1) as usize,
            }
        }

        // ---------------------------------------------------------------
        // Normalisation
        // ---------------------------------------------------------------
        "BatchNormalization" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            let momentum = get_attr_float(node, "momentum", 0.9) as f64;
            // Use BatchNorm2d as the representative variant (most common in practice).
            Op::BatchNorm2d {
                num_features: 0,
                eps,
                momentum,
                affine: true,
                track_running_stats: true,
            }
        }

        "LayerNormalization" | "LayerNorm" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::LayerNorm {
                normalized_shape: vec![],
                axis: get_attr_int(node, "axis", -1),
                eps,
                affine: true,
            }
        }

        "RMSNormalization" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::RmsNorm {
                normalized_shape: vec![],
                eps,
                affine: true,
            }
        }

        "GroupNormalization" => {
            let num_groups = get_attr_int(node, "num_groups", 1).max(1) as usize;
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::GroupNorm {
                num_groups,
                num_channels: 0,
                eps,
                affine: true,
            }
        }

        "InstanceNormalization" => {
            let eps = get_attr_float(node, "epsilon", 1e-5) as f64;
            Op::InstanceNorm2d {
                num_features: 0,
                eps,
                momentum: 0.1,
                affine: true,
                track_running_stats: false,
            }
        }

        "LRN" => {
            let alpha = get_attr_float(node, "alpha", 0.0001) as f64;
            let beta = get_attr_float(node, "beta", 0.75) as f64;
            let bias = get_attr_float(node, "bias", 1.0) as f64;
            let size = get_attr_int(node, "size", 1).max(1) as usize;
            Op::LRN {
                alpha,
                beta,
                bias,
                size,
            }
        }

        "MeanVarianceNormalization" => {
            let axes = get_attr_ints(node, "axes").unwrap_or(&[0, 2, 3]).to_vec();
            Op::MeanVarianceNormalization { axes }
        }

        "LpNormalization" => {
            let axis = get_attr_int(node, "axis", -1);
            let p = get_attr_int(node, "p", 2);
            Op::LpNormalization { axis, p }
        }

        // ---------------------------------------------------------------
        // Activations
        // ---------------------------------------------------------------
        "Elu" => Op::Elu {
            alpha: get_attr_float(node, "alpha", 1.0) as f64,
        },
        "Selu" => {
            // The kernel has PyTorch's constants baked in, which are ONNX's defaults.
            let alpha = get_attr_float(node, "alpha", 1.673_263_2);
            let gamma = get_attr_float(node, "gamma", 1.050_701);
            if (alpha - 1.673_263_2).abs() > 1e-5 || (gamma - 1.050_701).abs() > 1e-5 {
                return Err(Error::Unsupported(format!(
                    "Selu '{}' with alpha {alpha}, gamma {gamma}: only the defaults \
                     (1.6733, 1.0507) are supported",
                    node.name
                ))
                .into());
            }
            Op::Selu
        }
        "Celu" => Op::Celu {
            alpha: get_attr_float(node, "alpha", 1.0) as f64,
        },
        "Gelu" => match get_attr_string(node, "approximate", "none") {
            "tanh" => Op::Gelu,
            _ => Op::GeluErf,
        },
        "Mish" => Op::Mish,
        "HardSigmoid" => {
            // The kernel is PyTorch's hardsigmoid, clamp(x / 6 + 1 / 2, 0, 1): ONNX's
            // alpha = 1/6, beta = 0.5. ONNX's own defaults (0.2, 0.5) differ, so any
            // other pair is refused rather than computed with PyTorch's.
            let alpha = get_attr_float(node, "alpha", 0.2);
            let beta = get_attr_float(node, "beta", 0.5);
            if (alpha - 1.0 / 6.0).abs() > 1e-6 || (beta - 0.5).abs() > 1e-6 {
                return Err(Error::Unsupported(format!(
                    "HardSigmoid '{}' with alpha {alpha}, beta {beta}: only PyTorch's \
                     (1/6, 0.5) is supported",
                    node.name
                ))
                .into());
            }
            Op::Hardsigmoid
        }
        "HardSwish" => Op::Hardswish,
        "Swish" => Op::Swish,
        "LeakyRelu" => Op::LeakyRelu {
            negative_slope: get_attr_float(node, "alpha", 0.01) as f64,
        },
        "Softplus" => Op::Softplus {
            beta: 1.0,
            threshold: 20.0,
        },
        "Softsign" => Op::Softsign,
        "LogSoftmax" => Op::LogSoftmax {
            axis: get_attr_int(node, "axis", -1),
        },
        "Hardmax" => Op::Hardmax {
            axis: get_attr_int(node, "axis", -1),
        },
        "PRelu" => Op::PRelu,
        "ThresholdedRelu" => Op::ThresholdedRelu {
            alpha: get_attr_float(node, "alpha", 1.0) as f64,
        },
        "Shrink" => Op::Shrink {
            lambd: get_attr_float(node, "lambd", 0.5) as f64,
            bias: get_attr_float(node, "bias", 0.0) as f64,
        },
        // Opset 6 put the bounds in attributes; opset 11 moved them to optional
        // operands. Both forms are read: an export states one or the other, and
        // dropping the attribute form would leave those models silently unclamped.
        "Clip" => Op::Clip {
            min: get_attr_float_opt(node, "min").map(f64::from),
            max: get_attr_float_opt(node, "max").map(f64::from),
        },

        // ---------------------------------------------------------------
        // Element-wise unary
        // ---------------------------------------------------------------
        "Abs" => Op::Abs,
        "Neg" => Op::Neg,
        "Ceil" => Op::Ceil,
        "Floor" => Op::Floor,
        "Round" => Op::Round,
        "Sqrt" => Op::Sqrt,
        "Reciprocal" => Op::Reciprocal,
        "Exp" => Op::Exp,
        "Log" => Op::Log,
        "Erf" => Op::Erf,
        "Sign" => Op::Sign,
        "IsNaN" => Op::IsNaN,
        "IsInf" => Op::IsInf {
            detect_negative: get_attr_int(node, "detect_negative", 1) != 0,
            detect_positive: get_attr_int(node, "detect_positive", 1) != 0,
        },
        "Not" => Op::Not,
        "BitwiseNot" => Op::BitwiseNot,
        "Sin" => Op::Sin,
        "Cos" => Op::Cos,
        "Tan" => Op::Tan,
        "Asin" => Op::Asin,
        "Acos" => Op::Acos,
        "Atan" => Op::Atan,
        "Sinh" => Op::Sinh,
        "Cosh" => Op::Cosh,
        "Asinh" => Op::Asinh,
        "Acosh" => Op::Acosh,
        "Atanh" => Op::Atanh,

        // ---------------------------------------------------------------
        // Element-wise binary / variadic
        // ---------------------------------------------------------------
        "Mul" => Op::Mul,
        "Sub" => Op::Sub,
        "Div" => Op::Div,
        "Pow" => Op::Pow,
        "Mod" => Op::Mod {
            fmod: get_attr_int(node, "fmod", 0) != 0,
        },
        "Min" => Op::ElemMin,
        "Max" => Op::ElemMax,
        "Mean" => Op::ElemMean,
        "Sum" => Op::ElemSum,
        "Equal" => Op::Equal,
        "Greater" => Op::Greater,
        "GreaterOrEqual" => Op::GreaterOrEqual,
        "Less" => Op::Less,
        "LessOrEqual" => Op::LessOrEqual,
        "And" => Op::And,
        "Or" => Op::Or,
        "Xor" => Op::Xor,
        "BitwiseAnd" => Op::BitwiseAnd,
        "BitwiseOr" => Op::BitwiseOr,
        "BitwiseXor" => Op::BitwiseXor,
        "BitShift" => Op::BitShift {
            direction: get_attr_string(node, "direction", "LEFT").to_string(),
        },

        // ---------------------------------------------------------------
        // Tensor structural
        // ---------------------------------------------------------------
        "Reshape" => Op::Reshape,
        "Transpose" => Op::Transpose {
            perm: get_attr_ints(node, "perm")
                .map(|v| v.iter().map(|&i| i.max(0) as usize).collect())
                .unwrap_or_default(),
        },
        "Squeeze" => Op::Squeeze {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
        },
        "Unsqueeze" => Op::Unsqueeze {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
        },
        "Concat" => Op::Concat {
            axis: get_attr_int(node, "axis", 0),
        },
        "Split" => Op::Split {
            axis: get_attr_int(node, "axis", 0),
            num_outputs: node.output.len().max(1),
        },
        "Slice" => Op::Slice,
        "Gather" => Op::Gather {
            axis: get_attr_int(node, "axis", 0),
        },
        "GatherElements" => Op::GatherElements {
            axis: get_attr_int(node, "axis", 0),
        },
        "GatherND" => Op::GatherND {
            batch_dims: get_attr_int(node, "batch_dims", 0),
        },
        "ScatterElements" | "ScatterND" if get_attr_string(node, "reduction", "none") != "none" => {
            return Err(Error::Unsupported(format!(
                "{} '{}' with reduction '{}': only 'none' is supported",
                node.op_type,
                node.name,
                get_attr_string(node, "reduction", "none")
            ))
            .into());
        }
        "ScatterElements" => Op::ScatterElements {
            axis: get_attr_int(node, "axis", 0),
        },
        "ScatterND" => Op::ScatterND,
        "Scatter" => Op::Scatter {
            axis: get_attr_int(node, "axis", 0),
        },
        "TensorScatter" => Op::TensorScatter,
        "Tile" => Op::Tile,
        "Expand" => Op::Expand,
        "Shape" => Op::ShapeOf {
            start: get_attr_int(node, "start", 0),
            end: get_attr_int(node, "end", i64::MAX),
        },
        "Size" => Op::SizeOf,
        "Identity" => Op::Identity,
        "Cast" => {
            let to = get_attr_int(node, "to", 1);
            let dtype = tensor_dtype_to_dtype(to as i32)?;
            Op::Cast { to: dtype }
        }
        "CastLike" => Op::CastLike,
        "Where" => Op::Where,
        "Compress" => Op::Compress {
            axis: get_attr_int(node, "axis", 0),
        },
        "Range" => Op::Range,
        "Constant" => {
            // Shape/dtype extracted from the tensor attribute.
            if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                let tensor = &attr.t;
                let dtype = tensor_dtype_to_dtype(tensor.data_type)?;
                let shape = dims_to_shape(&tensor.dims);
                Op::Constant { dtype, shape }
            } else {
                Op::Constant {
                    dtype: DtypeRepr::F32,
                    shape: vec![],
                }
            }
        }
        "ConstantOfShape" => {
            // ONNX's default when the node has no `value`: a float zero.
            let Some(attr) = node.attribute.iter().find(|a| a.name == "value") else {
                return Ok(Op::ConstantOfShape {
                    dtype: DtypeRepr::F32,
                    fill: Vec::new(),
                });
            };
            let dtype = tensor_dtype_to_dtype(attr.t.data_type)?;
            // The attribute is a one-element tensor holding the fill value. Its bytes
            // go in the op so that lowering can fill the output on the device without
            // the value having to travel beside the graph; `Onnx::tensors` keeps the
            // same tensor for the constant folder, which uses it to evaluate the node
            // outright when the output is small enough to be worth holding.
            let value = tensor_data(&attr.t, &format!("ConstantOfShape '{}'", node.name))?;
            let width = dtype_width(dtype);
            if value.bytes.len() != width {
                return Err(Error::InvalidModel(format!(
                    "ConstantOfShape '{}': its value attribute carries {} bytes, but one \
                     {dtype:?} is {width}; the attribute holds the single fill value",
                    node.name,
                    value.bytes.len()
                ))
                .into());
            }
            Op::ConstantOfShape {
                dtype,
                fill: value.bytes,
            }
        }
        "Trilu" => Op::Trilu {
            upper: get_attr_int(node, "upper", 1) != 0,
        },
        "BitCast" => {
            let to = get_attr_int(node, "to", 1);
            let dtype = tensor_dtype_to_dtype(to as i32)?;
            Op::BitCast { to: dtype }
        }
        "Pad" => Op::Pad {
            mode: get_attr_string(node, "mode", "constant").to_string(),
        },
        "ReverseSequence" => Op::ReverseSequence {
            batch_axis: get_attr_int(node, "batch_axis", 1),
            time_axis: get_attr_int(node, "time_axis", 0),
        },
        "NonZero" => Op::NonZero,

        // ---------------------------------------------------------------
        // Matrix
        // ---------------------------------------------------------------
        "Gemm" => Op::Gemm {
            alpha: get_attr_float(node, "alpha", 1.0) as f64,
            beta: get_attr_float(node, "beta", 1.0) as f64,
            trans_a: get_attr_int(node, "transA", 0) != 0,
            trans_b: get_attr_int(node, "transB", 0) != 0,
        },
        "MatMul" => Op::MatMul,
        "MatMulInteger" => Op::MatMulInteger,
        "Einsum" => Op::Einsum {
            equation: get_attr_string(node, "equation", "").to_string(),
        },
        "Det" => Op::Det,
        "QLinearMatMul" => Op::QLinearMatMul,
        "ConvInteger" => Op::ConvInteger {
            groups: get_attr_int(node, "group", 1).max(1) as usize,
        },
        "DeformConv" => Op::DeformConv {
            group: get_attr_int(node, "group", 1).max(1) as usize,
            offset_group: get_attr_int(node, "offset_group", 1).max(1) as usize,
        },
        "QLinearConv" => Op::QLinearConv {
            groups: get_attr_int(node, "group", 1).max(1) as usize,
        },
        "Col2Im" => {
            let kernel = get_attr_ints(node, "kernel_shape").unwrap_or(&[1, 1]);
            Op::Col2Im {
                kernel_h: kernel.first().copied().unwrap_or(1).max(1) as usize,
                kernel_w: kernel.get(1).copied().unwrap_or(1).max(1) as usize,
            }
        }

        // ---------------------------------------------------------------
        // Reductions
        // ---------------------------------------------------------------
        "ReduceSum" => Op::ReduceSum {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceMean" => Op::ReduceMean {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceMax" => Op::ReduceMax {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceMin" => Op::ReduceMin {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceProd" => Op::ReduceProd {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceL1" => Op::ReduceL1 {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceL2" => Op::ReduceL2 {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceLogSum" => Op::ReduceLogSum {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceLogSumExp" => Op::ReduceLogSumExp {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "ReduceSumSquare" => Op::ReduceSumSquare {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            noop_with_empty_axes: get_attr_int(node, "noop_with_empty_axes", 0) != 0,
        },
        "CumSum" => Op::CumSum {
            // ONNX puts the axis in operand 1, which is usually a Constant node and so
            // not visible here; `materialize::adopt_axis_operands` writes it once the
            // fold has computed it. Zero until then, which is also ONNX's meaning for
            // a rank-1 input and the only safe guess.
            axis: 0,
            exclusive: get_attr_int(node, "exclusive", 0) != 0,
            reverse: get_attr_int(node, "reverse", 0) != 0,
        },
        "CumProd" => Op::CumProd {
            // As `CumSum` above.
            axis: 0,
            exclusive: get_attr_int(node, "exclusive", 0) != 0,
            reverse: get_attr_int(node, "reverse", 0) != 0,
        },
        "ArgMax" => Op::ArgMax {
            axis: get_attr_int(node, "axis", 0),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            select_last_index: get_attr_int(node, "select_last_index", 0) != 0,
        },
        "ArgMin" => Op::ArgMin {
            axis: get_attr_int(node, "axis", 0),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
            select_last_index: get_attr_int(node, "select_last_index", 0) != 0,
        },

        // ---------------------------------------------------------------
        // Resize / spatial
        // ---------------------------------------------------------------
        "Upsample" | "Resize"
            if get_attr_float(node, "cubic_coeff_a", -0.75) != -0.75
                || get_attr_int(node, "exclude_outside", 0) != 0
                || get_attr_float(node, "extrapolation_value", 0.0) != 0.0
                || get_attr_string(node, "keep_aspect_ratio_policy", "stretch") != "stretch" =>
        {
            return Err(Error::Unsupported(format!(
                "Node '{}' (Resize): cubic_coeff_a other than -0.75, exclude_outside, \
                 extrapolation_value and keep_aspect_ratio_policy are not supported",
                node.name
            ))
            .into());
        }
        "Upsample" | "Resize" => Op::Resize {
            mode: get_attr_string(node, "mode", "nearest").to_string(),
            // Resize defaults to half_pixel; the deprecated Upsample has no such
            // attribute and always means asymmetric.
            coordinate_transformation_mode: get_attr_string(
                node,
                "coordinate_transformation_mode",
                if node.op_type == "Upsample" {
                    "asymmetric"
                } else {
                    "half_pixel"
                },
            )
            .to_string(),
            // ONNX's default is a rounding, not a floor. An exporter writing a plain
            // nearest upsample asks for `floor` explicitly, and reading one for the
            // other shifts the whole axis by an element, so it is carried rather than
            // assumed.
            // Upsample (deprecated) has no nearest_mode and always floors.
            nearest_mode: if node.op_type == "Upsample" {
                "floor".to_string()
            } else {
                get_attr_string(node, "nearest_mode", "round_prefer_floor").to_string()
            },
            antialias: get_attr_int(node, "antialias", 0) != 0,
        },

        "GridSample" => Op::GridSample {
            mode: get_attr_string(node, "mode", "bilinear").to_string(),
            padding_mode: get_attr_string(node, "padding_mode", "zeros").to_string(),
            align_corners: get_attr_int(node, "align_corners", 0) != 0,
        },

        "SpaceToDepth" => Op::SpaceToDepth {
            blocksize: get_attr_int(node, "blocksize", 2).max(1) as usize,
        },
        "DepthToSpace" => Op::DepthToSpace {
            blocksize: get_attr_int(node, "blocksize", 2).max(1) as usize,
            mode: get_attr_string(node, "mode", "DCR").to_string(),
        },

        "RoiAlign" => Op::RoiAlign {
            output_h: get_attr_int(node, "output_height", 1).max(1) as usize,
            output_w: get_attr_int(node, "output_width", 1).max(1) as usize,
            sampling_ratio: get_attr_int(node, "sampling_ratio", 0),
            spatial_scale: get_attr_float(node, "spatial_scale", 1.0) as f64,
        },

        "AffineGrid" => Op::AffineGrid {
            align_corners: get_attr_int(node, "align_corners", 0) != 0,
        },

        "CenterCropPad" => Op::CenterCropPad {
            axes: get_attr_ints(node, "axes").unwrap_or_default().to_vec(),
        },

        "NonMaxSuppression" => Op::NonMaxSuppression {
            center_point_box: get_attr_int(node, "center_point_box", 0) != 0,
        },

        // ---------------------------------------------------------------
        // Recurrent
        // ---------------------------------------------------------------
        "LSTM" => {
            let hidden_size = get_attr_int(node, "hidden_size", 1).max(1) as usize;
            let direction = get_attr_string(node, "direction", "forward").to_string();
            let bidirectional = direction == "bidirectional";
            Op::Lstm {
                hidden_size,
                direction,
                bidirectional,
            }
        }
        "GRU" => {
            let hidden_size = get_attr_int(node, "hidden_size", 1).max(1) as usize;
            let direction = get_attr_string(node, "direction", "forward").to_string();
            let bidirectional = direction == "bidirectional";
            Op::Gru {
                hidden_size,
                direction,
                bidirectional,
            }
        }
        "RNN" => {
            let hidden_size = get_attr_int(node, "hidden_size", 1).max(1) as usize;
            let direction = get_attr_string(node, "direction", "forward").to_string();
            let bidirectional = direction == "bidirectional";
            Op::Rnn {
                hidden_size,
                direction,
                bidirectional,
            }
        }

        // ---------------------------------------------------------------
        // Attention
        // ---------------------------------------------------------------
        "Attention" => Op::MultiHeadAttention {
            q_num_heads: get_attr_int(node, "q_num_heads", 1).max(1) as usize,
            kv_num_heads: get_attr_int(node, "kv_num_heads", 1).max(1) as usize,
        },
        "RotaryEmbedding" => Op::RotaryEmbedding,
        // score_mod/prob_mod are GRAPH-typed attributes (a user-provided scoring subgraph) --
        // not captured, consistent with Loop/If/Scan not capturing their subgraph bodies.
        "FlexAttention" => Op::FlexAttention {
            scale: get_attr_float(node, "scale", 0.0) as f64,
        },
        "LinearAttention" => Op::LinearAttention {
            q_num_heads: get_attr_int(node, "q_num_heads", 1).max(1) as usize,
            kv_num_heads: get_attr_int(node, "kv_num_heads", 1).max(1) as usize,
            update_rule: get_attr_string(node, "update_rule", "").to_string(),
            scale: get_attr_float(node, "scale", 0.0) as f64,
        },
        "CausalConvWithState" => Op::CausalConvWithState {
            activation: get_attr_string(node, "activation", "").to_string(),
        },

        // ---------------------------------------------------------------
        // Misc
        // ---------------------------------------------------------------
        "TopK" => Op::TopK {
            axis: get_attr_int(node, "axis", -1),
            largest: get_attr_int(node, "largest", 1) != 0,
            sorted: get_attr_int(node, "sorted", 1) != 0,
        },
        "Unique" => Op::Unique {
            sorted: get_attr_int(node, "sorted", 1) != 0,
        },
        "Dropout" => Op::Dropout {
            training_mode: get_attr_int(node, "training_mode", 0) != 0,
        },
        "EyeLike" => {
            let to = get_attr_int(node, "dtype", -1);
            let dtype = if to >= 0 {
                Some(tensor_dtype_to_dtype(to as i32)?)
            } else {
                None
            };
            Op::EyeLike {
                dtype,
                k: get_attr_int(node, "k", 0),
            }
        }
        "OneHot" => Op::OneHot {
            axis: get_attr_int(node, "axis", -1),
        },
        "Bernoulli" => {
            let to = get_attr_int(node, "dtype", -1);
            let dtype = if to >= 0 {
                Some(tensor_dtype_to_dtype(to as i32)?)
            } else {
                None
            };
            Op::Bernoulli { dtype }
        }
        "RandomUniformLike" => {
            let to = get_attr_int(node, "dtype", -1);
            let dtype = if to >= 0 {
                Some(tensor_dtype_to_dtype(to as i32)?)
            } else {
                None
            };
            Op::RandomUniformLike {
                dtype,
                high: get_attr_float(node, "high", 1.0) as f64,
                low: get_attr_float(node, "low", 0.0) as f64,
            }
        }

        // ---------------------------------------------------------------
        // Quantisation
        // ---------------------------------------------------------------
        "QuantizeLinear" => Op::QuantizeLinear {
            axis: get_attr_int(node, "axis", 1),
            saturate: get_attr_int(node, "saturate", 1) != 0,
        },
        "DequantizeLinear" => Op::DequantizeLinear {
            axis: get_attr_int(node, "axis", 1),
        },
        "DynamicQuantizeLinear" => Op::DynamicQuantizeLinear,

        // ---------------------------------------------------------------
        // Signal
        // ---------------------------------------------------------------
        "DFT" => Op::Dft {
            inverse: get_attr_int(node, "inverse", 0) != 0,
            onesided: get_attr_int(node, "onesided", 0) != 0,
        },
        "STFT" => Op::Stft,
        "MelWeightMatrix" => Op::MelWeightMatrix,
        "HannWindow" => Op::HannWindow {
            periodic: get_attr_int(node, "periodic", 1) != 0,
        },
        "BlackmanWindow" => Op::BlackmanWindow {
            periodic: get_attr_int(node, "periodic", 1) != 0,
        },
        "HammingWindow" => Op::HammingWindow {
            periodic: get_attr_int(node, "periodic", 1) != 0,
        },

        // ---------------------------------------------------------------
        // Loss
        // ---------------------------------------------------------------
        "NegativeLogLikelihoodLoss" => Op::NegativeLogLikelihoodLoss {
            reduction: get_attr_string(node, "reduction", "mean").to_string(),
        },
        "SoftmaxCrossEntropyLoss" => Op::SoftmaxCrossEntropyLoss {
            reduction: get_attr_string(node, "reduction", "mean").to_string(),
        },

        // ---------------------------------------------------------------
        // Sequences / optionals
        // ---------------------------------------------------------------
        "SequenceAt" => Op::SequenceAt,
        "SequenceConstruct" => Op::SequenceConstruct,
        "SequenceEmpty" => Op::SequenceEmpty,
        "SequenceErase" => Op::SequenceErase,
        "SequenceInsert" => Op::SequenceInsert,
        "SequenceLength" => Op::SequenceLength,
        "SequenceMap" => Op::SequenceMap,
        "SplitToSequence" => Op::SplitToSequence {
            axis: get_attr_int(node, "axis", 0),
            keepdims: get_attr_int(node, "keepdims", 1) != 0,
        },
        "ConcatFromSequence" => Op::ConcatFromSequence {
            axis: get_attr_int(node, "axis", 0),
            new_axis: get_attr_int(node, "new_axis", 0) != 0,
        },
        "OptionalGetElement" => Op::OptionalGetElement,
        "OptionalHasElement" => Op::OptionalHasElement,

        // ---------------------------------------------------------------
        // Control flow
        // ---------------------------------------------------------------
        "Loop" => Op::Loop,
        "Scan" => Op::Scan {
            num_scan_inputs: get_attr_int(node, "num_scan_inputs", 0),
        },
        "If" => Op::If,

        // ---------------------------------------------------------------
        // Optimiser ops
        // ---------------------------------------------------------------
        "Adagrad" => Op::Adagrad,
        "Adam" => Op::Adam,
        "Momentum" => Op::Momentum,
        "Gradient" => Op::Gradient,

        // ---------------------------------------------------------------
        // String / NLP
        // ---------------------------------------------------------------
        "StringNormalizer" => Op::StringNormalizer,
        "RegexFullMatch" => Op::RegexFullMatch {
            pattern: get_attr_string(node, "pattern", "").to_string(),
        },
        "StringConcat" => Op::StringConcat,
        "StringSplit" => Op::StringSplit,
        "TfIdfVectorizer" => Op::TfIdfVectorizer,
        "LabelEncoder" => Op::LabelEncoder,

        // ---------------------------------------------------------------
        // Other ML / deprecated
        // ---------------------------------------------------------------
        "ArrayFeatureExtractor" => Op::ArrayFeatureExtractor,
        "Binarizer" => Op::Binarizer {
            threshold: get_attr_float(node, "threshold", 0.0) as f64,
        },
        "TreeEnsemble" => Op::TreeEnsemble,
        "ImageDecoder" => Op::ImageDecoder,

        other => {
            return Err(Error::Unsupported(format!(
                "Unsupported ONNX op '{}' in node '{}'",
                other, node.name
            ))
            .into());
        }
    };
    Ok(op)
}

#[cfg(test)]
mod tensor_data_tests {
    use super::*;

    fn proto(data_type: i32, dims: &[i64]) -> TensorProto {
        let mut t = TensorProto::new();
        t.data_type = data_type;
        t.dims = dims.to_vec();
        t
    }

    /// `int32_data` carrying bools must land one byte per element.
    ///
    /// Taken naively, four bytes per element puts element 0 in the right place and
    /// every later one in the exporter's zero padding, so a mask of alternating true
    /// and false decodes as all-false past the first element. Alternating values are
    /// what make that failure visible.
    #[test]
    fn test_bools_in_the_int32_carrier_field_narrow_to_one_byte() {
        let mut t = proto(tensor_proto::DataType::BOOL as i32, &[4]);
        t.int32_data = vec![1, 0, 1, 1];

        let td = tensor_data(&t, "mask").expect("a bool tensor should decode");
        assert_eq!(td.dtype, DtypeRepr::Bool);
        assert_eq!(td.bytes, vec![1u8, 0, 1, 1]);
        assert_eq!(td.to_i64_vec(), Ok(vec![1i64, 0, 1, 1]));
    }

    /// The same field carries the small integer types, which narrow the same way.
    #[test]
    fn test_small_ints_in_the_int32_carrier_field_narrow_to_their_own_width() {
        let mut t = proto(tensor_proto::DataType::INT16 as i32, &[3]);
        t.int32_data = vec![1, -2, 300];
        let td = tensor_data(&t, "i16").expect("an int16 tensor should decode");
        assert_eq!(
            td.bytes.len(),
            6,
            "three int16s are six bytes: {:?}",
            td.bytes
        );
        assert_eq!(td.to_i64_vec(), Ok(vec![1i64, -2, 300]));

        let mut t = proto(tensor_proto::DataType::INT8 as i32, &[3]);
        t.int32_data = vec![7, -1, 0];
        let td = tensor_data(&t, "i8").expect("an int8 tensor should decode");
        assert_eq!(td.to_i64_vec(), Ok(vec![7i64, -1, 0]));
    }

    /// `uint64_data` carries uint32 as well, which narrows to four bytes.
    #[test]
    fn test_uint32_in_the_uint64_carrier_field_narrows_to_four_bytes() {
        let mut t = proto(tensor_proto::DataType::UINT32 as i32, &[2]);
        t.uint64_data = vec![5, 4_000_000_000];
        let td = tensor_data(&t, "u32").expect("a uint32 tensor should decode");
        assert_eq!(td.bytes.len(), 8);
        assert_eq!(td.to_i64_vec(), Ok(vec![5i64, 4_000_000_000]));
    }

    /// A dtype stored at its natural width in its own field is unaffected.
    #[test]
    fn test_int64_and_int32_keep_their_full_width() {
        let mut t = proto(tensor_proto::DataType::INT64 as i32, &[2]);
        t.int64_data = vec![-1, 1 << 40];
        let td = tensor_data(&t, "i64").expect("an int64 tensor should decode");
        assert_eq!(td.to_i64_vec(), Ok(vec![-1i64, 1 << 40]));

        let mut t = proto(tensor_proto::DataType::INT32 as i32, &[2]);
        t.int32_data = vec![-1, 70_000];
        let td = tensor_data(&t, "i32").expect("an int32 tensor should decode");
        assert_eq!(td.bytes.len(), 8);
        assert_eq!(td.to_i64_vec(), Ok(vec![-1i64, 70_000]));
    }

    /// `raw_data` is already at the dtype's width, so it is passed through untouched
    /// even for the dtypes the carrier fields narrow.
    #[test]
    fn test_raw_data_is_passed_through_for_bools() {
        let mut t = proto(tensor_proto::DataType::BOOL as i32, &[3]);
        t.raw_data = vec![1, 0, 1];
        let td = tensor_data(&t, "mask").expect("a bool tensor should decode");
        assert_eq!(td.bytes, vec![1u8, 0, 1]);
    }
}

#[cfg(test)]
mod omitted_operand_tests {
    use super::*;
    use protobuf::Message;

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        let mut v = ValueInfoProto::new();
        v.name = name.to_string();
        let mut ty = onnx_proto3::TypeProto::new();
        let tt = ty.mut_tensor_type();
        tt.elem_type = tensor_proto::DataType::FLOAT as i32;
        let mut shape = onnx_proto3::TensorShapeProto::new();
        for d in dims {
            let mut dim = onnx_proto3::tensor_shape_proto::Dimension::new();
            dim.set_dim_value(*d);
            shape.dim.push(dim);
        }
        tt.shape = Some(shape).into();
        v.type_ = Some(ty).into();
        v
    }

    fn i64_initializer(name: &str, vals: &[i64]) -> TensorProto {
        let mut t = TensorProto::new();
        t.name = name.to_string();
        t.data_type = tensor_proto::DataType::INT64 as i32;
        t.dims = vec![vals.len() as i64];
        t.int64_data = vals.to_vec();
        t
    }

    /// `Resize(X, "", "", sizes)`, as an exporter writes a resize to a fixed size:
    /// `roi` and `scales` are omitted, and `sizes` is the fourth operand.
    fn resize_model() -> Vec<u8> {
        let mut g = onnx_proto3::GraphProto::new();
        g.input.push(value_info("x", &[1, 3, 8, 8]));
        g.output.push(value_info("y", &[]));
        g.initializer
            .push(i64_initializer("sizes", &[1, 3, 16, 16]));

        let mut n = NodeProto::new();
        n.op_type = "Resize".to_string();
        n.name = "/Resize".to_string();
        n.input = ["x", "", "", "sizes"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        n.output = vec!["y".to_string()];
        g.node.push(n);

        let mut m = ModelProto::new();
        m.graph = Some(g).into();
        let mut op = onnx_proto3::OperatorSetIdProto::new();
        op.version = 17;
        m.opset_import.push(op);
        m.write_to_bytes().expect("serialise")
    }

    /// An omitted operand keeps its position, because position is the only thing that
    /// says what an operand means.
    ///
    /// Skipping it shifts every later operand one place left. For this model that puts
    /// `sizes` where `roi` belongs, so nothing can work out the output shape; for
    /// `Clip(x, "", max)` it would put the upper bound where the lower one belongs.
    #[test]
    fn test_an_omitted_operand_keeps_its_position() {
        let bytes = resize_model();
        let import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");

        let resize = import
            .graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Resize { .. }))
            .expect("the Resize should be imported");
        assert_eq!(
            resize.inputs.len(),
            4,
            "all four operand positions should be present"
        );
        let op_of = |i: usize| &import.graph.nodes[resize.inputs[i]].op;
        assert!(matches!(op_of(0), Op::Input), "operand 0 is the tensor");
        assert!(matches!(op_of(1), Op::Omitted), "roi is omitted");
        assert!(matches!(op_of(2), Op::Omitted), "scales is omitted");
        assert!(
            matches!(op_of(3), Op::Input),
            "sizes should still be at position 3, got {:?}",
            op_of(3)
        );
    }

    /// One node serves every omitted operand, so a model with hundreds does not grow
    /// hundreds of nodes.
    #[test]
    fn test_omitted_operands_share_one_node() {
        let bytes = resize_model();
        let import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");
        let omitted: Vec<usize> = import
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.op, Op::Omitted))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(omitted.len(), 1, "one shared node: {omitted:?}");
    }

    /// With the positions right, the shape follows from the operand the model actually
    /// supplied. This is the end the whole mechanism exists for.
    #[test]
    fn test_a_resize_with_omitted_operands_gets_its_shape() {
        let bytes = resize_model();
        let mut import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");
        let tensors = import.tensors.clone();
        let (_, report) = crate::materialize::materialize_constants(
            &mut import.graph,
            &tensors,
            &crate::materialize::MaterializeConfig::default(),
        );
        assert_eq!(report.shapes_conflicting, 0, "{report:?}");

        let resize = import
            .graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Resize { .. }))
            .expect("the Resize should be imported");
        assert_eq!(
            resize.shape,
            vec![Some(1), Some(3), Some(16), Some(16)],
            "the sizes operand should give the output shape"
        );
    }
}

#[cfg(test)]
mod split_expansion_tests {
    use super::*;
    use crate::onnx::onnx_proto3::AttributeProto;
    use crate::onnx::onnx_proto3::attribute_proto;
    use protobuf::Message;

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        let mut v = ValueInfoProto::new();
        v.name = name.to_string();
        let mut ty = onnx_proto3::TypeProto::new();
        let tt = ty.mut_tensor_type();
        tt.elem_type = tensor_proto::DataType::FLOAT as i32;
        let mut shape = onnx_proto3::TensorShapeProto::new();
        for d in dims {
            let mut dim = onnx_proto3::tensor_shape_proto::Dimension::new();
            dim.set_dim_value(*d);
            shape.dim.push(dim);
        }
        tt.shape = Some(shape).into();
        v.type_ = Some(ty).into();
        v
    }

    fn i64_initializer(name: &str, vals: &[i64]) -> TensorProto {
        let mut t = TensorProto::new();
        t.name = name.to_string();
        t.data_type = tensor_proto::DataType::INT64 as i32;
        t.dims = vec![vals.len() as i64];
        t.int64_data = vals.to_vec();
        t
    }

    /// `Split(x, split)` with two uneven outputs, both of them used, which is the
    /// shape of the export's own splits.
    fn split_model(axis: i64, sizes: &[i64]) -> Vec<u8> {
        let mut g = onnx_proto3::GraphProto::new();
        g.input.push(value_info("x", &[2, 5]));
        g.initializer.push(i64_initializer("split", sizes));

        let mut n = NodeProto::new();
        n.op_type = "Split".to_string();
        n.name = "/Split".to_string();
        n.input = ["x", "split"].iter().map(|s| s.to_string()).collect();
        n.output = (0..sizes.len()).map(|i| format!("out{i}")).collect();
        let mut a = AttributeProto::new();
        a.name = "axis".to_string();
        a.i = axis;
        a.type_ = protobuf::EnumOrUnknown::new(attribute_proto::AttributeType::INT);
        n.attribute.push(a);
        g.node.push(n);

        // Both outputs are read, by an op that keeps its operand's shape.
        for (i, name) in n_outputs(sizes.len()).iter().enumerate() {
            let mut r = NodeProto::new();
            r.op_type = "Relu".to_string();
            r.name = format!("/Relu{i}");
            r.input = vec![name.clone()];
            r.output = vec![format!("y{i}")];
            g.node.push(r);
        }

        let mut m = ModelProto::new();
        m.graph = Some(g).into();
        let mut op = onnx_proto3::OperatorSetIdProto::new();
        op.version = 17;
        m.opset_import.push(op);
        m.write_to_bytes().expect("serialise")
    }

    fn n_outputs(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("out{i}")).collect()
    }

    /// A multi-output op cannot be one node: every output name would resolve to the
    /// same node, and so to the same buffer, and a consumer of output 1 would read
    /// output 0 with nothing to show that it had.
    #[test]
    fn test_each_split_output_becomes_its_own_node() {
        let bytes = split_model(1, &[2, 3]);
        let import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");

        assert!(
            !import
                .graph
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::Split { .. })),
            "the Split should have been rewritten"
        );
        let a = import.value_to_node["out0"];
        let b = import.value_to_node["out1"];
        assert_ne!(a, b, "the two outputs share a node");
        assert!(matches!(import.graph.nodes[a].op, Op::Slice));
        assert!(matches!(import.graph.nodes[b].op, Op::Slice));
        // Both slices read the same tensor.
        assert_eq!(
            import.graph.nodes[a].inputs[0],
            import.graph.nodes[b].inputs[0]
        );
    }

    /// The sizes become the slices' own bounds, and shape inference works out what
    /// each one's extent is. Uneven sizes are the case that tells a real split from a
    /// guess at equal shares.
    #[test]
    fn test_the_slices_get_the_extents_the_sizes_asked_for() {
        let bytes = split_model(1, &[2, 3]);
        let mut import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");
        let a = import.value_to_node["out0"];
        let b = import.value_to_node["out1"];
        crate::materialize::materialize_constants(
            &mut import.graph,
            &import.tensors,
            &crate::materialize::MaterializeConfig::default(),
        );
        assert_eq!(import.graph.nodes[a].shape, vec![Some(2), Some(2)]);
        assert_eq!(import.graph.nodes[b].shape, vec![Some(2), Some(3)]);
    }

    /// A negative axis is recorded as given and resolved later, which is what keeps
    /// the rewrite from needing a rank it does not have during import.
    #[test]
    fn test_a_negative_axis_resolves_later() {
        let bytes = split_model(-1, &[1, 4]);
        let mut import = Onnx::import_from_reader(&mut &bytes[..]).expect("import");
        let a = import.value_to_node["out0"];
        let b = import.value_to_node["out1"];
        crate::materialize::materialize_constants(
            &mut import.graph,
            &import.tensors,
            &crate::materialize::MaterializeConfig::default(),
        );
        assert_eq!(import.graph.nodes[a].shape, vec![Some(2), Some(1)]);
        assert_eq!(import.graph.nodes[b].shape, vec![Some(2), Some(4)]);
    }
}
