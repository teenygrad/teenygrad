/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use egg::{Analysis, DidMerge, EGraph, Id, define_language};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::error::Error;

// Define the core language for your compiler IR
define_language! {
    pub enum ComputeIR {
        // Tensors and basic operations
        "tensor" = Tensor([Id; 2]), // [shape_id, dtype_id]
        "load" = Load(Id),          // Load from memory
        "store" = Store([Id; 2]),   // [tensor_id, address_id]

        // Arithmetic operations
        "add" = Add([Id; 2]),
        "mul" = Mul([Id; 2]),
        "sub" = Sub([Id; 2]),
        "div" = Div([Id; 2]),
        "neg" = Neg(Id),
        "abs" = Abs(Id),
        "exp" = Exp(Id),
        "log" = Log(Id),
        "sqrt" = Sqrt(Id),
        "sin" = Sin(Id),
        "cos" = Cos(Id),
        "tanh" = Tanh(Id),

        // Matrix operations
        "matmul" = MatMul([Id; 2]),
        "transpose" = Transpose(Id),
        "reshape" = Reshape([Id; 2]), // [tensor_id, new_shape_id]

        // Reduction operations
        "sum" = Sum([Id; 2]),    // [tensor_id, axis_id]
        "max" = Max([Id; 2]),
        "min" = Min([Id; 2]),
        "mean" = Mean([Id; 2]),

        // Indexing and slicing
        "index" = Index([Id; 2]),   // [tensor_id, indices_id]
        "slice" = Slice([Id; 3]),   // [tensor_id, start_id, end_id]
        "concat" = Concat([Id; 2]), // [tensors_id, axis_id]

        // Control flow
        "select" = Select([Id; 3]), // [condition_id, true_id, false_id]
        "loop" = Loop([Id; 3]),     // [init_id, condition_id, body_id]

        // Memory layout optimizations
        "contiguous" = Contiguous(Id),
        "view" = View([Id; 2]), // [tensor_id, shape_id]

        // Constants and literals
        Constant(ConstantValue),
        Shape(ShapeValue),
        Dtype(DtypeValue),
        Symbol(egg::Symbol),
    }
}

// Value types for constants
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstantValue {
    Int(i64),
    // Float(OrderedFloat),
    Bool(bool),
    IntList(Vec<i64>),
    // FloatList(Vec<OrderedFloat>),
}

impl FromStr for ConstantValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct OrderedFloat(pub f64);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShapeValue {
    Static(Vec<i64>),
    Dynamic(Vec<Option<i64>>), // None represents dynamic dimensions
    Symbolic(Vec<String>),     // For symbolic shapes
}

impl FromStr for ShapeValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DtypeValue {
    F32,
    F64,
    I32,
    I64,
    Bool,
    Complex64,
    Complex128,
}

impl FromStr for DtypeValue {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for ConstantValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantValue::Int(i) => write!(f, "{i}"),
            // ConstantValue::Float(OrderedFloat(fl)) => write!(f, "{}", fl),
            ConstantValue::Bool(b) => write!(f, "{b}"),
            ConstantValue::IntList(list) => write!(
                f,
                "[{}]",
                list.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            // ConstantValue::FloatList(list) => write!(
            //     f,
            //     "[{}]",
            //     list.iter()
            //         .map(|x| x.0.to_string())
            //         .collect::<Vec<_>>()
            //         .join(", ")
            // ),
        }
    }
}

impl Display for ShapeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeValue::Static(dims) => write!(
                f,
                "[{}]",
                dims.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ShapeValue::Dynamic(dims) => write!(
                f,
                "[{}]",
                dims.iter()
                    .map(|x| match x {
                        Some(d) => d.to_string(),
                        None => "?".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ShapeValue::Symbolic(syms) => write!(f, "[{}]", syms.join(", ")),
        }
    }
}

impl Display for DtypeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DtypeValue::F32 => write!(f, "f32"),
            DtypeValue::F64 => write!(f, "f64"),
            DtypeValue::I32 => write!(f, "i32"),
            DtypeValue::I64 => write!(f, "i64"),
            DtypeValue::Bool => write!(f, "bool"),
            DtypeValue::Complex64 => write!(f, "c64"),
            DtypeValue::Complex128 => write!(f, "c128"),
        }
    }
}

// Analysis for tracking tensor properties and optimization opportunities
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorAnalysis {
    pub shape: Option<ShapeValue>,
    pub dtype: Option<DtypeValue>,
    pub is_contiguous: bool,
    pub memory_format: MemoryFormat,
    pub compute_cost: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryFormat {
    ChannelsLast,
    ChannelsFirst,
    Contiguous,
    Unknown,
}

impl Default for TensorAnalysis {
    fn default() -> Self {
        Self {
            shape: None,
            dtype: None,
            is_contiguous: false,
            memory_format: MemoryFormat::Unknown,
            compute_cost: 0,
        }
    }
}

impl Analysis<ComputeIR> for TensorAnalysis {
    type Data = TensorAnalysis;

    fn make(egraph: &mut EGraph<ComputeIR, Self>, enode: &ComputeIR) -> Self::Data {
        use ComputeIR::*;

        match enode {
            Tensor([shape_id, dtype_id]) => {
                let shape = egraph[*shape_id].data.shape.clone();
                let dtype = egraph[*dtype_id].data.dtype.clone();
                TensorAnalysis {
                    shape,
                    dtype,
                    is_contiguous: true,
                    memory_format: MemoryFormat::Contiguous,
                    compute_cost: 1,
                }
            }

            Add([a, b]) | Mul([a, b]) | Sub([a, b]) | Div([a, b]) => {
                let a_data = &egraph[*a].data;
                let b_data = &egraph[*b].data;

                // Infer result shape and type
                let shape = match (&a_data.shape, &b_data.shape) {
                    (Some(s1), Some(s2)) => broadcast_shapes(s1, s2),
                    (Some(s), None) | (None, Some(s)) => Some(s.clone()),
                    _ => None,
                };

                TensorAnalysis {
                    shape,
                    dtype: a_data.dtype.clone().or_else(|| b_data.dtype.clone()),
                    is_contiguous: a_data.is_contiguous && b_data.is_contiguous,
                    memory_format: MemoryFormat::Unknown,
                    compute_cost: a_data.compute_cost + b_data.compute_cost + 10,
                }
            }

            MatMul([a, b]) => {
                let a_data = &egraph[*a].data;
                let b_data = &egraph[*b].data;

                let shape = match (&a_data.shape, &b_data.shape) {
                    (Some(ShapeValue::Static(s1)), Some(ShapeValue::Static(s2))) => {
                        matmul_output_shape(s1, s2)
                    }
                    _ => None,
                };

                TensorAnalysis {
                    shape,
                    dtype: a_data.dtype.clone().or_else(|| b_data.dtype.clone()),
                    is_contiguous: false, // MatMul often produces non-contiguous results
                    memory_format: MemoryFormat::Unknown,
                    compute_cost: a_data.compute_cost + b_data.compute_cost + 100, // High cost
                }
            }

            Contiguous(id) => {
                let mut data = egraph[*id].data.clone();
                data.is_contiguous = true;
                data.memory_format = MemoryFormat::Contiguous;
                data.compute_cost += 5; // Cost of making contiguous
                data
            }

            Shape(shape_val) => TensorAnalysis {
                shape: Some(shape_val.clone()),
                dtype: None,
                is_contiguous: true,
                memory_format: MemoryFormat::Contiguous,
                compute_cost: 0,
            },

            Dtype(dtype_val) => TensorAnalysis {
                shape: None,
                dtype: Some(dtype_val.clone()),
                is_contiguous: true,
                memory_format: MemoryFormat::Contiguous,
                compute_cost: 0,
            },

            Constant(_) => TensorAnalysis {
                shape: Some(ShapeValue::Static(vec![])), // Scalar
                dtype: None,                             // Would need to infer from constant type
                is_contiguous: true,
                memory_format: MemoryFormat::Contiguous,
                compute_cost: 0,
            },

            _ => TensorAnalysis::default(),
        }
    }

    fn merge(&mut self, _other: &mut Self::Data, other: Self::Data) -> DidMerge {
        let mut changed = false;

        // Merge shapes if compatible
        if self.shape.is_none() && other.shape.is_some() {
            self.shape = other.shape;
            changed = true;
        }

        // Merge dtypes if compatible
        if self.dtype.is_none() && other.dtype.is_some() {
            self.dtype = other.dtype;
            changed = true;
        }

        // Take minimum compute cost
        if other.compute_cost < self.compute_cost {
            self.compute_cost = other.compute_cost;
            changed = true;
        }

        // Update contiguity
        if !self.is_contiguous && other.is_contiguous {
            self.is_contiguous = true;
            changed = true;
        }

        DidMerge(changed, changed)
    }
}

// Helper functions for shape inference
fn broadcast_shapes(s1: &ShapeValue, s2: &ShapeValue) -> Option<ShapeValue> {
    match (s1, s2) {
        (ShapeValue::Static(dims1), ShapeValue::Static(dims2)) => {
            // Implement broadcasting logic
            let mut result = Vec::new();
            let max_len = dims1.len().max(dims2.len());

            for i in 0..max_len {
                let d1 = dims1
                    .get(dims1.len().saturating_sub(max_len - i))
                    .unwrap_or(&1);
                let d2 = dims2
                    .get(dims2.len().saturating_sub(max_len - i))
                    .unwrap_or(&1);

                match (*d1, *d2) {
                    (1, _) => result.push(*d2),
                    (_, 1) => result.push(*d1),
                    (_, _) if d1 == d2 => result.push(*d1),
                    _ => return None, // Incompatible shapes
                }
            }
            Some(ShapeValue::Static(result))
        }
        _ => None, // Can't broadcast dynamic/symbolic shapes easily
    }
}

fn matmul_output_shape(s1: &[i64], s2: &[i64]) -> Option<ShapeValue> {
    if s1.len() < 2 || s2.len() < 2 {
        return None;
    }

    let m = s1[s1.len() - 2];
    let k1 = s1[s1.len() - 1];
    let k2 = s2[s2.len() - 2];
    let n = s2[s2.len() - 1];

    if k1 != k2 {
        return None; // Incompatible dimensions
    }

    let mut result = Vec::new();

    // Handle batch dimensions
    let batch_dims1 = &s1[..s1.len() - 2];
    let _batch_dims2 = &s2[..s2.len() - 2];

    // For simplicity, assume batch dims are broadcastable
    // In practice, you'd need full broadcasting logic here
    result.extend_from_slice(batch_dims1);

    // Add matrix dimensions
    result.push(m);
    result.push(n);

    Some(ShapeValue::Static(result))
}

// Higher-level IR for mapping from FX graphs
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    pub egraph: EGraph<ComputeIR, TensorAnalysis>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
    pub node_map: HashMap<String, Id>, // For mapping from FX node names
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            egraph: EGraph::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_tensor(&mut self, name: String, shape: ShapeValue, dtype: DtypeValue) -> Id {
        let shape_id = self.egraph.add(ComputeIR::Shape(shape));
        let dtype_id = self.egraph.add(ComputeIR::Dtype(dtype));
        let tensor_id = self.egraph.add(ComputeIR::Tensor([shape_id, dtype_id]));
        self.node_map.insert(name, tensor_id);
        tensor_id
    }

    pub fn add_operation(&mut self, name: String, op: ComputeIR) -> Id {
        let id = self.egraph.add(op);
        self.node_map.insert(name, id);
        id
    }

    pub fn get_node(&self, name: &str) -> Option<Id> {
        self.node_map.get(name).copied()
    }
}
