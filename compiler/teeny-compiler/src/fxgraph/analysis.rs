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

use egg::{Analysis, DidMerge, EGraph};

use crate::fxgraph::dtype::DtypeValue;
use crate::fxgraph::lang::FxGraphLang::{
    self, Add, Constant, Contiguous, Div, Dtype, MatMul, Mul, Placeholder, Shape, Sub, Tensor,
};
use crate::fxgraph::shape::ShapeValue;

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

impl Analysis<FxGraphLang> for TensorAnalysis {
    type Data = TensorAnalysis;

    fn make(egraph: &mut EGraph<FxGraphLang, Self>, enode: &FxGraphLang) -> Self::Data {
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
