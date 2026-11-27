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

//! Rotary position embedding as Hugging Face's `apply_rotary_pos_emb` applies it
//! (`rotate_half`, non-interleaved): for each head of `D` lanes split in halves
//! `x1 = x[:D/2]`, `x2 = x[D/2:]`,
//!
//! ```text
//! y[:D/2] = x1 * cos - x2 * sin
//! y[D/2:] = x2 * cos + x1 * sin
//! ```
//!
//! `cos` and `sin` are inputs `[T, D]` (both halves equal, as Hugging Face builds
//! them), computed on the host for the step's positions, so one op serves the local
//! (`theta = 10^4`) and global (`theta = 10^6`) layers alike.
//!
//! **Layout**: `x` `[T, heads * D]`. Grid: `T * heads`; `BLOCK` (power of two) at
//! least `D / 2`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The rotation above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn rope_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    cos_ptr: InPtr<T::Pointer<D>>,
    sin_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    HEADS: i32,
    HD: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let t = pid / HEADS;
    let half = HD / 2;
    let offs = T::arange(0, BLOCK);
    let m = offs.lt(half);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let xb = pid * HD;
    let cb = t * HD;
    let x1 = T::load(
        x_ptr.add_offsets(offs + xb),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let x2 = T::load(
        x_ptr.add_offsets(offs + (xb + half)),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let c = T::load(
        cos_ptr.add_offsets(offs + cb),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let s = T::load(
        sin_ptr.add_offsets(offs + cb),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(offs + xb),
        x1 * c - x2 * s,
        Some(m),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (xb + half)),
        x2 * c + x1 * s,
        Some(m),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level op
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// Graph-level RoPE: inputs `x` `[T, heads * head_dim]`, `cos`, `sin` `[T, head_dim]`.
/// Shape-preserving, no weights.
pub struct RopeOp<D: FloatBytes + Send + Sync + 'static> {
    /// Heads in `x`.
    pub heads: usize,
    /// Lanes per head.
    pub head_dim: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RopeOp<D> {
    /// Creates the op.
    pub fn new(heads: usize, head_dim: usize) -> Self {
        Self {
            heads,
            head_dim,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for RopeOp<D> {
    fn name(&self) -> &str {
        "llm.rope"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let block = (self.head_dim / 2).next_power_of_two() as i32;
        let kernel = RopeForward::<D>::new(block);
        Some((
            "rope_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(RopeRuntimeOp::<D> {
                heads: self.heads,
                head_dim: self.head_dim,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`RopeOp`].
pub struct RopeRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    heads: usize,
    head_dim: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for RopeRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        3
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(inputs[2].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.head_dim as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.heads) as u32, 1, 1]
    }
}

// ---------------------------------------------------------------------------
// Host-side tables
// ---------------------------------------------------------------------------

/// RoPE `cos` and `sin` tables `[positions.len(), head_dim]` for base `theta`, as
/// Hugging Face's default rotary embedding builds them (both halves equal).
pub fn rope_tables(theta: f64, head_dim: usize, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * head_dim);
    let mut sin = Vec::with_capacity(positions.len() * head_dim);
    for &p in positions {
        let row: Vec<f32> = (0..half)
            .map(|i| {
                // As Hugging Face: inv_freq in f32, then pos * inv_freq in f32.
                let inv = (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32;
                p as f32 * inv
            })
            .collect();
        for _ in 0..2 {
            cos.extend(row.iter().map(|a| a.cos()));
            sin.extend(row.iter().map(|a| a.sin()));
        }
    }
    (cos, sin)
}
