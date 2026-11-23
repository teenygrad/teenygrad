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

//! The gather at the end of Inkling-2's relative position bias
//! (`Inkling2LoopRelativePositionBias`).
//!
//! The bias is conditioned on the hidden state, not on position alone. The full
//! path is three steps, and only the third needs a kernel of its own:
//!
//! 1. `relative_states = linear(hidden)`, reshaped to `[T, heads, d_rel]` -
//!    [`crate::kernels::linear::Linear4Op`];
//! 2. the scaled L2 clip (and, on log-scaling layers, the log scale) -
//!    [`super::l2_clip::ScaledL2ClipOp`];
//! 3. `rel_logits = relative_states @ proj` with `proj` of `[d_rel, rel_extent]`,
//!    again `Linear4Op` over `[T * heads, d_rel]` once `proj` is transposed at load
//!    time, and then **this kernel**, which turns the per-distance profile into a
//!    per-key bias:
//!
//! ```text
//! dist = (past + t) - j
//! bias[t, h, j] = rel_logits[t, h, dist]   if 0 <= dist < rel_extent
//!               = 0                        otherwise
//! ```
//!
//! So each query reads its own trained profile backwards over the keys, and the bias
//! falls to zero beyond `rel_extent` (1024 in Inkling-2-Mini). Future keys, which a
//! causal model never attends to anyway, give `dist < 0` and are likewise zero.
//!
//! The reference clamps the gather index into range and *then* masks the result; this
//! kernel masks the load itself, so out-of-range distances are never addressed.
//!
//! Grid: `ceil(KV / BLOCK)` by `T * heads`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gather above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn rel_bias_forward<T: Triton, D: Float, const BLOCK: i32>(
    logits_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: OutPtr<T::Pointer<D>>,
    KV: i32,
    HEADS: i32,
    EXTENT: i32,
    PAST: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: core::ops::BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One row per (token, head); the token's absolute position includes the cache.
    let row = T::program_id(Axis::Y);
    let bid = T::program_id(Axis::X);
    let pos = PAST + row / HEADS;

    let j = T::arange(0, BLOCK) + bid * BLOCK;
    let in_kv = j.lt(KV);
    // dist counts backwards from the query to the key. Written as a negated arange
    // plus a scalar, since the DSL has tensor-scalar arithmetic but not tensor-tensor
    // subtraction.
    let dist = T::arange(0, BLOCK) * -1 + (pos - bid * BLOCK);
    let valid = in_kv & dist.ge(0) & dist.lt(EXTENT);

    // Masked lanes are not loaded at all, so an out-of-range dist never addresses
    // memory; the address it would have used is irrelevant.
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        logits_ptr.add_offsets(dist + row * EXTENT),
        Some(valid),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        bias_ptr.add_offsets(j + row * KV),
        x,
        Some(in_kv),
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

const BLOCK: i32 = 256;

/// Graph-level relative-bias gather: `[T * heads, rel_extent]` to `[T * heads, KV]`.
pub struct RelBiasOp<D: FloatBytes + Send + Sync + 'static> {
    /// Attention heads.
    pub heads: usize,
    /// How far back the bias reaches. 1024 in Inkling-2-Mini.
    pub rel_extent: usize,
    /// Keys already in the cache, so query `t` sits at `past + t`.
    pub past: usize,
    /// Keys attended over: `past + T`.
    pub kv_len: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RelBiasOp<D> {
    /// A gather for `heads` heads reaching `rel_extent` back, with `past` cached keys
    /// and `kv_len` keys in all.
    pub fn new(heads: usize, rel_extent: usize, past: usize, kv_len: usize) -> Self {
        assert!(
            kv_len >= past,
            "rel bias: kv_len {kv_len} is shorter than the {past} keys already cached"
        );
        Self {
            heads,
            rel_extent,
            past,
            kv_len,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for RelBiasOp<D> {
    fn name(&self) -> &str {
        "llm.rel_bias"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.kv_len)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = RelBiasForward::<D>::new(BLOCK);
        Some((
            "rel_bias_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(RelBiasRuntimeOp::<D> {
                heads: self.heads,
                rel_extent: self.rel_extent,
                past: self.past,
                kv_len: self.kv_len,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`RelBiasOp`].
pub struct RelBiasRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    heads: usize,
    rel_extent: usize,
    past: usize,
    kv_len: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for RelBiasRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
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
        assert_eq!(inputs[0].1[1], self.rel_extent, "rel bias: profile width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.kv_len as i32);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.rel_extent as i32);
        visitor.visit_i32(self.past as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.kv_len.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
