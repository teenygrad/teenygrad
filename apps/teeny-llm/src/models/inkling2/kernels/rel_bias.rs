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

pub use crate::kernels::attention::PastLen;

const BLOCK: i32 = 256;

/// Graph-level relative-bias gather: `[T * heads, rel_extent]` to `[T * heads, KV]`.
///
/// The past length is read from a shared [`PastLen`] at each launch, as
/// [`super::attention`] does, so one compiled graph serves every step of a sequence.
/// Baking it in would mean recompiling the whole model for every token.
pub struct RelBiasOp<D: FloatBytes + Send + Sync + 'static> {
    /// Attention heads.
    pub heads: usize,
    /// How far back the bias reaches. 1024 in Inkling-2-Mini.
    pub rel_extent: usize,
    /// Cached keys, read at launch.
    pub past: PastLen,
    /// A pinned output width, for a graph reused across a whole sequence.
    ///
    /// The runtime resolves output shapes once, when the graph is compiled, so a bias
    /// whose width tracks `past` can only ever be right for the length it was compiled
    /// at. Pinning it to the cache capacity makes the shape constant; keys past
    /// `past + T` fall outside the causal range and come out zero anyway.
    pub width: Option<usize>,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RelBiasOp<D> {
    /// A gather for `heads` heads reaching `rel_extent` back, reading the cache
    /// length from `past` at each launch.
    pub fn new(heads: usize, rel_extent: usize, past: PastLen) -> Self {
        Self::with_width(heads, rel_extent, past, None)
    }

    /// As [`Self::new`], with the output width pinned to `width` when given.
    pub fn with_width(
        heads: usize,
        rel_extent: usize,
        past: PastLen,
        width: Option<usize>,
    ) -> Self {
        Self {
            width,
            heads,
            rel_extent,
            past,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for RelBiasOp<D> {
    fn name(&self) -> &str {
        "llm.rel_bias"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // KV is not known until launch; the runtime op allocates by the concrete
        // shape it computes below.
        vec![input_shapes[0][0], None]
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
                past: self.past.clone(),
                width: self.width,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`RelBiasOp`].
pub struct RelBiasRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    heads: usize,
    rel_extent: usize,
    past: PastLen,
    width: Option<usize>,
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
        let past = self.past.get();
        let kv = self
            .width
            .unwrap_or_else(|| past + inputs[0].1[0] / self.heads);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(kv as i32);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.rel_extent as i32);
        visitor.visit_i32(past as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let kv = self
            .width
            .unwrap_or_else(|| self.past.get() + output_shape[0] / self.heads);
        [
            kv.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }

    fn compute_concrete_output_shape(
        &self,
        _input_shapes: &[&[usize]],
        resolved: &[usize],
    ) -> Vec<usize> {
        // [T * heads, past + T]: the width grows with the cache, so it cannot come
        // from the symbolic shape.
        //
        // The runtime resolves this once, when the graph is compiled, so a generation
        // loop that reuses one compiled graph has to pin the width instead: `width`
        // holds the whole cache and the keys beyond `past + T` come out zero, which
        // the distance check already does for them.
        let rows = resolved[0];
        vec![
            rows,
            self.width
                .unwrap_or_else(|| self.past.get() + rows / self.heads),
        ]
    }
}

// ---------------------------------------------------------------------------
// Clip, projection and gather in one kernel
// ---------------------------------------------------------------------------

/// The whole relative bias after `wr_du` in one launch: the scaled L2 clip of each
/// (token, head)'s `d_rel` vector, its projection onto the `rel_extent` distance
/// profile, and that profile read backwards onto the keys:
///
/// ```text
/// r[d]          = x[t, h, d] * s[h, d], clipped to norm THRESHOLD
/// bias[t, h, j] = sum(d) proj[d, pos - j] * r[d]   for 0 <= pos - j < EXTENT and j < KV
///               = 0                                otherwise
/// ```
///
/// Separately this was four launches (clip, regroup, a linear to `[T * heads,
/// EXTENT]`, gather) and a round trip of that profile through memory. Here each
/// program owns BLOCK distances of one (token, head): it streams `proj` in the
/// checkpoint's own `[d_rel, EXTENT]` layout, one contiguous row per `d`, and stores
/// to keys `pos - dist`, a reversed contiguous run. It also zeroes the BLOCK keys
/// after the query that line up with its distances, so every key past the query
/// comes out zero, as the gather's mask made them.
///
/// Grid: `ceil(EXTENT / BLOCK)` by `T * heads`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn rel_bias_fused_forward<T: Triton, D: Float, const BLOCK: i32, const D_REL: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    s_ptr: InPtr<T::Pointer<D>>,
    p_ptr: InPtr<T::Pointer<D>>,
    bias_ptr: OutPtr<T::Pointer<D>>,
    KV: i32,
    HEADS: i32,
    EXTENT: i32,
    PAST: i32,
    THRESHOLD: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::BoolTensor: core::ops::BitAnd<Output = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let bid = T::program_id(Axis::X);
    let row = T::program_id(Axis::Y);
    let head = row - (row / HEADS) * HEADS;
    let pos = PAST + row / HEADS;
    let splat = T::zeros::<i32>(&[BLOCK]);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // The clip's scale factor, uniform across the program: the norm of x * s over
    // this head's D_REL values, each read as a broadcast.
    let mut ss = T::zeros::<D>(&[BLOCK]);
    for d in 0..D_REL {
        let v = T::load(
            x_ptr.add_offsets(splat + (row * D_REL + d)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ) * T::load(
            s_ptr.add_offsets(splat + (head * D_REL + d)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        ss = ss + v * v;
    }
    let ratio = T::minimum(
        T::full(&[BLOCK], D::from_f64(THRESHOLD as f64))
            / (T::sqrt(ss) + T::full(&[BLOCK], D::from_f64(1e-6))),
        T::full(&[BLOCK], D::from_f64(1.0)),
    );

    // This program's distances, and the keys they land on.
    let dist = T::arange(0, BLOCK) + bid * BLOCK;
    let in_extent = dist.lt(EXTENT);
    let key = T::arange(0, BLOCK) * -1 + (pos - bid * BLOCK);
    let on_key = in_extent & key.ge(0) & key.lt(KV);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    for d in 0..D_REL {
        let r = T::load(
            x_ptr.add_offsets(splat + (row * D_REL + d)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        ) * T::load(
            s_ptr.add_offsets(splat + (head * D_REL + d)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let p = T::load(
            p_ptr.add_offsets(dist + d * EXTENT),
            Some(in_extent),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + p * r;
    }
    T::store(
        bias_ptr.add_offsets(key + row * KV),
        acc * ratio,
        Some(on_key),
        &[],
        None,
        None,
    );

    // Keys after the query: pos + 1 + dist, while within the row.
    let later = T::arange(0, BLOCK) + (pos + 1 + bid * BLOCK);
    T::store(
        bias_ptr.add_offsets(later + row * KV),
        zeros,
        Some(later.lt(KV)),
        &[],
        None,
        None,
    );
}

/// Graph-level relative bias from `wr_du`'s output `[T, heads * d_rel]` to the
/// `[T * heads, KV]` bias, in one launch ([`rel_bias_fused_forward`]). Parameters:
/// `scale` `[heads, d_rel]` (the clip's) and `proj` `[d_rel, rel_extent]`, the
/// checkpoint's own layout.
pub struct RelBiasFusedOp<D: FloatBytes + Send + Sync + 'static> {
    /// Attention heads.
    pub heads: usize,
    /// Values per head of the relative states (16 in Inkling-2-Mini).
    pub d_rel: usize,
    /// How far back the bias reaches.
    pub rel_extent: usize,
    /// The clip's norm bound.
    pub threshold: f32,
    /// Cached keys, read at launch.
    pub past: PastLen,
    /// A pinned output width, as [`RelBiasOp::width`].
    pub width: Option<usize>,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RelBiasFusedOp<D> {
    /// The fused bias; see [`RelBiasFusedOp`].
    pub fn new(
        heads: usize,
        d_rel: usize,
        rel_extent: usize,
        threshold: f32,
        past: PastLen,
        width: Option<usize>,
    ) -> Self {
        assert_eq!(d_rel, 16, "rel bias: only d_rel 16 is compiled");
        Self {
            heads,
            d_rel,
            rel_extent,
            threshold,
            past,
            width,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for RelBiasFusedOp<D> {
    fn name(&self) -> &str {
        "llm.rel_bias_fused"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.heads), None]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = RelBiasFusedForward::<D>::new(BLOCK, 16);
        Some((
            "rel_bias_fused_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(RelBiasFusedRuntimeOp {
                heads: self.heads,
                d_rel: self.d_rel,
                rel_extent: self.rel_extent,
                threshold: self.threshold,
                past: self.past.clone(),
                width: self.width,
            }),
        ))
    }
}

/// Runtime dispatch for [`RelBiasFusedOp`].
pub struct RelBiasFusedRuntimeOp {
    heads: usize,
    d_rel: usize,
    rel_extent: usize,
    threshold: f32,
    past: PastLen,
    width: Option<usize>,
}

impl RelBiasFusedRuntimeOp {
    fn kv(&self, tokens: usize) -> usize {
        self.width.unwrap_or_else(|| self.past.get() + tokens)
    }
}

impl RuntimeOp for RelBiasFusedRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![
            vec![self.heads, self.d_rel],
            vec![self.d_rel, self.rel_extent],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["scale", "proj"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(
            inputs[0].1[1],
            self.heads * self.d_rel,
            "rel bias: input width"
        );
        // Keys further back than the extent are not written, which is right only
        // while none exist: the row must be no wider than the extent.
        let kv = self.kv(inputs[0].1[0]);
        assert!(
            kv <= self.rel_extent,
            "rel bias: {kv} keys exceed the {} the fused kernel covers",
            self.rel_extent
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(kv as i32);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.rel_extent as i32);
        visitor.visit_i32(self.past.get() as i32);
        visitor.visit_f32(self.threshold);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.rel_extent.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }

    fn compute_concrete_output_shape(
        &self,
        input_shapes: &[&[usize]],
        _resolved: &[usize],
    ) -> Vec<usize> {
        let t = input_shapes[0][0];
        vec![t * self.heads, self.kv(t)]
    }
}
