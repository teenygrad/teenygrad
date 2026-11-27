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

//! Inkling-2's token embedding (`Inkling2LoopNormedEmbedding`): a plain lookup,
//! `y[t, :] = w[ids[t], :]`.
//!
//! **Not Gemma's.** `Gemma3TextScaledWordEmbedding` multiplies by `sqrt(hidden)`;
//! Inkling-2 does not scale at all, it follows the lookup with an RMS norm
//! (`embed_norm`, a separate [`super::norm::RmsNormOp`]). Reusing the Gemma kernel
//! here would scale every embedding by 48 for Mini's width of 2304.
//!
//! Token ids arrive as an `f32` activation `[T]`, exact for ids below 2^24 against a
//! vocabulary of 201024. Grid: `T * ceil(H / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The embedding lookup above, on bf16 weights two to a word: word `j` of a row
/// holds column `j` in its low half and column `j + H / 2` in its high half
/// ([`crate::kernels::linear::pack_bf16_halves`]), and a bf16 is the top half of its
/// f32, so each half becomes the exact f32 with a shift or a mask. Half the bytes of
/// an f32 table, which for a 201024-token vocabulary is 0.9 GB less to load.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn embed_forward<T: Triton, D: Float, const BLOCK: i32>(
    ids_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    H: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let half = H / 2;
    let tiles = (half + BLOCK - 1) / BLOCK;
    let pid = T::program_id(Axis::X);
    let t = pid / tiles;
    let tile = pid % tiles;
    let izero = T::zeros::<i32>(&[BLOCK]);
    let j = T::arange(0, BLOCK) + tile * BLOCK;
    let m = j.lt(half);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let id = T::load(
        ids_ptr.add_offsets(izero + t),
        None,
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let row = T::cast::<D, i32>(id, None, false);
    let word = T::cast::<D, i32>(
        T::load(w_ptr.add_offsets(j + row * half), Some(m), Some(zeros), &[], None, None, None, false),
        None,
        true,
    );
    let lo = T::cast::<i32, D>(word << 16, None, true);
    let hi = T::cast::<i32, D>(word & -65536, None, true);
    T::store(y_ptr.add_offsets(j + t * H), lo, Some(m), &[], None, None);
    T::store(y_ptr.add_offsets(j + (t * H + half)), hi, Some(m), &[], None, None);
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

/// Graph-level embedding: ids `[T]` to `[T, hidden]`. Weight slot `weight_bf16`,
/// `[vocab, hidden / 2]` words of bf16 pairs.
pub struct EmbedOp<D: FloatBytes + Send + Sync + 'static> {
    /// Vocabulary size.
    pub vocab: usize,
    /// Hidden size.
    pub hidden: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> EmbedOp<D> {
    /// Creates the op.
    pub fn new(vocab: usize, hidden: usize) -> Self {
        assert!(hidden.is_multiple_of(2), "embed: hidden size must be even for bf16 pairs");
        Self {
            vocab,
            hidden,
            _phantom: PhantomData,
        }
    }
}

const EMBED_BLOCK: i32 = 128;

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for EmbedOp<D> {
    fn name(&self) -> &str {
        "gemma.embed"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.hidden)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = EmbedForward::<D>::new(EMBED_BLOCK);
        Some((
            "embed_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(EmbedRuntimeOp::<D> {
                vocab: self.vocab,
                hidden: self.hidden,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`EmbedOp`].
pub struct EmbedRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    vocab: usize,
    hidden: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for EmbedRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.vocab, self.hidden / 2]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_bf16"]
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.hidden as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let tiles = (self.hidden / 2).div_ceil(EMBED_BLOCK as usize);
        [(output_shape[0] * tiles) as u32, 1, 1]
    }
}
