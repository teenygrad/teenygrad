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

//! The weighted scatter-add that closes Inkling-2's mixture of experts:
//!
//! ```text
//! out[t, n] = sum(slot) weight[t, slot] * y[t * top_k + slot, n]
//! ```
//!
//! Each token's `top_k` expert outputs are combined with the weights the router
//! produced, giving back one row per token.
//!
//! The reference does this with `index_add_` into a zeroed buffer, scattering from
//! whichever tokens hit a given expert. Here the arithmetic is turned round into a
//! **gather over slots**: one program owns one output row and reads its own `top_k`
//! contributions. Nothing is written twice, so no atomics and no accumulation order
//! to depend on, and the result is deterministic. That only works because the slots
//! of a token are contiguous, which is how the expert gather lays them out.
//!
//! Weights are read in place from the first half of the router's packed row;
//! `WEIGHTS_STRIDE` is the row width (`2 * top_k`).
//!
//! The reference accumulates in fp32 whatever the model dtype is, so the caller
//! should keep this at f32. Grid: `ceil(N / BLOCK)` by `T`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The weighted combination above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_scatter_add_forward<T: Triton, D: Float, const BLOCK: i32>(
    y_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    out_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    TOP_K: i32,
    WEIGHTS_STRIDE: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let bid = T::program_id(Axis::X);
    let token = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK) + bid * BLOCK;
    let m = offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    for slot in 0..TOP_K {
        let wv = T::load(
            w_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (token * WEIGHTS_STRIDE + slot)),
            None,
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let y = T::load(
            y_ptr.add_offsets(offs + (token * TOP_K + slot) * N),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + y * T::broadcast_to(wv, &[BLOCK]);
    }
    T::store(
        out_ptr.add_offsets(offs + token * N),
        acc,
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

const BLOCK: i32 = 256;

/// Graph-level weighted combine: `[T * top_k, N]` plus the router row to `[T, N]`.
pub struct ExpertScatterAddOp<D: FloatBytes + Send + Sync + 'static> {
    /// Width of a row.
    pub n: usize,
    /// Experts firing per token.
    pub top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ExpertScatterAddOp<D> {
    /// A combine over `top_k` slots of `n` lanes.
    pub fn new(n: usize, top_k: usize) -> Self {
        assert!(top_k > 0, "scatter add: top_k must be positive");
        Self {
            n,
            top_k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertScatterAddOp<D> {
    fn name(&self) -> &str {
        "llm.expert_scatter_add"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|s| s / self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertScatterAddForward::<D>::new(BLOCK);
        Some((
            "expert_scatter_add_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertScatterAddRuntimeOp::<D> {
                n: self.n,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertScatterAddOp`].
pub struct ExpertScatterAddRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    n: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertScatterAddRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    /// One row per token: the input's (token, slot) rows divided by `top_k`.
    fn compute_concrete_output_shape(
        &self,
        input_shapes: &[&[usize]],
        _resolved: &[usize],
    ) -> Vec<usize> {
        vec![input_shapes[0][0] / self.top_k, self.n]
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
        assert_eq!(inputs[0].1[1], self.n, "scatter add: expert output width");
        let stride = inputs[1].1[1];
        assert!(
            stride >= self.top_k,
            "scatter add: router row of {stride} cannot hold {} weights",
            self.top_k
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.top_k as i32);
        visitor.visit_i32(stride as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.n.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
