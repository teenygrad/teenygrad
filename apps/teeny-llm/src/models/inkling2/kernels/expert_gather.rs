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

//! The expert gather: a linear projection whose weight matrix is chosen per row.
//!
//! ```text
//! y[s, n] = sum(k) x[token(s), k] * W[expert(s), n, k]
//! ```
//!
//! One output row is one (token, slot) pair, `slot` being which of that token's
//! `top_k` experts this is, so `T` tokens give `T * top_k` rows and the router
//! decided which expert each belongs to.
//!
//! This is what [`crate::kernels::linear::Linear4Op`] cannot express: there the whole
//! batch shares one weight matrix, while here each row reads a different slice of a
//! `[experts, N, K]` parameter. The gather lives in the addressing, so no expert
//! weight is ever copied or materialised.
//!
//! The expert ids are read in place from the second half of the router's packed row,
//! as floats: `IDS_STRIDE` is the router row width (`2 * top_k`) and `IDS_OFFSET` is
//! where the ids start (`top_k`). That avoids an unpacking pass between the two.
//!
//! Inkling-2-Mini runs this twice per expert, once for the fused gate-and-up
//! projection (`w13`, `N = 2 * intermediate`) and once for the down projection
//! (`w2`, `N = hidden`).
//!
//! Lanes run along `K`, as in `Linear4Op`, and are reduced at the end. One output
//! per program, unlike `Linear4Op`'s four: the weight rows of different slots are far
//! apart, so there is no load of `x` to share between them.
//!
//! Grid: `N` by `T * top_k`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gathered projection above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_gather_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    ids_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
    TOP_K: i32,
    IDS_STRIDE: i32,
    IDS_OFFSET: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let n = T::program_id(Axis::X);
    let slot = T::program_id(Axis::Y);
    let token = slot / TOP_K;
    let which = slot - token * TOP_K;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // The id sits in the router's packed row as a float; broadcast it so it can take
    // part in the block-wide address arithmetic below.
    let id = T::load(
        ids_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (token * IDS_STRIDE + IDS_OFFSET + which)),
        None,
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let expert = T::cast::<D, i32>(id, None, false);

    let mut acc = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (K + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let k = offs + i * BLOCK;
        let km = k.lt(K);
        let x = T::load(
            x_ptr.add_offsets(k + token * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        // W is [experts, N, K]: the chosen expert's plane, then row n.
        let w = T::load(
            w_ptr.add_offsets((k + n * K) + expert * (N * K)),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + x * w;
    }
    // One scalar per program: reduce the lanes, then write a single element.
    let out = T::sum(acc, Some(0), true);
    T::store(
        y_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (slot * N + n)),
        out,
        Some(offs.lt(1)),
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

/// Graph-level expert gather. Inputs are `x` of `[T, K]` and the router's packed
/// `[T, 2 * top_k]`; the output is `[T * top_k, N]`.
pub struct ExpertGatherOp<D: FloatBytes + Send + Sync + 'static> {
    /// Experts the parameter holds.
    pub experts: usize,
    /// Outputs per expert row.
    pub n: usize,
    /// Reduction width.
    pub k: usize,
    /// Experts firing per token.
    pub top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ExpertGatherOp<D> {
    /// A gather over `experts` planes of `[n, k]`, with `top_k` slots per token.
    pub fn new(experts: usize, n: usize, k: usize, top_k: usize) -> Self {
        assert!(
            top_k > 0 && top_k <= experts,
            "expert gather: top_k out of range"
        );
        Self {
            experts,
            n,
            k,
            top_k,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertGatherOp<D> {
    fn name(&self) -> &str {
        "llm.expert_gather"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertGatherForward::<D>::new(BLOCK);
        Some((
            "expert_gather_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertGatherRuntimeOp::<D> {
                experts: self.experts,
                n: self.n,
                k: self.k,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertGatherOp`].
pub struct ExpertGatherRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    n: usize,
    k: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertGatherRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.experts, self.n, self.k]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
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
        assert_eq!(inputs[0].1[1], self.k, "expert gather: input width");
        let ids_stride = inputs[1].1[1];
        assert!(
            ids_stride >= self.top_k,
            "expert gather: router row of {ids_stride} cannot hold {} ids",
            self.top_k
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(self.top_k as i32);
        visitor.visit_i32(ids_stride as i32);
        // The ids sit after the weights in the router's packed row.
        visitor.visit_i32((ids_stride - self.top_k) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [self.n as u32, output_shape[0] as u32, 1]
    }
}
