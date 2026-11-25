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

//! Bias-free `nn.Linear`: `y[m, n] = sum_k x[m, k] * w[n, k]`, with PyTorch's weight
//! layout `[N, K]`.
//!
//! A portable kernel for both backends (teeny's `LinearForward` uses CUDA tensor
//! descriptors). Each program computes four consecutive outputs of one row, so every
//! load of `x` feeds four accumulators. Lanes run along `K` (contiguous in both `x` and
//! `w`), and each accumulator is reduced across lanes at the end.
//!
//! Grid: `M * N / 4` (`N` must be a multiple of 4).

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The linear layer above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear4_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let groups = N / 4;
    let m = pid / groups;
    let n = (pid % groups) * 4;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut a0 = T::zeros::<D>(&[BLOCK]);
    let mut a1 = T::zeros::<D>(&[BLOCK]);
    let mut a2 = T::zeros::<D>(&[BLOCK]);
    let mut a3 = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (K + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let k = offs + i * BLOCK;
        let km = k.lt(K);
        let x = T::load(
            x_ptr.add_offsets(k + m * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let w0 = T::load(
            w_ptr.add_offsets(k + n * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a0 = a0 + x * w0;
        let w1 = T::load(
            w_ptr.add_offsets(k + (n + 1) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a1 = a1 + x * w1;
        let w2 = T::load(
            w_ptr.add_offsets(k + (n + 2) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a2 = a2 + x * w2;
        let w3 = T::load(
            w_ptr.add_offsets(k + (n + 3) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a3 = a3 + x * w3;
    }
    // Each sum is a one-element tensor; store it through a one-lane mask.
    let one = offs.lt(1);
    let base = m * N + n;
    T::store(
        y_ptr.add_offsets(offs + base),
        T::sum(a0, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 1)),
        T::sum(a1, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 2)),
        T::sum(a2, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 3)),
        T::sum(a3, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
}

/// [`linear4_forward`] with program ids numbered down the rows: consecutive ids compute
/// the same four outputs for consecutive rows. The launcher hands each thread a run of
/// consecutive ids, so a thread reuses those outputs' weights (`4 * K` values) from
/// cache across rows instead of reading them from memory once per row.
///
/// Grid: `M * N / 4`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear4_rows_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let m = pid % M;
    let n = (pid / M) * 4;

    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut a0 = T::zeros::<D>(&[BLOCK]);
    let mut a1 = T::zeros::<D>(&[BLOCK]);
    let mut a2 = T::zeros::<D>(&[BLOCK]);
    let mut a3 = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (K + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let k = offs + i * BLOCK;
        let km = k.lt(K);
        let x = T::load(
            x_ptr.add_offsets(k + m * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        let w0 = T::load(
            w_ptr.add_offsets(k + n * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a0 = a0 + x * w0;
        let w1 = T::load(
            w_ptr.add_offsets(k + (n + 1) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a1 = a1 + x * w1;
        let w2 = T::load(
            w_ptr.add_offsets(k + (n + 2) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a2 = a2 + x * w2;
        let w3 = T::load(
            w_ptr.add_offsets(k + (n + 3) * K),
            Some(km),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        );
        a3 = a3 + x * w3;
    }
    // Each sum is a one-element tensor; store it through a one-lane mask.
    let one = offs.lt(1);
    let base = m * N + n;
    T::store(
        y_ptr.add_offsets(offs + base),
        T::sum(a0, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 1)),
        T::sum(a1, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 2)),
        T::sum(a2, Some(0), true) + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        y_ptr.add_offsets(offs + (base + 3)),
        T::sum(a3, Some(0), true) + zeros,
        Some(one),
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

/// Graph-level bias-free linear layer, `[M, K]` to `[M, N]`. Weight slot `weight`
/// `[N, K]`.
pub struct Linear4Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input features `K`.
    pub k: usize,
    /// Output features `N` (a multiple of 4).
    pub n: usize,
    /// Lanes along `K`.
    pub block: i32,
    /// Number program ids down the rows ([`linear4_rows_forward`]) rather than along
    /// them, so weights are reused from cache across rows.
    pub weight_reuse: bool,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Linear4Op<D> {
    /// The op with [`Self::weight_reuse`] set.
    pub fn with_weight_reuse(mut self) -> Self {
        self.weight_reuse = true;
        self
    }

    /// Creates the op.
    pub fn new(k: usize, n: usize, block: i32) -> Self {
        assert!(
            n.is_multiple_of(4),
            "linear4: out features must be a multiple of 4"
        );
        Self {
            k,
            n,
            block,
            weight_reuse: false,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Linear4Op<D> {
    fn name(&self) -> &str {
        "llm.linear4"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let (name, source) = if self.weight_reuse {
            (
                "linear4_rows_forward",
                Linear4RowsForward::<D>::new(self.block).source,
            )
        } else {
            (
                "linear4_forward",
                Linear4Forward::<D>::new(self.block).source,
            )
        };
        Some((
            name.to_string(),
            source,
            "entry_point".to_string(),
            Arc::new(Linear4RuntimeOp::<D> {
                k: self.k,
                n: self.n,
                weight_reuse: self.weight_reuse,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`Linear4Op`].
pub struct Linear4RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    n: usize,
    weight_reuse: bool,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Linear4RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[0].1[1], self.k, "linear4: input features");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        if self.weight_reuse {
            visitor.visit_i32(output_shape[0] as i32);
        }
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.n / 4) as u32, 1, 1]
    }
}
