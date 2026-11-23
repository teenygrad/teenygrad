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

//! Inkling-2's scaled L2 clip (`Inkling2LoopScaledL2Clip`): a learned per-lane scale,
//! then each head vector shortened to `threshold` if it is longer, over rows of
//! `head_dim` inside a `[T, heads * head_dim]` tensor.
//!
//! ```text
//! v = x * scale[head]
//! y = v * min(threshold / (||v||_2 + 1e-6), 1)
//! ```
//!
//! Two details that are easy to get wrong and are taken from the reference rather
//! than inferred:
//!
//! - the `1e-6` is added to the **norm**, not to its square, and not under the root;
//! - the ratio is **clamped at 1**, so a vector already shorter than `threshold` is
//!   left exactly as the scale made it rather than being stretched up to it.
//!
//! The scale is `[heads, head_dim]`, one row per head, applied before the norm is
//! taken. Queries use threshold `qk_l2_clip_threshold`, the relative term
//! `rel_l2_clip_threshold`.
//!
//! Grid: one program per head vector, `T * heads` of them; `BLOCK` (a power of two)
//! must be at least `head_dim`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The scaled clip above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn scaled_l2_clip_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    s_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    HEADS: i32,
    HEAD_DIM: i32,
    THRESHOLD: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    // One program per head vector: pid = token * HEADS + head, so the row base is
    // pid * HEAD_DIM and the scale row is (pid % HEADS) * HEAD_DIM.
    let pid = T::program_id(Axis::X);
    let head = pid - (pid / HEADS) * HEADS;
    let offs = T::arange(0, BLOCK);
    let m = offs.lt(HEAD_DIM);
    let zeros = T::zeros::<D>(&[BLOCK]);

    let x = T::load(
        x_ptr.add_offsets(offs + pid * HEAD_DIM),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let s = T::load(
        s_ptr.add_offsets(offs + head * HEAD_DIM),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // Masked lanes load as zero, so they add nothing to the sum of squares.
    let v = x * s;
    let norm = T::sqrt(T::sum(v * v, Some(0), true));
    let eps = T::full(&[BLOCK], D::from_f64(1e-6));
    let thr = T::full(&[BLOCK], D::from_f64(THRESHOLD as f64));
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let ratio = T::minimum(thr / (norm + eps), one);
    T::store(
        y_ptr.add_offsets(offs + pid * HEAD_DIM),
        v * ratio,
        Some(m),
        &[],
        None,
        None,
    );
}

/// The same clip with no learned scale (`Inkling2LoopL2Clip`), which is what the
/// KEY projection uses. Queries and the relative term have a scale; keys do not.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn l2_clip_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    HEAD_DIM: i32,
    THRESHOLD: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK);
    let m = offs.lt(HEAD_DIM);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let v = T::load(
        x_ptr.add_offsets(offs + pid * HEAD_DIM),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let norm = T::sqrt(T::sum(v * v, Some(0), true));
    let ratio = T::minimum(
        T::full(&[BLOCK], D::from_f64(THRESHOLD as f64))
            / (norm + T::full(&[BLOCK], D::from_f64(1e-6))),
        T::full(&[BLOCK], D::from_f64(1.0)),
    );
    T::store(
        y_ptr.add_offsets(offs + pid * HEAD_DIM),
        v * ratio,
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

/// Graph-level scaled L2 clip over `[T, heads * head_dim]`, shape preserving.
pub struct ScaledL2ClipOp<D: FloatBytes + Send + Sync + 'static> {
    /// Heads in the tensor.
    pub heads: usize,
    /// Lanes per head.
    pub head_dim: usize,
    /// Length the head vector is shortened to.
    pub threshold: f32,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ScaledL2ClipOp<D> {
    /// A clip over `heads` heads of `head_dim` lanes at `threshold`.
    pub fn new(heads: usize, head_dim: usize, threshold: f32) -> Self {
        assert!(
            threshold > 0.0,
            "l2 clip: threshold {threshold} is not positive"
        );
        Self {
            heads,
            head_dim,
            threshold,
            block: head_dim.next_power_of_two() as i32,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ScaledL2ClipOp<D> {
    fn name(&self) -> &str {
        "llm.scaled_l2_clip"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ScaledL2ClipForward::<D>::new(self.block);
        Some((
            "scaled_l2_clip_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ScaledL2ClipRuntimeOp::<D> {
                heads: self.heads,
                head_dim: self.head_dim,
                threshold: self.threshold,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ScaledL2ClipOp`].
pub struct ScaledL2ClipRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    heads: usize,
    head_dim: usize,
    threshold: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ScaledL2ClipRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.heads, self.head_dim]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["scale"]
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
            self.heads * self.head_dim,
            "l2 clip: input width"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.heads as i32);
        visitor.visit_i32(self.head_dim as i32);
        visitor.visit_f32(self.threshold);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * self.heads) as u32, 1, 1]
    }
}

/// Graph-level L2 clip with no learned scale, over rows of `head_dim`.
pub struct L2ClipOp<D: FloatBytes + Send + Sync + 'static> {
    /// Lanes per head.
    pub head_dim: usize,
    /// Length the head vector is shortened to.
    pub threshold: f32,
    block: i32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> L2ClipOp<D> {
    /// A clip over rows of `head_dim` lanes at `threshold`.
    pub fn new(head_dim: usize, threshold: f32) -> Self {
        assert!(
            threshold > 0.0,
            "l2 clip: threshold {threshold} is not positive"
        );
        Self {
            head_dim,
            threshold,
            block: head_dim.next_power_of_two() as i32,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for L2ClipOp<D> {
    fn name(&self) -> &str {
        "llm.l2_clip"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = L2ClipForward::<D>::new(self.block);
        Some((
            "l2_clip_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(L2ClipRuntimeOp::<D> {
                head_dim: self.head_dim,
                threshold: self.threshold,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`L2ClipOp`].
pub struct L2ClipRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    head_dim: usize,
    threshold: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for L2ClipRuntimeOp<D> {
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
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.head_dim as i32);
        visitor.visit_f32(self.threshold);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            (output_shape.iter().product::<usize>() / self.head_dim) as u32,
            1,
            1,
        ]
    }
}
