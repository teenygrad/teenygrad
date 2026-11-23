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

//! Two-operand contraction over one summed axis (ONNX `Einsum`).
//!
//! One kernel for the family an einsum equation describes once a single label is
//! contracted:
//!
//! ```text
//! y[coords] = sum over c in 0..K of
//!     a[dot(coords, a_strides) + c * a_contract_stride]
//!   * b[dot(coords, b_strides) + c * b_contract_stride]
//! ```
//!
//! The strides come from [`teeny_core::graph::einsum_plan`], which reads them off the
//! equation, so the kernel never sees a label. A label an operand does not mention
//! gets stride 0 there, and every coordinate along it then reads the same element,
//! which is how a label free on one side broadcasts along the other's axes with no
//! special case. A label in both operands and the output is a batch axis and simply
//! has a stride on both sides.
//!
//! That makes a batched matrix multiply one instance of this kernel rather than a
//! different thing: `bij,bjk->bik` is a batch label, one free label on each side and
//! one contracted label.
//!
//! Shapes are [`MAX_RANK`] dimensions, outermost first and row-major, with a
//! lower-rank output left-padded by extent 1 and stride 0, following
//! [`crate::nn::tensor::strided_copy`]. [`MAX_RANK`] is 4 here rather than 6: three
//! pointers and sixteen scalars is the whole 20-argument kernel ABI, and twelve of
//! those scalars are the extent and two stride vectors.
//!
//! Grid: `cdiv(n_elements, BLOCK_SIZE)`, one CTA per block of output elements. Each
//! lane walks the contraction serially, so this is a correct contraction rather than a
//! fast one; a tiled kernel would reuse operands across lanes.

use teeny_core::dtype::{AddOffsets, Comparison, Num, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

pub use teeny_core::graph::EINSUM_MAX_RANK as MAX_RANK;

/// Contract `a` and `b` over one axis of length `k`.
///
/// `s0..s3` are the output extents, `a0..a3` and `b0..b3` the per-output-coordinate
/// strides into each operand, all outermost first. Unused leading dimensions take
/// extent 1 and stride 0.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub fn einsum_forward<T, D: Num, const BLOCK_SIZE: i32>(
    a_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    k: i32,
    a_step: i32,
    b_step: i32,
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
    a0: i32,
    a1: i32,
    a2: i32,
    a3: i32,
    b0: i32,
    b1: i32,
    b2: i32,
    b3: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // Peel the flat output index into coordinates, fastest-varying first, and
    // accumulate each operand's base offset as we go. There is no integer remainder in
    // the kernel DSL, so each coordinate is `a - (a / b) * b`.
    let mut a_base = T::zeros::<i32>(&[BLOCK_SIZE]);
    let mut b_base = T::zeros::<i32>(&[BLOCK_SIZE]);
    let mut rem = p;

    let q3 = rem / s3;
    a_base = a_base + (rem - q3 * s3) * a3;
    b_base = b_base + (rem - q3 * s3) * b3;
    rem = q3;

    let q2 = rem / s2;
    a_base = a_base + (rem - q2 * s2) * a2;
    b_base = b_base + (rem - q2 * s2) * b2;
    rem = q2;

    let q1 = rem / s1;
    a_base = a_base + (rem - q1 * s1) * a1;
    b_base = b_base + (rem - q1 * s1) * b1;
    rem = q1;

    let q0 = rem / s0;
    a_base = a_base + (rem - q0 * s0) * a0;
    b_base = b_base + (rem - q0 * s0) * b0;

    let mut acc = T::zeros::<D>(&[BLOCK_SIZE]);
    let mut c: i32 = 0;
    while c < k {
        let av = T::load(
            a_ptr.add_offsets(a_base + c * a_step),
            Some(in_bounds),
            Some(T::zeros::<D>(&[BLOCK_SIZE])),
            &[],
            None,
            None,
            None,
            false,
        );
        let bv = T::load(
            b_ptr.add_offsets(b_base + c * b_step),
            Some(in_bounds),
            Some(T::zeros::<D>(&[BLOCK_SIZE])),
            &[],
            None,
            None,
            None,
            false,
        );
        acc = acc + av * bv;
        c += 1;
    }

    T::store(y_ptr.add_offsets(p), acc, Some(in_bounds), &[], None, None);
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use teeny_core::device::program::ArgVisitor;
use teeny_core::graph::EinsumPlan;
use teeny_core::model::{RawPtr, RuntimeOp};

/// Left-pads `values` to [`MAX_RANK`] with `fill`, keeping the trailing (fastest)
/// dimensions in place.
fn pad_left(values: &[i64], fill: i64) -> [i32; MAX_RANK] {
    let mut out = [fill as i32; MAX_RANK];
    let start = MAX_RANK.saturating_sub(values.len());
    for (i, v) in values.iter().take(MAX_RANK).enumerate() {
        out[start + i] = *v as i32;
    }
    out
}

/// Runtime dispatch for [`einsum_forward`].
///
/// The extents and strides are fixed by the plan at lowering time, so the plan's
/// output shape is the only one the kernel can index correctly. The op therefore
/// reports that shape as the concrete output, and launches over the plan's element
/// count rather than whatever shape the executor resolved: a larger resolved shape
/// (a dynamic dimension filled in with the batch size, say) would otherwise run lanes
/// past the plan's end, and those compute operand offsets beyond the real tensors.
pub struct EinsumRuntimeOp<D: Num + Send + Sync + 'static> {
    block_size: i32,
    /// The plan's output shape, unclamped: what the kernel actually writes.
    out_dims: Vec<usize>,
    /// Elements in `out_dims`, the count the launch covers.
    n_elements: usize,
    k: i32,
    a_step: i32,
    b_step: i32,
    out_shape: [i32; MAX_RANK],
    a_strides: [i32; MAX_RANK],
    b_strides: [i32; MAX_RANK],
    _phantom: PhantomData<D>,
}

impl<D: Num + Send + Sync + 'static> EinsumRuntimeOp<D> {
    /// Builds the op from a plan.
    ///
    /// Returns `None` only if the plan's rank exceeds [`MAX_RANK`], which the planner
    /// already rejects; the check is kept so the padding cannot silently truncate.
    pub fn new(block_size: i32, plan: &EinsumPlan) -> Option<Self> {
        if plan.out_shape.len() > MAX_RANK {
            return None;
        }
        let shape: Vec<i64> = plan.out_shape.iter().map(|&d| d as i64).collect();
        Some(Self {
            block_size,
            out_dims: plan.out_shape.clone(),
            n_elements: plan.out_shape.iter().product(),
            k: plan.k as i32,
            a_step: plan.a_contract_stride as i32,
            b_step: plan.b_contract_stride as i32,
            // Output extents are clamped to at least 1 so the index peel's divisions
            // are defined even on lanes past the end, which compute an index that is
            // then never dereferenced.
            out_shape: pad_left(&shape.iter().map(|&d| d.max(1)).collect::<Vec<_>>(), 1),
            a_strides: pad_left(&plan.a_strides, 0),
            b_strides: pad_left(&plan.b_strides, 0),
            _phantom: PhantomData,
        })
    }

    /// The kernel's scalar arguments after the three pointers, in order:
    /// `n_elements`, `k`, the two contraction strides, then [`MAX_RANK`] output
    /// extents and the two [`MAX_RANK`] stride vectors. Exposed so callers and tests
    /// can see the mapping without going through an [`ArgVisitor`].
    pub fn kernel_scalars(&self, n_elements: i32) -> Vec<i32> {
        let mut v = Vec::with_capacity(4 + 3 * MAX_RANK);
        v.push(n_elements);
        v.push(self.k);
        v.push(self.a_step);
        v.push(self.b_step);
        v.extend_from_slice(&self.out_shape);
        v.extend_from_slice(&self.a_strides);
        v.extend_from_slice(&self.b_strides);
        v
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for EinsumRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    /// The plan's output shape, whatever the executor resolved, so the buffer it
    /// allocates is exactly the one the kernel writes.
    fn compute_concrete_output_shape(&self, _: &[&[usize]], _: &[usize]) -> Vec<usize> {
        self.out_dims.clone()
    }

    // The output shape is ignored here and in `grid` in favour of the plan's element
    // count; see the struct's documentation.
    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _: &[RawPtr],
        output: RawPtr,
        _: &[usize],
        _: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        for v in self.kernel_scalars(self.n_elements as i32) {
            visitor.visit_i32(v);
        }
    }

    fn grid(&self, _: &[usize]) -> [u32; 3] {
        [
            self.n_elements.div_ceil(self.block_size as usize) as u32,
            1,
            1,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{EinsumRuntimeOp, MAX_RANK};
    use teeny_core::device::program::ArgVisitor;
    use teeny_core::graph::einsum_plan;
    use teeny_core::model::{RawPtr, RuntimeOp};

    /// Records the `i32` scalars `pack_args` emits, and counts the pointers.
    #[derive(Default)]
    struct Scalars {
        ptrs: usize,
        i32s: Vec<i32>,
    }

    impl ArgVisitor for Scalars {
        fn visit_ptr(&mut self, _: *mut core::ffi::c_void) {
            self.ptrs += 1;
        }
        fn visit_i32(&mut self, val: i32) {
            self.i32s.push(val);
        }
        fn visit_bool(&mut self, _: bool) {
            unreachable!("einsum passes no bool")
        }
        fn visit_i8(&mut self, _: i8) {
            unreachable!("einsum passes no i8")
        }
        fn visit_i16(&mut self, _: i16) {
            unreachable!("einsum passes no i16")
        }
        fn visit_i64(&mut self, _: i64) {
            unreachable!("einsum passes no i64")
        }
        fn visit_u8(&mut self, _: u8) {
            unreachable!("einsum passes no u8")
        }
        fn visit_u16(&mut self, _: u16) {
            unreachable!("einsum passes no u16")
        }
        fn visit_u32(&mut self, _: u32) {
            unreachable!("einsum passes no u32")
        }
        fn visit_u64(&mut self, _: u64) {
            unreachable!("einsum passes no u64")
        }
        fn visit_f32(&mut self, _: f32) {
            unreachable!("einsum passes no f32")
        }
        fn visit_f64(&mut self, _: f64) {
            unreachable!("einsum passes no f64")
        }
    }

    fn op(equation: &str, a: &[usize], b: &[usize]) -> EinsumRuntimeOp<f32> {
        let plan = einsum_plan(equation, a, b).expect("plans");
        EinsumRuntimeOp::new(1024, &plan).expect("within MAX_RANK")
    }

    fn pack(op: &EinsumRuntimeOp<f32>, output_shape: &[usize]) -> Scalars {
        let null: RawPtr = core::ptr::null_mut();
        let (a, b) = ([2usize, 3], [3usize, 4]);
        let mut v = Scalars::default();
        op.pack_args(
            &[(null, &a), (null, &b)],
            &[],
            null,
            output_shape,
            0,
            &mut v,
        );
        v
    }

    /// The executor allocates what `compute_concrete_output_shape` returns, so it
    /// must be the plan's shape even when the resolved one differs.
    #[test]
    fn test_the_concrete_output_shape_is_the_plans() {
        let op = op("bij,bjk->bik", &[4, 2, 3], &[4, 3, 5]);
        assert_eq!(
            op.compute_concrete_output_shape(&[], &[4, 2, 5]),
            vec![4, 2, 5]
        );
        // A dynamic batch axis filled in with some other batch size.
        assert_eq!(
            op.compute_concrete_output_shape(&[], &[9, 2, 5]),
            vec![4, 2, 5]
        );
    }

    /// The launch covers the plan's elements, not the resolved shape's. Lanes past the
    /// plan's end would index the operands beyond their real extent, because the masks
    /// test against the element count passed here.
    #[test]
    fn test_launches_over_the_plans_elements_not_the_resolved_shape() {
        let op = op("ij,jk->ik", &[2, 3], &[3, 4]);
        let plan_elems = 2 * 4;

        for resolved in [vec![2, 4], vec![64, 4], vec![]] {
            let v = pack(&op, &resolved);
            assert_eq!(v.ptrs, 3);
            assert_eq!(v.i32s.len(), 4 + 3 * MAX_RANK);
            assert_eq!(v.i32s[0], plan_elems, "n_elements for {resolved:?}");
            assert_eq!(op.grid(&resolved), [1, 1, 1], "grid for {resolved:?}");
        }
    }

    /// The grid is the plan's element count in blocks, rounded up.
    #[test]
    fn test_the_grid_rounds_the_plans_elements_up_to_whole_blocks() {
        // 1 * 120 * 160 * 160 = 3,072,000 elements, 3000 blocks of 1024.
        let large = op("aijk,aijh->ajkh", &[1, 4, 120, 160], &[1, 4, 120, 160]);
        assert_eq!(large.grid(&[]), [3000, 1, 1]);
        // 2 * 5 = 10 elements, one partly filled block.
        let small = op("ij,jk->ik", &[2, 3], &[3, 5]);
        assert_eq!(small.grid(&[]), [1, 1, 1]);
    }
}
