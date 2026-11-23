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

//! The expert gather with NVFP4 weights unpacked in its inner loop.
//!
//! Same shape as [`super::expert_gather`]:
//!
//! ```text
//! y[s, n] = sum(k) x[token(s), k] * W[expert(s), n, k]
//! ```
//!
//! but `W` is never materialised as f32. Each weight is decoded where it is used:
//! an E2M1 nibble, an E4M3 scale per 16 weights, and one f32 per expert, exactly as
//! [`super::nvfp4`] does it and as the RVV kernel in `experiments/nvfp4-riscv`
//! measures it (1.9x to 3.1x over the unfused form on a SpacemiT X100).
//!
//! This is what makes the model fit at all: the experts are 20.2 GB packed and about
//! 160 GB in f32, against a board's 31 GB.
//!
//! **Packed, four bytes to an f32 element.** Parameter buffers take the graph's
//! dtype, so the packed nibbles travel inside f32 elements and are taken apart here:
//! `T::cast` bitcasts one to an i32 and the byte is masked out of it. That is
//! 0.5 bytes per weight, the true NVFP4 density.
//!
//! Which of the four bytes a lane wants varies along `k`, so all four are computed
//! and selected rather than branched on. There are no shifts: `Shr` is not lowered
//! by teenyc, and masking first is needed anyway because `Div` truncates towards
//! zero, so the top byte has to be masked before it is divided down and its sign bit
//! added back by hand.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gathered projection above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn expert_gather_nvfp4_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    b_ptr: InPtr<T::Pointer<D>>,
    s_ptr: InPtr<T::Pointer<D>>,
    s2_ptr: InPtr<T::Pointer<D>>,
    ids_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
    K: i32,
    TOP_K: i32,
    IDS_STRIDE: i32,
    IDS_OFFSET: i32,
    BWORDS: i32,
    SWORDS: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Add<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
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
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let eighth = T::full(&[BLOCK], D::from_f64(0.125));
    let seven = T::full(&[BLOCK], D::from_f64(7.0));
    let sub6 = T::full(&[BLOCK], D::from_f64(0.015_625));
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
        // The packed row of this (expert, n): K/8 words of four bytes each, and
        // K/64 words of four E4M3 scales.
        let brow = expert * (N * BWORDS) + (n * BWORDS);
        let srow = expert * (N * SWORDS) + (n * SWORDS);

        let wword = T::cast::<D, i32>(
            T::load(
                b_ptr.add_offsets(k / 8 + brow),
                Some(km),
                Some(zeros),
                &[],
                None,
                None,
                None,
                true,
            ),
            None,
            true,
        );
        // Byte (k/2) % 4 of that word. The index varies by lane, so all four are
        // computed and selected; the top byte is masked to seven bits and 128 added
        // back when the word is negative.
        let bsel = (k / 2) - ((k / 8) * 4);
        let byte = {
            let b0 = wword & 0xFF;
            let b1 = (wword & 0xFF00) / 256;
            let b2 = (wword & 0x00FF_0000) / 65536;
            let low7 = (wword & 0x7F00_0000) / 16_777_216;
            let b3 = T::where_(wword.lt(0), low7 + 128, low7 + 0);
            T::where_(
                bsel.eq(0),
                b0,
                T::where_(bsel.eq(1), b1, T::where_(bsel.eq(2), b2, b3)),
            )
        };
        let hi = byte / 16;
        let nib = T::where_((k - (k / 2) * 2).eq(1), hi, byte - hi * 16);
        let sgn = nib / 8;
        let rest = nib - sgn * 8;
        let e = rest / 2;
        let mant = T::cast::<i32, D>(rest - e * 2, None, false);
        let ef = T::cast::<i32, D>(e, None, false);
        let mag = T::where_(
            e.eq(0),
            mant * half,
            (one + mant * half) * T::exp2(ef - one),
        );
        let val = T::where_(sgn.eq(1), zeros - mag, mag);

        let sword = T::cast::<D, i32>(
            T::load(
                s_ptr.add_offsets(k / 64 + srow),
                Some(km),
                Some(zeros),
                &[],
                None,
                None,
                None,
                true,
            ),
            None,
            true,
        );
        let ssel = (k / 16) - ((k / 64) * 4);
        let sb = {
            let s0 = sword & 0xFF;
            let s1 = (sword & 0xFF00) / 256;
            let s2 = (sword & 0x00FF_0000) / 65536;
            let slow7 = (sword & 0x7F00_0000) / 16_777_216;
            let s3 = T::where_(sword.lt(0), slow7 + 128, slow7 + 0);
            T::where_(
                ssel.eq(0),
                s0,
                T::where_(ssel.eq(1), s1, T::where_(ssel.eq(2), s2, s3)),
            )
        };
        let ssgn = sb / 128;
        let srest = sb - ssgn * 128;
        let se = srest / 8;
        let smf = T::cast::<i32, D>(srest - se * 8, None, false);
        let sef = T::cast::<i32, D>(se, None, false);
        let sval = T::where_(
            se.eq(0),
            smf * eighth * sub6,
            (one + smf * eighth) * T::exp2(sef - seven),
        );
        let sval = T::where_(ssgn.eq(1), zeros - sval, sval);

        acc = acc + x * (val * sval);
    }
    // scale2 is one value per expert, so it multiplies the reduced sum once.
    let s2 = T::load(
        s2_ptr.add_offsets(expert),
        None,
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    // One scalar per program: reduce the lanes, then write a single element.
    let out = T::sum(acc * s2, Some(0), true);
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
pub struct ExpertGatherNvfp4Op<D: FloatBytes + Send + Sync + 'static> {
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

impl<D: FloatBytes + Send + Sync + 'static> ExpertGatherNvfp4Op<D> {
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

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ExpertGatherNvfp4Op<D> {
    fn name(&self) -> &str {
        "llm.expert_gather_nvfp4"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(|t| t * self.top_k), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ExpertGatherNvfp4Forward::<D>::new(BLOCK);
        Some((
            "expert_gather_nvfp4_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ExpertGatherNvfp4RuntimeOp::<D> {
                experts: self.experts,
                n: self.n,
                k: self.k,
                top_k: self.top_k,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`ExpertGatherNvfp4Op`].
pub struct ExpertGatherNvfp4RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    experts: usize,
    n: usize,
    k: usize,
    top_k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ExpertGatherNvfp4RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        // x and the router's packed row; the weights are parameters.
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![
            vec![self.experts, self.n, self.k.div_ceil(8)],
            vec![self.experts, self.n, (self.k / 16).div_ceil(4)],
            vec![self.experts],
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight", "scale", "scale2"]
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
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(params[2]);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(self.top_k as i32);
        visitor.visit_i32(ids_stride as i32);
        // The ids sit after the weights in the router's packed row.
        visitor.visit_i32((ids_stride - self.top_k) as i32);
        // Words per row, four bytes each. div_ceil, because a row shorter than
        // one word still needs one.
        visitor.visit_i32(self.k.div_ceil(8) as i32);
        visitor.visit_i32((self.k / 16).div_ceil(4) as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [self.n as u32, output_shape[0] as u32, 1]
    }
}
